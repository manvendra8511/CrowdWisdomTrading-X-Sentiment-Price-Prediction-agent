"""One reproducible execution from raw inputs to submission artifacts."""
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import uuid

import joblib
import numpy as np
import pandas as pd

from src.core.sentiment.rule_based import RuleBasedAnalyzer
from src.storage.repository import Repository
from .data import (utc, schedule, normalize_prices, normalize_posts, normalize_news,
                   read_records, write_json, json_default)
from .demo import make_demo
from .features import build_features, FEATURES
from .model import walk_forward, evaluate
from .providers import Apify, fetch_news
from .reporting import render_reports
from .storage import ForecastStore

ROOT = Path(__file__).resolve().parents[2]


async def score_posts(posts):
    """Frozen rule scorer; no future-labeled text is used to fit sentiment."""
    repository = Repository({'path': ':memory:'})
    await repository.initialize()
    rule = RuleBasedAnalyzer({'lexicon_path': str(ROOT / 'data/lexicons')}, repository)
    try:
        await rule.initialize()
        # Keep additions versioned in source and fixed for all validation folds.
        rule.positive_lexicon.update({'bullish', 'outperform', 'upside', 'rally', 'growth', 'gains', 'strong'})
        rule.negative_lexicon.update({'bearish', 'underperform', 'downside', 'selloff', 'losses', 'weak', 'downgrade'})
        result = posts.copy()
        result['sentiment'] = [(await rule.analyze(text))['score'] for text in result['text']]
        return result
    finally:
        await rule.shutdown()
        await repository.shutdown()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default).encode()).hexdigest()


def run_pipeline(*, asset='SPY', mode='demo', as_of=None, days=180, calendar='XNYS',
                 prices_path=None, posts_path=None, news_path=None, output=None, database=None,
                 keywords=None, min_train=40, test_size=10, alpha=10.0, cost_bps=5.0,
                 max_posts_per_window=1000, seed=42):
    asset = asset.upper().strip()
    if not asset or not all(c.isalnum() or c in '.-^=' for c in asset):
        raise ValueError('Asset must be a ticker symbol')
    if days < 90 or days > 3660 or max_posts_per_window < 1:
        raise ValueError('Use 90..3660 history days and a positive post cap')
    end = utc(as_of) if as_of else pd.Timestamp.now(tz='UTC')
    if end > pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=1):
        raise ValueError('as_of cannot be in the future')
    if min_train < 20 or test_size < 1 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Use min_train >= 20, test_size >= 1, and alpha > 0')
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 1000:
        raise ValueError('Cost must be finite and in [0, 1000) basis points per side')
    start = end - pd.Timedelta(days=days)
    run_id = str(uuid.uuid4())
    output = Path(output) if output else ROOT / 'reports' / mode / f'{asset}-{run_id[:8]}'
    if (output / 'manifest.json').exists():
        raise ValueError(f'{output} already contains a run; choose a new output directory')
    database = database or ROOT / 'data' / f'forecast-{mode}.db'
    warnings, actor_runs = [], []
    keywords = keywords or []
    if mode == 'demo':
        raw_prices, raw_posts, raw_news = make_demo(asset, start, end, calendar, seed)
        warnings.append('SYNTHETIC DEMO: inputs and results are generated test data, not market evidence.')
    elif mode == 'import':
        if not prices_path or not posts_path:
            raise ValueError('Import mode requires --prices and --posts')
        raw_prices, raw_posts = read_records(prices_path), read_records(posts_path)
        raw_news = read_records(news_path) if news_path else []
        if not news_path:
            warnings.append('No news snapshot provided; current news requirement needs a Tavily live run or news import.')
    elif mode == 'live':
        # Validate credentials before starting any remote runs.
        if not os.getenv('APIFY_TOKEN') or not os.getenv('TAVILY_API_KEY'):
            raise ValueError('Live mode requires APIFY_TOKEN and TAVILY_API_KEY in the environment or .env')
        if end < pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1):
            raise ValueError('Historical as-of runs require imported news snapshots; live mode is for current data')
        apify = Apify(os.environ['APIFY_TOKEN'])
        try:
            x_actor = os.getenv('APIFY_X_ACTOR_ID', 'apidojo/tweet-scraper')
            raw_posts, truncated = apify.posts(x_actor, asset, keywords, start, end, max_posts_per_window)
            if truncated:
                warnings.append('Post cap reached in windows: ' + ', '.join(truncated) + '; coverage is incomplete.')
            if prices_path:
                raw_prices = read_records(prices_path)
            else:
                price_actor = os.getenv('APIFY_PRICE_ACTOR_ID', 'logiover/yahoo-finance-scraper')
                raw_prices = apify.run(price_actor, {
                    'mode': 'historical', 'symbols': [asset], 'range': '1y' if days <= 365 else 'max',
                    'interval': '1d',
                })
            actor_runs = apify.runs
        finally:
            apify.close()
        raw_news = fetch_news(os.environ['TAVILY_API_KEY'], f'{asset} {" ".join(keywords)} latest market news')
    else:
        raise ValueError('mode must be demo, import, or live')

    synthetic = mode == 'demo' or any(row.get('_synthetic', False) for row in [*raw_prices, *raw_posts, *raw_news])
    if synthetic and mode != 'demo':
        warnings.append('SYNTHETIC DEMO: imported inputs contain generated test records; results are not market evidence.')
    prices = normalize_prices(raw_prices, asset, start, end, calendar)
    posts = normalize_posts(raw_posts, asset, start, end)
    news = normalize_news(raw_news, end)
    if not news:
        warnings.append('No recent news survived normalization; news context is unavailable.')
    if any(row['published_at'] is None for row in news):
        warnings.append('Some news results have no publication timestamp; their recency is unverified.')
    expected = schedule(start.date(), end.date(), calendar)
    expected = expected[(expected['close'] >= start) & (expected['close'] <= end)]
    if len(prices) != len(expected):
        warnings.append(f'Price coverage: {len(prices)}/{len(expected)} expected completed sessions.')
    if len(expected) and prices['timestamp'].iloc[-1] != expected['close'].iloc[-1]:
        raise ValueError('Latest completed price session is missing; import current data before forecasting')
    if posts['created_at'].min() > start + pd.Timedelta(days=7):
        warnings.append('X history starts more than a week after the requested window start.')
    if posts['created_at'].max() < end - pd.Timedelta(days=7):
        warnings.append('No X posts in the last seven days; sentiment outlook is stale.')
    if (prices['close'].pct_change().abs() > 0.3).any():
        warnings.append('A daily price move exceeds 30%; inspect split/corporate-action consistency.')
    posts = asyncio.run(score_posts(posts))
    manifest = {
        'run_id': run_id, 'status': 'running', 'asset': asset, 'mode': mode, 'calendar': calendar, 'period': 'daily',
        'as_of': end.isoformat(), 'window_start': start.isoformat(), 'history_days': days,
        'data_kind': 'synthetic' if synthetic else 'provider_records',
        'retrieved_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'information_cutoff': prices['timestamp'].iloc[-1].isoformat(),
        'forecast_definition': 'n is the next unobserved session; n+0..n+5 are 1..6 sessions ahead',
        'min_train': min_train, 'test_size': test_size, 'alpha': alpha, 'seed': seed,
        'transaction_cost_bps_per_side': cost_bps, 'keywords': keywords,
        'max_posts_per_window': max_posts_per_window, 'actors': actor_runs,
        'counts': {'prices': len(prices), 'posts': len(posts), 'news': len(news)},
        'raw_input_sha256': {'prices': digest(raw_prices), 'posts': digest(raw_posts), 'news': digest(raw_news)},
        'sentiment_method': 'fixed rule lexicon + versioned finance additions v1; no fitted text classifier',
        'feature_columns': FEATURES, 'warnings': warnings, 'python': platform.python_version(),
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [*Path(__file__).parent.glob('*.py'),
                                       * (ROOT / 'data/lexicons/en').glob('*.txt')]},
        'packages': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'pandas', 'scikit-learn', 'exchange-calendars', 'httpx', 'matplotlib')},
    }
    # Keep raw input snapshots even if validation/training later fails.
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('prices', raw_prices), ('posts', raw_posts), ('news', raw_news)]:
        write_json(output / f'raw_{name}.json', value)
    write_json(output / 'manifest.json', manifest)
    store = ForecastStore(database)
    try:
        store.ingest(run_id, manifest, prices, posts, news)
        bars = store.joined_bars(run_id)
        features = build_features(bars)
        if features.empty:
            raise ValueError('Not enough completed daily bars to build rolling features')
        model, oos, folds, forecast_returns = walk_forward(features, bars, min_train, test_size, alpha)
        metrics, risk, daily = evaluate(oos, cost_bps, 365 if calendar == '24/7' else 252)
        future = schedule(end.date(), (end + pd.Timedelta(days=40)).date(), calendar)
        future = future[future['close'] > prices['timestamp'].iloc[-1]].iloc[:6]
        if len(future) != 6:
            raise ValueError('Unable to resolve six future exchange sessions')
        predictions = []
        reference = float(prices['close'].iloc[-1])
        for h, target_time in enumerate(future['close']):
            residuals = oos.loc[oos['horizon'] == h, 'actual_log_return'] - oos.loc[oos['horizon'] == h, 'predicted_log_return']
            low, high = np.quantile(residuals, [0.1, 0.9])
            values = reference * np.exp([forecast_returns[h], forecast_returns[h] + low, forecast_returns[h] + high])
            if not np.isfinite(values).all():
                raise ValueError('Model produced nonfinite forecasts')
            predictions.append({'horizon': h, 'label': f'n+{h}', 'target_timestamp': target_time.isoformat(),
                                'price': float(values[0]), 'lower_price': float(values[1]),
                                'upper_price': float(values[2]), 'residual_samples': len(residuals)})
        store.save_outputs(run_id, features, predictions)
        features.to_csv(output / 'features.csv')
        bars.to_csv(output / 'joined_bars.csv')
        oos.to_csv(output / 'oos_predictions.csv', index=False)
        daily.to_csv(output / 'strategy_returns.csv', index=False)
        write_json(output / 'folds.json', folds)
        write_json(output / 'metrics.json', {'forecast_metrics': metrics, 'strategy': risk})
        write_json(output / 'forecast.json', {'asset': asset, 'mode': mode, 'run_id': run_id,
            'information_cutoff': manifest['information_cutoff'], 'reference_price': reference,
            'data_kind': manifest['data_kind'],
            'forecast_definition': manifest['forecast_definition'], 'predictions': predictions})
        render_reports(output, manifest, bars, posts, news, predictions, metrics, risk, daily, oos)
        manifest['status'] = 'complete'
        manifest['fold_count'] = len(folds)
        write_json(output / 'manifest.json', manifest)
        joblib.dump({'model': model, 'feature_columns': FEATURES, 'manifest': manifest}, output / 'model.joblib')
        with store.db:
            store.db.execute('UPDATE forecast_runs SET manifest=? WHERE run_id=?',
                (json.dumps(manifest, default=json_default, allow_nan=False), run_id))
        return {'run_id': run_id, 'output': str(output.resolve()), 'mode': mode,
                'counts': manifest['counts'], 'folds': len(folds), 'risk': risk}
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error_type'] = type(exc).__name__
        write_json(output / 'manifest.json', manifest)
        raise
    finally:
        store.close()
