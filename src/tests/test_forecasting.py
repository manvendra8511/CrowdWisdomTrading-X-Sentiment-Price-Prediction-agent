"""Assignment acceptance tests, especially point-in-time and OOS correctness."""
import json
import sqlite3

import httpx
import numpy as np
import pandas as pd
import pytest

from src.forecasting.data import normalize_posts, normalize_prices, schedule, utc
from src.forecasting.demo import make_demo
from src.forecasting.features import build_features, build_targets, FEATURES
from src.forecasting.model import evaluate, risk_metrics, walk_forward
from src.forecasting.pipeline import run_pipeline
from src.forecasting.providers import Apify, fetch_news
from src.forecasting.storage import ForecastStore


START = pd.Timestamp('2026-01-01T00:00:00Z')
END = pd.Timestamp('2026-06-30T23:00:00Z')


def research_bars():
    prices, _, _ = make_demo('SPY', START, END)
    bars = normalize_prices(prices, 'SPY', START, END).set_index('timestamp')
    bars['post_count'] = np.arange(len(bars)) % 10
    bars['sentiment_mean'] = np.sin(np.arange(len(bars)) / 7) * 0.3
    bars['positive_share'] = 0.5 + bars['sentiment_mean']
    bars['negative_share'] = 0.5 - bars['sentiment_mean']
    return bars


def test_calendar_dst_holiday_and_early_close():
    sessions = schedule('2026-03-06', '2026-03-09')
    assert sessions.loc['2026-03-06', 'close'].hour == 21
    assert sessions.loc['2026-03-09', 'close'].hour == 20
    november = schedule('2026-11-25', '2026-11-30')
    assert '2026-11-26' not in november.index.strftime('%Y-%m-%d')
    assert november.loc['2026-11-27', 'close'].hour == 18
    with pytest.raises(ValueError, match='timezone'):
        utc('2026-01-01 12:00:00')


def test_prices_exclude_unfinished_bars_and_reject_missing_or_duplicate_sessions():
    records, _, _ = make_demo('SPY', START, END)
    normalized = normalize_prices(records, 'SPY', START, END)
    assert len(normalized) > 100
    incomplete_end = normalized['timestamp'].iloc[-1] - pd.Timedelta(minutes=1)
    assert len(normalize_prices(records, 'SPY', START, incomplete_end)) == len(normalized) - 1
    with pytest.raises(ValueError, match='Duplicate'):
        normalize_prices(records + [records[0]], 'SPY', START, END)
    with pytest.raises(ValueError, match='Missing exchange'):
        normalize_prices(records[:10] + records[11:], 'SPY', START, END)
    with pytest.raises(ValueError, match='Nonpositive'):
        normalize_prices([{**records[0], 'close': 0}], 'SPY', START, END)


def test_posts_are_deduplicated_and_keep_engagement_fields():
    item = {'id': '1', 'text': '$SPY bullish', 'createdAt': 'Mon Mar 09 15:00:00 +0000 2026',
            'likeCount': 3, 'retweetCount': 2, 'viewCount': 10, 'author': {'followers': 12}}
    posts = normalize_posts([item, item], 'SPY', START, END)
    assert len(posts) == 1
    assert posts.iloc[0][['likes', 'reposts', 'views', 'followers']].tolist() == [3, 2, 10, 12]
    assert posts.iloc[0]['created_at'].tzinfo is not None


def test_database_join_does_not_assign_after_close_posts_to_previous_bar(tmp_path):
    bars = research_bars().iloc[:3].reset_index()
    first, second = bars['timestamp'].iloc[:2]
    records = [
        {'id': 'before', 'createdAt': (first - pd.Timedelta(minutes=1)).isoformat(), 'text': 'good'},
        {'id': 'after', 'createdAt': (first + pd.Timedelta(minutes=1)).isoformat(), 'text': 'bad'},
        {'id': 'future', 'createdAt': (bars['timestamp'].iloc[-1] + pd.Timedelta(minutes=1)).isoformat(), 'text': 'good'},
    ]
    posts = normalize_posts(records, 'SPY', START, END)
    posts['sentiment'] = [1., -1., 1.]
    store = ForecastStore(tmp_path / 'test.db')
    try:
        store.ingest('run', {'asset': 'SPY', 'mode': 'demo', 'as_of': END.isoformat()}, bars, posts, [])
        joined = store.joined_bars('run')
        assert joined.loc[first, 'sentiment_mean'] == 1
        assert joined.loc[second, 'sentiment_mean'] == -1
        assert joined.iloc[-1]['post_count'] == 0
        assert joined.iloc[-1]['sentiment_mean'] == 0
    finally:
        store.close()


def test_features_are_causal_and_exclude_snapshots():
    bars = research_bars()
    original = build_features(bars)
    changed = bars.copy()
    pivot = bars.index[70]
    changed.loc[pivot:, 'close'] *= 2
    changed.loc[pivot:, 'sentiment_mean'] = -1
    changed['followers'] = 9999999
    changed['news_sentiment'] = 1
    modified = build_features(changed)
    pd.testing.assert_frame_equal(original.loc[original.index < pivot], modified.loc[modified.index < pivot])
    assert list(original.columns) == FEATURES
    assert not any('followers' in c or 'news' in c or 'views' in c for c in original.columns)
    targets = build_targets(bars, original.index)
    first = bars.index.get_loc(original.index[0])
    assert targets.iloc[0, 0] == pytest.approx(np.log(bars['close'].iloc[first + 1] / bars['close'].iloc[first]))
    assert targets.iloc[0, 5] == pytest.approx(np.log(bars['close'].iloc[first + 6] / bars['close'].iloc[first]))


def test_walk_forward_labels_do_not_cross_test_boundary_and_future_changes_do_not_affect_first_fold():
    bars = research_bars()
    _, original, folds, _ = walk_forward(build_features(bars), bars, min_train=30)
    assert len(folds) >= 2
    for fold in folds:
        assert pd.Timestamp(fold['train_label_end']) < pd.Timestamp(fold['test_start'])
    assert not original.duplicated(['origin', 'horizon']).any()
    changed = bars.copy()
    changed.loc[pd.Timestamp(folds[1]['test_start']):, 'close'] *= 1.2
    _, other, _, _ = walk_forward(build_features(changed), changed, min_train=30)
    np.testing.assert_allclose(original[original.fold == 0]['predicted_price'],
                               other[other.fold == 0]['predicted_price'])


def test_risk_metrics_include_initial_loss_and_handle_zero_denominators():
    assert risk_metrics([-0.1, 0.0])['max_drawdown'] == pytest.approx(-0.1)
    assert risk_metrics([0.0, 0.0])['sharpe_ratio'] is None
    assert risk_metrics([0.01, 0.01])['sortino_ratio'] is None
    with pytest.raises(ValueError):
        risk_metrics([np.nan])


def test_strategy_uses_next_open_and_charges_both_sides():
    bars = research_bars()
    _, oos, _, _ = walk_forward(build_features(bars), bars, min_train=30)
    metrics, risk, daily = evaluate(oos, transaction_cost_bps=5)
    assert len(metrics) == 18  # Six horizons, full model and two baselines.
    expected = daily['position'] * (daily['actual_price'] / daily['next_open'] - 1) - daily['position'].abs() * 0.001
    np.testing.assert_allclose(daily['strategy_return'], expected)
    assert risk['observations'] == len(daily)


def test_apify_auth_polling_and_dataset_pagination():
    seen = []
    def handler(request):
        seen.append(request)
        assert request.headers['Authorization'] == 'Bearer test-token'
        assert 'token=' not in str(request.url)
        if request.method == 'POST':
            return httpx.Response(201, json={'data': {'id': 'r', 'status': 'RUNNING'}})
        if '/actor-runs/' in request.url.path:
            return httpx.Response(200, json={'data': {'id': 'r', 'status': 'SUCCEEDED', 'defaultDatasetId': 'd'}})
        offset = int(request.url.params.get('offset', 0))
        return httpx.Response(200, json=[{'id': i} for i in range(1000)] if offset == 0 else [{'id': 1000}])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = Apify('test-token', client)
        assert len(api.run('user/actor', {'query': 'SPY'})) == 1001
        assert api.runs[0]['actor'] == 'user/actor'
    assert len(seen) == 4


def test_apify_failed_actor_is_not_accepted():
    def handler(request):
        return httpx.Response(201, json={'data': {'id': 'r', 'status': 'FAILED'}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match='FAILED'):
            Apify('test-token', client).run('user/actor', {})


def test_weekly_queries_cover_requested_window_and_flag_caps(monkeypatch):
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        api = Apify('test-token', client)
        payloads = []
        def run(actor, payload):
            payloads.append(payload)
            return [{'id': str(len(payloads))}]
        monkeypatch.setattr(api, 'run', run)
        records, warnings = api.posts('user/actor', 'SPY', ['S&P 500'], START, START + pd.Timedelta(days=15), max_items=1)
        assert len(payloads) == 3
        assert 'since:2026-01-01' in payloads[0]['searchTerms'][0]
        assert '$SPY' in payloads[0]['searchTerms'][0]
        assert '"S&P 500"' in payloads[0]['searchTerms'][0]
        assert 'until:2026-01-17' in payloads[-1]['searchTerms'][0]
        assert len(records) == len(warnings) == 3


def test_tavily_request_is_recent_news():
    def handler(request):
        payload = json.loads(request.content)
        assert payload['topic'] == 'news' and payload['time_range'] == 'week'
        assert request.headers['Authorization'] == 'Bearer test-token'
        return httpx.Response(200, json={'results': [{'title': 'test'}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert fetch_news('test-token', 'SPY', client) == [{'title': 'test'}]


def test_live_missing_credentials_fails_before_network(monkeypatch, tmp_path):
    monkeypatch.delenv('APIFY_TOKEN', raising=False)
    monkeypatch.delenv('TAVILY_API_KEY', raising=False)
    with pytest.raises(ValueError, match='requires APIFY_TOKEN'):
        run_pipeline(mode='live', output=tmp_path)


def test_end_to_end_demo_and_reimport_are_reproducible(tmp_path):
    result = run_pipeline(mode='demo', as_of=END.isoformat(), output=tmp_path / 'demo', database=tmp_path / 'demo.db')
    output = tmp_path / 'demo'
    for filename in ['forecast.json', 'evaluation.md', 'sentiment_summary.md', 'sentiment_summary.pdf',
                     'sentiment_price.png', 'forecast_matrix.png', 'equity.png', 'oos_predictions.csv',
                     'features.csv', 'folds.json', 'metrics.json', 'manifest.json', 'model.joblib']:
        assert (output / filename).stat().st_size > 0
    forecast = json.loads((output / 'forecast.json').read_text())
    assert [p['horizon'] for p in forecast['predictions']] == list(range(6))
    assert all(p['price'] > 0 and p['lower_price'] <= p['upper_price'] for p in forecast['predictions'])
    assert forecast['mode'] == 'demo'
    assert 'SYNTHETIC DEMO' in (output / 'evaluation.md').read_text()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert result['counts']['posts'] > 100
    with sqlite3.connect(tmp_path / 'demo.db') as connection:
        assert connection.execute('SELECT COUNT(*) FROM forecast_outputs').fetchone()[0] == 6
        assert connection.execute('SELECT COUNT(*) FROM asset_news').fetchone()[0] == 1
    run_pipeline(mode='import', as_of=END.isoformat(), output=tmp_path / 'import', database=tmp_path / 'import.db',
                 prices_path=output / 'raw_prices.json', posts_path=output / 'raw_posts.json', news_path=output / 'raw_news.json')
    imported = json.loads((tmp_path / 'import/forecast.json').read_text())
    assert forecast['predictions'] == imported['predictions']
    assert imported['data_kind'] == 'synthetic'
    assert 'SYNTHETIC DEMO' in (tmp_path / 'import/evaluation.md').read_text()
