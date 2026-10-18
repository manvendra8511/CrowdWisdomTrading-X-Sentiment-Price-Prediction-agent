"""Standalone charts, evaluation report, and one-page sentiment outlook."""
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


def charting():
    # Avoid writing font caches outside the project in restricted deployments.
    os.environ.setdefault('MPLCONFIGDIR', str(Path('/tmp') / 'cwt-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def number(value):
    return 'undefined' if value is None else f'{value:.4f}'


def render_reports(output, manifest, bars, posts, news, predictions, metrics, risk, daily, oos):
    output = Path(output)
    label = 'SYNTHETIC DEMO — not market evidence' if manifest['data_kind'] == 'synthetic' else 'Historical research'
    asset = manifest['asset']
    plt = charting()
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(bars.index, bars['close'], color='#126a91', label='Daily close')
    axes[0].set_ylabel('Price')
    axes[0].legend(loc='upper left')
    axes[1].plot(bars.index, bars['sentiment_mean'], color='#8c3eaa', alpha=0.5, label='Daily X sentiment')
    axes[1].plot(bars.index, bars['sentiment_mean'].rolling(5).mean(), color='#2c1840', label='5-session mean')
    axes[1].axhline(0, color='grey', linewidth=0.7)
    axes[1].set_ylabel('Sentiment')
    axes[1].legend(loc='upper left')
    axes[2].bar(bars.index, bars['post_count'], color='#238879', width=1)
    axes[2].set_ylabel('X posts')
    axes[2].set_xlabel('UTC close timestamp')
    fig.suptitle(f'{asset}: price, X sentiment and activity\n{label}')
    fig.tight_layout()
    fig.savefig(output / 'sentiment_price.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for h, ax in enumerate(axes.flat):
        rows = oos[oos['horizon'] == h]
        ax.plot(rows['target_timestamp'], rows['actual_price'], label='Actual', color='#126a91')
        ax.plot(rows['target_timestamp'], rows['predicted_price'], label='Predicted', color='#e68822')
        ax.set_title(f'n+{h} ({h + 1} session ahead)')
        ax.tick_params(axis='x', rotation=35, labelsize=7)
        ax.legend(fontsize=8)
    fig.suptitle(f'{asset}: out-of-sample forecast matrix\n{label}')
    fig.tight_layout()
    fig.savefig(output / 'forecast_matrix.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily['target_timestamp'], daily['equity'], color='#238879')
    ax.set_title(f'Out-of-sample next-open to close strategy, after costs\n{label}')
    ax.set_ylabel('Equity (initial capital = 1)')
    fig.tight_layout()
    fig.savefig(output / 'equity.png', dpi=160)
    plt.close(fig)

    forecasts = ['| Horizon | Target close (UTC) | Price | Residual 80% range |',
                 '|---|---|---:|---:|']
    for p in predictions:
        forecasts.append(f"| n+{p['horizon']} | {p['target_timestamp']} | {p['price']:.3f} | {p['lower_price']:.3f}–{p['upper_price']:.3f} |")
    table = ['| Horizon | Model | N | MAE | RMSE | MAPE % | Direction accuracy |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in metrics:
        table.append(f"| n+{row['horizon']} | {row['model']} | {row['samples']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['mape_pct']:.2f} | {row['direction_accuracy']:.3f} |")
    warning_text = '\n'.join('- ' + value for value in manifest['warnings']) or '- No automated coverage warnings.'
    report = f'''# {asset} evaluation report

**{label}**

Run: `{manifest['run_id']}`. Requested window: {manifest['window_start']} through
{manifest['as_of']}. Price information cutoff: {bars.index[-1].isoformat()}.

## Forecast definition and method

`n` is the first exchange session whose close is not yet observed at the price
information cutoff. Thus n+0 predicts the next unobserved close, and n+5 predicts
the sixth unobserved close. The last known close is a separate reference value,
not a claimed n+0 prediction. Forecasts use completed daily bars; intraday posts
after that cutoff appear in the outlook but not in model inputs.

The model is StandardScaler + Ridge (alpha={manifest['alpha']}) with six log-return
targets, converted back to prices. Inputs are only lagged price behavior and
fixed-lexicon X sentiment/activity. No fundamentals, news, or retrospectively
scraped engagement/follower counts enter historical features. Hyperparameters
are fixed before evaluation, with no tuning on the reported test periods.

## Walk-forward validation

Expanding training windows start with {manifest['min_train']} samples. Each test
block contains up to {manifest['test_size']} origins. Six rows are purged between
training origins and each test block, so every training label ends strictly
before the first test origin. The scaler is fit on each training fold only.
`folds.json` records these boundaries. Tests are chronological and disjoint.
Horizon labels overlap, so the forecast errors are not independent observations.

The final model is refit on all fully labeled rows, then predicts from the most
recent feature row. It does not generate the reported out-of-sample results.
Ranges use 10th/90th percentiles of earlier out-of-sample log-return residuals;
they are empirical uncertainty ranges, not guaranteed coverage intervals.

{chr(10).join(forecasts)}

## Out-of-sample errors and baselines

Persistence predicts the last observed close at every horizon. The price-only
ablation uses the same folds and estimator without sentiment/activity features.
Direction accuracy counts exact sign matches, including a flat prediction as zero.

{chr(10).join(table)}

## Risk metrics

- Sharpe ratio: {number(risk['sharpe_ratio'])}
- Sortino ratio: {number(risk['sortino_ratio'])}
- Maximum drawdown: {risk['max_drawdown']:.2%}
- Total strategy return: {risk['total_return']:.2%}
- Daily out-of-sample observations: {risk['observations']}

Strategy: sign of the n+0 predicted return decides long/short after the previous
close; enter at the next session's open and exit at its close. Charge
{risk['transaction_cost_bps_per_side']} bps per side for each nonzero position.
No overlapping multi-day bets are stacked. Sharpe uses sample standard deviation
and zero risk-free rate; Sortino uses the root mean square of negative returns
with a zero target. Annualization uses sqrt({risk['periods_per_year']}). Undefined
ratios are null in JSON. Drawdown includes the initial equity of 1, including a
loss on the first trade. Borrow costs, impact and financing are not modeled.

## Data quality and limitations

{warning_text}

A retrospective scrape cannot recover deleted/unavailable posts or guarantee
point-in-time text availability. Text is assumed available at its creation time;
engagement and follower snapshots are retained only for inspection. The fixed
lexicon handles some negation and financial vocabulary but is not a calibrated
financial NLP model. No accuracy or profitability is guaranteed by this run.
News is a current contextual supplement and never backfilled into historical
features. Raw closes are used throughout; inspect corporate actions before using
assets with splits. A 180-day window provides limited statistical evidence.

## Artifacts

- `manifest.json`: input hashes, actor/run IDs, versions, counts and assumptions (no tokens).
- `raw_prices.json`, `raw_posts.json`, `raw_news.json`: reproducible source snapshots.
- `features.csv`, `oos_predictions.csv`, `folds.json`, `metrics.json`: audit data.
- `forecast.json`, `model.joblib`: latest forecasts and fitted estimator.
- `sentiment_summary.md`, `sentiment_summary.pdf`: current one-page outlook.
- `sentiment_price.png`, `forecast_matrix.png`, `equity.png`: visual outputs.

![Sentiment and price](sentiment_price.png)
![Forecast matrix](forecast_matrix.png)
![Out-of-sample equity](equity.png)
'''
    (output / 'evaluation.md').write_text(report)

    cutoff = pd.Timestamp(manifest['as_of'])
    recent = posts[posts['created_at'] >= cutoff - pd.Timedelta(days=7)]
    prior = posts[(posts['created_at'] < cutoff - pd.Timedelta(days=7)) &
                  (posts['created_at'] >= cutoff - pd.Timedelta(days=14))]
    score = float(recent['sentiment'].mean()) if len(recent) else 0.0
    previous_score = float(prior['sentiment'].mean()) if len(prior) else None
    outlook = 'positive' if score >= 0.1 else 'negative' if score <= -0.1 else 'mixed/neutral'
    if not len(recent):
        outlook = 'insufficient recent posts'
    change = f'{score - previous_score:+.3f}' if previous_score is not None and len(recent) else 'unavailable'
    headlines = '\n'.join(f"- {row['title'][:130]} ({row['published_at'] or 'publication date unavailable'})\n  {row['url']}"
                          for row in news[:3]) or '- No dated relevant news returned.'
    summary = f'''# {asset}: current sentiment outlook

**{label}** | As of {manifest['as_of']}

The last seven calendar days contain {len(recent)} retrieved X posts. Their mean
sentiment is {score:+.3f}, giving a **{outlook}** outlook. Change from the previous
seven-day mean: {change}. Positive/negative labels use thresholds of +0.1/-0.1.
This summarizes the retrieved sample, not the full X population.

The latest completed close is {bars['close'].iloc[-1]:.3f} at {bars.index[-1].isoformat()}.
The model's n+0 close forecast is {predictions[0]['price']:.3f}; its n+5 forecast is
{predictions[-1]['price']:.3f}. These use information available at the last price
close; newer posts and headlines below are context only. See evaluation.md for
all six horizons, baselines, validation and residual uncertainty ranges.

## Recent news context

{headlines}

## Interpretation and limits

Post volume, simple lexicon sentiment and recent price behavior can disagree.
Engagement/follower counts are stored but excluded from historical predictions
because today's counts were not known at earlier closes. Recent news is not
historical training evidence. Lexicon scores are not calibrated probabilities;
sarcasm and ambiguous ticker references can be misclassified. Check data coverage
and the out-of-sample report before interpreting the outlook.

Coverage alerts: {len(manifest['warnings'])}. See evaluation.md for details.
'''
    (output / 'sentiment_summary.md').write_text(summary)
    # A literal single-page PDF companion, with bounded headline text.
    plain = summary.replace('**', '').replace('# ', '').replace('## ', '')
    lines = []
    for line in plain.splitlines():
        lines.extend(textwrap.wrap(line, width=100) or [''])
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.96, '\n'.join(lines), va='top', fontsize=min(9, 600 / max(len(lines), 1)),
             family='sans-serif')
    fig.savefig(output / 'sentiment_summary.pdf')
    plt.close(fig)
