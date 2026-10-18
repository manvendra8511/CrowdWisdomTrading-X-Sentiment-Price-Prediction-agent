# CrowdWisdomTrading X Sentiment Price Prediction Agent

A Python research pipeline that collects X posts, daily asset prices and recent
news; joins them in SQLite; builds price/sentiment features; and forecasts six
future session closes using purged walk-forward validation. A separate FastAPI
service supports individual and batch text sentiment requests.

See the [original assessment](docs/assignment.pdf) and
[requirement-by-requirement gap analysis](docs/assessment-gap-analysis.md).

## Install and test

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
```

## Run the complete offline demo

```bash
python scripts/forecast.py --demo --asset SPY --keyword 'S&P 500'
```

This runs all stages without network calls or credentials. Outputs go to a new
`reports/demo/SPY-<run-id>/` directory; the CLI prints the full location.
**All demo inputs and performance results are synthetic**, including news.
To reproduce a run, supply the same timezone-aware `--as-of` and `--seed 42`.

## Run with live Apify and Tavily data

Copy `.env.example` to `.env` and fill in credentials locally. The CLI loads this
file; existing environment variables take precedence. Tokens are not written to
reports or manifests.

```bash
cp .env.example .env
# Edit .env with your account credentials.
python scripts/forecast.py --live --asset SPY --keyword 'S&P 500' --days 180
```

Live mode uses account credits. Default adapters are:

- X: `apidojo/tweet-scraper`, with weekly `searchTerms` windows and a configurable
  `--max-posts-per-window` cap (default 1000). Capped windows are flagged as incomplete.
- Price: `logiover/yahoo-finance-scraper`, historical daily rows. It requests a
  year and filters to the same 180-day window as posts. Actor IDs are configurable
  for actors with compatible input/output schemas.
- News: Tavily `/search`, topic `news`, last week, with source links and publication
  timestamps when supplied. The report discloses unknown publication dates.

You may supply `--prices path/to/prices.csv` to replace only live price scraping.
Live integrations have mocked HTTP coverage but require your credentials for a
real 180-day run. Do not assume an actor's search result cap means complete coverage.

## Import existing data

```bash
python scripts/forecast.py --import-data --asset SPY \
  --prices prices.csv --posts posts.json --news news.json \
  --as-of 2026-06-30T23:00:00Z
```

- Prices: JSON array or CSV with `date`, `open`, `high`, `low`, `close`, optional
  `symbol`. `date` is the exchange session date (`YYYY-MM-DD`). Prices must have
  consistent raw OHLC, no duplicates, and no missing sessions inside their range.
  Adjusted close is not mixed with raw OHLC. Incomplete current bars are excluded.
- Posts: JSON array or CSV with `id`, `text`, `createdAt` (timezone required),
  `likeCount`, `retweetCount`, `viewCount`, and optional `author.followers` in JSON.
  Canonical `created_at`, `likes`, `reposts`, `views`, `followers` are also accepted.
  Repeated IDs are deduplicated. Unparseable records fail explicitly.
- News: a Tavily results array with `url`, `title`, `content`, `published_date`.
  Omit `--news` only if accepting an incomplete news deliverable; this is reported.

Use `--calendar XNYS` for US equities/ETFs and `--calendar 24/7` for crypto.
The calendar handles DST, holidays and early closes. Only daily bars are supported.
Run `python scripts/forecast.py --help` for model, cost, storage and output options.

## Forecast and validation conventions

`n` is the next session whose close has not yet been observed. The six outputs
`n+0` through `n+5` predict one through six sessions after the last completed bar.
The observed reference close is reported separately. Intraday posts after the last
completed bar appear in the current outlook but not in the model's feature row.

The estimator is StandardScaler + multi-output Ridge with fixed alpha. It uses
lagged returns, volatility, price ranges and X sentiment/activity only. Training
expands chronologically, with a six-row purge so all training labels end strictly
before the test block. Scaling is fit on training data only. Evaluation includes
persistence and price-only baselines, MAE, RMSE, MAPE and directional accuracy.

Sharpe, Sortino and drawdown use only out-of-sample n+0 decisions: enter at the next
open, exit at that close, deduct the configured per-side transaction cost. The
report explains annualization, zero-risk-free/target assumptions, uncertainty
ranges, small-sample limitations, and excluded borrowing/impact costs.

Historical engagement snapshots and current news are not used in historical
features. Fixed lexicon sentiment is interpretable but not a calibrated financial
NLP model. Retrospective X coverage and corporate actions require review.

## Generated artifacts

| File | Purpose |
|---|---|
| `evaluation.md`, `metrics.json` | Out-of-sample accuracy, baselines, risk metrics and methodology |
| `sentiment_summary.md`, `sentiment_summary.pdf` | One-page current sentiment/news outlook |
| `forecast.json` | Six target timestamps, price estimates and empirical residual ranges |
| `sentiment_price.png` | Price versus sentiment and post volume over time |
| `forecast_matrix.png` | Six-panel actual versus OOS predicted prices |
| `equity.png`, `strategy_returns.csv` | Cost-adjusted strategy history |
| `manifest.json`, `folds.json` | Provenance, hashes, versions, coverage and validation boundaries |
| `raw_*.json`, `joined_bars.csv`, `features.csv`, `oos_predictions.csv` | Reproducible input/output audit trail |
| `model.joblib` | Fitted final model and feature schema; load only trusted artifacts |

Pipeline SQLite data defaults to `data/forecast-<mode>.db`. Separate demo/live stores
and run IDs prevent accidental dataset mixing. Reports use new run directories;
explicit output directories containing a prior manifest are not overwritten.

## Existing sentiment API

```bash
python src/agent.py
# Or: uvicorn src.api.routes:app --host 0.0.0.0 --port 8000
```

- `GET /api/v1/health`
- `POST /api/v1/analyze`
- `POST /api/v1/analyze/batch`
- `GET /api/v1/analyze/history`
- Interactive API documentation: `/docs`

Configuration is in `configs/config.yaml`, with environment overrides. Resources
are shared for the application lifetime. The text API combines rules and ML;
its six-example fallback classifier is a demonstration model. The forecasting
pipeline uses its own fixed rule scorer and price-regression model as described above.

## Docker

```bash
docker compose up --build
# Run the offline pipeline in the same image:
docker compose run --rm sentiment-agent python scripts/forecast.py --demo
```

Compose mounts data, logs, configs and reports. To run live ingestion in Docker,
pass credentials with `docker run --env-file .env` or configure Compose environment
variables explicitly. The image does not include your `.env` file.

Project modules and data flow are documented in [flow.md](flow.md); modeling
choices and limitations are documented in [why.md](why.md).
