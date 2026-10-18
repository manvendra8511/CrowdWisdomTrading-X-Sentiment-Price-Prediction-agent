#!/usr/bin/env python3
"""Run the assessment pipeline in demo, import, or live mode."""
import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--demo', action='store_true', help='Use clearly labeled synthetic inputs; no network')
    modes.add_argument('--live', action='store_true', help='Run configured Apify actors and Tavily (uses account credits)')
    modes.add_argument('--import-data', action='store_true', help='Use local provider snapshots')
    parser.add_argument('--asset', default='SPY')
    parser.add_argument('--calendar', default='XNYS', help='Exchange calendar; use 24/7 for crypto')
    parser.add_argument('--as-of', help='Timezone-aware ISO timestamp; defaults to now')
    parser.add_argument('--days', type=int, default=180)
    parser.add_argument('--keyword', action='append', default=[])
    parser.add_argument('--prices', type=Path, help='JSON/CSV daily OHLC input; optionally replaces live price scraping')
    parser.add_argument('--posts', type=Path, help='JSON/CSV X records for import mode')
    parser.add_argument('--news', type=Path, help='Tavily results JSON array for import mode')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--database', type=Path)
    parser.add_argument('--min-train', type=int, default=40)
    parser.add_argument('--test-size', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--cost-bps', type=float, default=5)
    parser.add_argument('--max-posts-per-window', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    from src.forecasting.pipeline import run_pipeline
    try:
        result = run_pipeline(asset=args.asset, mode='demo' if args.demo else 'live' if args.live else 'import',
            as_of=args.as_of, days=args.days, calendar=args.calendar,
            prices_path=args.prices, posts_path=args.posts, news_path=args.news,
            output=args.output, database=args.database, keywords=args.keyword,
            min_train=args.min_train, test_size=args.test_size, alpha=args.alpha,
            cost_bps=args.cost_bps, max_posts_per_window=args.max_posts_per_window, seed=args.seed)
    except Exception as exc:
        # Do not include HTTP request headers, bodies or tokens in CLI errors.
        logging.error('Pipeline failed (%s): %s', type(exc).__name__, str(exc))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
