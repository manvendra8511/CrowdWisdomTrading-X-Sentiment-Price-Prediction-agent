"""SQLite snapshots preserve exactly which data produced each forecast run."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .data import json_default


class ForecastStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS forecast_runs (
                run_id TEXT PRIMARY KEY, asset TEXT NOT NULL, mode TEXT NOT NULL,
                as_of TEXT NOT NULL, manifest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS price_bars (
                run_id TEXT REFERENCES forecast_runs(run_id), timestamp TEXT NOT NULL,
                open_timestamp TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL,
                PRIMARY KEY(run_id, timestamp));
            CREATE TABLE IF NOT EXISTS x_posts (
                run_id TEXT REFERENCES forecast_runs(run_id), post_id TEXT NOT NULL,
                created_at TEXT NOT NULL, text TEXT NOT NULL, likes INTEGER, reposts INTEGER,
                views INTEGER, followers INTEGER, sentiment REAL NOT NULL, bar_end TEXT,
                PRIMARY KEY(run_id, post_id));
            CREATE INDEX IF NOT EXISTS x_posts_bar ON x_posts(run_id, bar_end);
            CREATE TABLE IF NOT EXISTS asset_news (
                run_id TEXT REFERENCES forecast_runs(run_id), url TEXT, title TEXT,
                content TEXT, published_at TEXT, PRIMARY KEY(run_id, url));
            CREATE TABLE IF NOT EXISTS forecast_features (
                run_id TEXT REFERENCES forecast_runs(run_id), timestamp TEXT, values_json TEXT,
                PRIMARY KEY(run_id, timestamp));
            CREATE TABLE IF NOT EXISTS forecast_outputs (
                run_id TEXT REFERENCES forecast_runs(run_id), horizon INTEGER,
                target_timestamp TEXT, price REAL, lower_price REAL, upper_price REAL,
                PRIMARY KEY(run_id, horizon));
        ''')

    def close(self):
        self.db.close()

    def ingest(self, run_id, manifest, prices, posts, news):
        closes = pd.DatetimeIndex(prices['timestamp'])
        positions = closes.searchsorted(pd.DatetimeIndex(posts['created_at']), side='left')
        posts = posts.copy()
        posts['bar_end'] = [closes[i].isoformat() if i < len(closes) else None for i in positions]
        with self.db:
            self.db.execute('INSERT INTO forecast_runs VALUES (?, ?, ?, ?, ?)',
                (run_id, manifest['asset'], manifest['mode'], manifest['as_of'],
                 json.dumps(manifest, default=json_default, allow_nan=False)))
            self.db.executemany('INSERT INTO price_bars VALUES (?, ?, ?, ?, ?, ?, ?)', [
                (run_id, row.timestamp.isoformat(), row.open_timestamp.isoformat(),
                 row.open, row.high, row.low, row.close) for row in prices.itertuples()])
            self.db.executemany('INSERT INTO x_posts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [
                (run_id, row.post_id, row.created_at.isoformat(), row.text, row.likes,
                 row.reposts, row.views, row.followers, row.sentiment, row.bar_end)
                for row in posts.itertuples()])
            self.db.executemany('INSERT INTO asset_news VALUES (?, ?, ?, ?, ?)', [
                (run_id, row['url'], row['title'], row['content'], row['published_at']) for row in news])

    def joined_bars(self, run_id):
        """Aggregate and join X sentiment to prices inside the database."""
        frame = pd.read_sql_query('''
            SELECT p.timestamp, p.open_timestamp, p.open, p.high, p.low, p.close,
                   COUNT(x.post_id) AS post_count,
                   COALESCE(AVG(x.sentiment), 0.0) AS sentiment_mean,
                   COALESCE(AVG(CASE WHEN x.post_id IS NOT NULL THEN
                     CASE WHEN x.sentiment >= 0.1 THEN 1.0 ELSE 0.0 END END), 0.0) AS positive_share,
                   COALESCE(AVG(CASE WHEN x.post_id IS NOT NULL THEN
                     CASE WHEN x.sentiment <= -0.1 THEN 1.0 ELSE 0.0 END END), 0.0) AS negative_share
            FROM price_bars p LEFT JOIN x_posts x
              ON p.run_id=x.run_id AND p.timestamp=x.bar_end
            WHERE p.run_id=? GROUP BY p.timestamp ORDER BY p.timestamp
        ''', self.db, params=(run_id,))
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True)
        frame['open_timestamp'] = pd.to_datetime(frame['open_timestamp'], utc=True)
        return frame.set_index('timestamp')

    def save_outputs(self, run_id, features, predictions):
        with self.db:
            self.db.executemany('INSERT INTO forecast_features VALUES (?, ?, ?)', [
                (run_id, ts.isoformat(), json.dumps(row.to_dict(), default=json_default, allow_nan=False))
                for ts, row in features.iterrows()])
            self.db.executemany('INSERT INTO forecast_outputs VALUES (?, ?, ?, ?, ?, ?)', [
                (run_id, row['horizon'], row['target_timestamp'], row['price'],
                 row['lower_price'], row['upper_price']) for row in predictions])
