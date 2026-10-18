"""Synthetic inputs for testing the full pipeline without credentials."""
import numpy as np
import pandas as pd

from .data import schedule


def make_demo(asset, start, end, calendar='XNYS', seed=42):
    rng = np.random.default_rng(seed)
    sessions = schedule(start.date(), end.date(), calendar)
    prices, posts = [], []
    previous, previous_signal = 100.0, 0.0
    for day, session in sessions.iterrows():
        if session['close'] > end or session['close'] < start:
            continue
        opening = previous * np.exp(rng.normal(0, 0.003))
        close = opening * np.exp(0.002 * previous_signal + rng.normal(0, 0.009))
        prices.append({'date': str(day.date()), 'symbol': asset, 'open': opening,
                       'high': max(opening, close) * 1.004, 'low': min(opening, close) * 0.996,
                       'close': close})
        signal = float(rng.normal())
        for number in range(int(rng.integers(3, 12))):
            positive = rng.random() < 1 / (1 + np.exp(-signal))
            text = (f'${asset} bullish outlook, strong growth and excellent gains' if positive else
                    f'${asset} bearish outlook, weak demand and terrible losses')
            posts.append({'id': f'demo-{day.date()}-{number}', 'text': text,
                          'createdAt': (session['close'] - pd.Timedelta(minutes=30 + number * 10)).isoformat(),
                          'likeCount': int(rng.integers(0, 100)), 'retweetCount': int(rng.integers(0, 20)),
                          'viewCount': int(rng.integers(100, 10000)),
                          'author': {'followers': int(rng.integers(100, 10000))}})
        previous, previous_signal = close, signal
    news = [{'title': f'Synthetic {asset} headline for pipeline testing',
             'url': 'https://example.com/synthetic-news',
             'content': 'Generated test content. This is not a real news story.',
             'published_date': (end - pd.Timedelta(hours=1)).isoformat()}]
    for record in [*prices, *posts, *news]:
        record['_synthetic'] = True
    return prices, posts, news
