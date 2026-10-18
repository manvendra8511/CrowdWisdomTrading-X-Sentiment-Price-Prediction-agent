"""Causal price and X-sentiment features. No news or engagement snapshots."""
import numpy as np
import pandas as pd

PRICE_FEATURES = ['return_1', 'return_2', 'return_5', 'return_10',
                  'volatility_5', 'volatility_20', 'price_to_mean_5', 'price_to_mean_20',
                  'intraday_return', 'range_fraction']
SENTIMENT_FEATURES = ['sentiment_mean', 'positive_share', 'negative_share', 'log_post_count',
                      'sentiment_mean_3', 'sentiment_mean_5', 'sentiment_change',
                      'post_volume_change', 'no_posts']
FEATURES = PRICE_FEATURES + SENTIMENT_FEATURES
HORIZONS = 6


def build_features(bars):
    result = pd.DataFrame(index=bars.index)
    close = bars['close']
    for lag in (1, 2, 5, 10):
        result[f'return_{lag}'] = np.log(close / close.shift(lag))
    for window in (5, 20):
        result[f'volatility_{window}'] = result['return_1'].rolling(window).std()
        result[f'price_to_mean_{window}'] = close / close.rolling(window).mean() - 1
    result['intraday_return'] = np.log(close / bars['open'])
    result['range_fraction'] = (bars['high'] - bars['low']) / close
    for name in ('sentiment_mean', 'positive_share', 'negative_share'):
        result[name] = bars[name]
    result['log_post_count'] = np.log1p(bars['post_count'])
    for window in (3, 5):
        result[f'sentiment_mean_{window}'] = bars['sentiment_mean'].rolling(window).mean()
    result['sentiment_change'] = bars['sentiment_mean'].diff()
    result['post_volume_change'] = result['log_post_count'].diff()
    result['no_posts'] = (bars['post_count'] == 0).astype(float)
    return result[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()


def build_targets(bars, index):
    # n is the next not-yet-completed session. All six outputs are predictions.
    close = bars['close']
    return pd.DataFrame({f'n+{h}': np.log(close.shift(-(h + 1)) / close)
                         for h in range(HORIZONS)}, index=bars.index).reindex(index)
