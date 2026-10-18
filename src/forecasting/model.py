"""Purged expanding-window validation and six-horizon price forecasting."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, PRICE_FEATURES, HORIZONS, build_targets


def estimator(alpha=10.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def walk_forward(features, bars, min_train=40, test_size=10, alpha=10.0):
    if min_train < 20 or test_size < 1 or alpha <= 0 or not np.isfinite(alpha):
        raise ValueError('Use min_train >= 20, test_size >= 1, and finite alpha > 0')
    targets = build_targets(bars, features.index)
    valid = targets.notna().all(axis=1)
    x, y = features.loc[valid], targets.loc[valid]
    if len(x) < min_train + HORIZONS + 5:
        raise ValueError(f'Only {len(x)} labeled rows; need at least {min_train + HORIZONS + 5}')
    records, folds = [], []
    # Training labels must end strictly before the first test origin. The six
    # purged rows prevent the longest label from crossing the fold boundary.
    for fold, start in enumerate(range(min_train + HORIZONS, len(x), test_size)):
        stop = min(start + test_size, len(x))
        train_end = start - HORIZONS
        train_x, train_y = x.iloc[:train_end], y.iloc[:train_end]
        test_x = x.iloc[start:stop]
        full = estimator(alpha).fit(train_x, train_y)
        price_only = estimator(alpha).fit(train_x[PRICE_FEATURES], train_y)
        predicted = full.predict(test_x)
        price_predicted = price_only.predict(test_x[PRICE_FEATURES])
        train_last_position = bars.index.get_loc(train_x.index[-1])
        label_end = bars.index[train_last_position + HORIZONS]
        assert label_end < test_x.index[0], 'Training labels overlap test origin'
        folds.append({'fold': fold, 'train_rows': len(train_x),
                      'train_start': train_x.index[0].isoformat(),
                      'train_end': train_x.index[-1].isoformat(),
                      'train_label_end': label_end.isoformat(),
                      'test_start': test_x.index[0].isoformat(),
                      'test_end': test_x.index[-1].isoformat()})
        for j, origin in enumerate(test_x.index):
            position = bars.index.get_loc(origin)
            reference = float(bars.loc[origin, 'close'])
            for h in range(HORIZONS):
                target = bars.iloc[position + h + 1]
                records.append({
                    'fold': fold, 'origin': origin, 'horizon': h,
                    'target_timestamp': bars.index[position + h + 1],
                    'reference_price': reference, 'actual_price': float(target['close']),
                    'predicted_price': float(reference * np.exp(predicted[j, h])),
                    'price_only_prediction': float(reference * np.exp(price_predicted[j, h])),
                    'actual_log_return': float(y.loc[origin].iloc[h]),
                    'predicted_log_return': float(predicted[j, h]),
                    'next_open': float(bars.iloc[position + 1]['open']),
                })
    final_model = estimator(alpha).fit(x, y)
    latest = features.iloc[[-1]]
    latest_returns = final_model.predict(latest)[0]
    return final_model, pd.DataFrame(records), folds, latest_returns


def risk_metrics(returns, periods_per_year=252):
    values = np.asarray(returns, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError('Returns must be finite, nonempty, and greater than -100%')
    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    downside = np.sqrt(np.mean(np.minimum(values, 0.0) ** 2))
    equity = np.r_[1.0, np.cumprod(1 + values)]
    drawdown = equity / np.maximum.accumulate(equity) - 1
    return {
        'sharpe_ratio': float(np.sqrt(periods_per_year) * mean / std) if std > 1e-12 else None,
        'sortino_ratio': float(np.sqrt(periods_per_year) * mean / downside) if downside > 1e-12 else None,
        'max_drawdown': float(drawdown.min()),
        'total_return': float(equity[-1] - 1), 'observations': len(values),
    }


def evaluate(oos, transaction_cost_bps=5.0, periods_per_year=252):
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError('Transaction cost must be finite and nonnegative')
    metrics = []
    for h, rows in oos.groupby('horizon'):
        actual = rows['actual_price'].to_numpy()
        for name, column in [('sentiment_price', 'predicted_price'),
                             ('price_only', 'price_only_prediction'),
                             ('persistence', 'reference_price')]:
            predicted = rows[column].to_numpy()
            metrics.append({'horizon': int(h), 'model': name, 'samples': len(rows),
                'mae': float(np.mean(np.abs(actual - predicted))),
                'rmse': float(np.sqrt(np.mean((actual - predicted) ** 2))),
                'mape_pct': float(np.mean(np.abs((actual - predicted) / actual)) * 100),
                'direction_accuracy': float(np.mean(np.sign(predicted - rows['reference_price']) ==
                                                   np.sign(actual - rows['reference_price'])))})
    daily = oos[oos['horizon'] == 0].sort_values('origin').copy()
    # Decide after previous close; enter at NEXT open, exit at that session's
    # close. No impossible fills at the already observed reference close.
    daily['position'] = np.sign(daily['predicted_log_return'])
    daily['asset_return'] = daily['actual_price'] / daily['next_open'] - 1
    daily['strategy_return'] = daily['position'] * daily['asset_return'] - (
        daily['position'].abs() * 2 * transaction_cost_bps / 10000)
    daily['equity'] = (1 + daily['strategy_return']).cumprod()
    risk = risk_metrics(daily['strategy_return'], periods_per_year)
    risk['transaction_cost_bps_per_side'] = transaction_cost_bps
    risk['periods_per_year'] = periods_per_year
    return metrics, risk, daily
