"""Normalize external records and enforce daily, point-in-time data contracts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def utc(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f'Timestamp must include a timezone: {value!r}')
    return timestamp.tz_convert('UTC')


def schedule(start, end, calendar='XNYS'):
    """Exchange sessions include holidays, early closes, and DST transitions."""
    import exchange_calendars as calendars
    cal = calendars.get_calendar(calendar, start=str(start)[:10], end=str(end)[:10])
    result = cal.schedule.loc[str(start)[:10]:str(end)[:10], ['open', 'close']].copy()
    result['open'] = pd.to_datetime(result['open'], utc=True)
    result['close'] = pd.to_datetime(result['close'], utc=True)
    return result


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path).replace({np.nan: None}).to_dict('records')
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f'{path}: expected a JSON array of records')
    return value


def normalize_prices(records, asset, start, end, calendar='XNYS'):
    """Use raw close consistently; never mix adjusted close and raw open."""
    rows = []
    sessions = schedule(start.date(), end.date(), calendar)
    for record in records:
        row = {str(k).lower(): v for k, v in record.items()}
        if str(row.get('symbol', asset)).upper() != asset.upper():
            continue
        # Daily provider timestamps generally denote the session date/open,
        # not when the close became available. Resolve the exchange close.
        day = str(row.get('date') or row.get('timestamp') or '')[:10]
        if day and (day < str(start.date()) or day > str(end.date())):
            continue
        if day not in sessions.index.strftime('%Y-%m-%d'):
            raise ValueError(f'Price date {day!r} is not a {calendar} session')
        session = sessions.loc[day]
        if not (start <= session['close'] <= end):
            continue  # Excludes unfinished bars.
        values = {key: float(row[key]) for key in ('open', 'high', 'low', 'close')}
        if not all(np.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError(f'Nonpositive or nonfinite OHLC at {day}')
        if values['low'] > min(values['open'], values['close']) or values['high'] < max(values['open'], values['close']) or values['low'] > values['high']:
            raise ValueError(f'Inconsistent OHLC at {day}')
        rows.append({'asset': asset, 'timestamp': session['close'],
                     'open_timestamp': session['open'], **values})
    if not rows:
        raise ValueError('No completed price bars in the requested window')
    frame = pd.DataFrame(rows).sort_values('timestamp')
    if frame['timestamp'].duplicated().any():
        raise ValueError('Duplicate daily price bars; resolve conflicting data before training')
    expected = sessions.loc[(sessions['close'] >= frame['timestamp'].min()) &
                            (sessions['close'] <= frame['timestamp'].max()), 'close']
    if len(expected) != len(frame):
        raise ValueError('Missing exchange sessions in price data; do not forward-fill prices')
    return frame.reset_index(drop=True)


def normalize_posts(records, asset, start, end):
    rows = []
    for record in records:
        text = record.get('text') or record.get('full_text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('X record has no text (check actor output schema)')
        created = utc(record.get('created_at') or record.get('createdAt') or record.get('timestamp'))
        if not start <= created <= end:
            continue
        author = record.get('author') or {}
        def count(*keys, fallback=0):
            value = next((record[k] for k in keys if record.get(k) is not None), fallback)
            value = int(value or 0)
            if value < 0:
                raise ValueError('Engagement counts cannot be negative')
            return value
        rows.append({
            'asset': asset, 'post_id': str(record.get('id') or record.get('post_id') or
                hashlib.sha256(f'{created}:{text}'.encode()).hexdigest()),
            'created_at': created, 'text': text.strip(),
            'likes': count('likes', 'likeCount'),
            'reposts': count('reposts', 'retweetCount'),
            'views': count('views', 'viewCount'),
            'followers': count('followers', fallback=author.get('followers', author.get('followersCount', 0))),
        })
    if not rows:
        raise ValueError('No X posts in the requested window; refusing a sentiment-free run')
    return pd.DataFrame(rows).drop_duplicates('post_id').sort_values('created_at').reset_index(drop=True)


def normalize_news(records, end):
    rows = []
    for record in records:
        url = record.get('url', '')
        if not url.startswith(('https://', 'http://')):
            continue
        published = record.get('published_date') or record.get('published_at')
        timestamp = utc(published) if published else None
        if timestamp is not None and (timestamp > end or timestamp < end - pd.Timedelta(days=7)):
            continue
        rows.append({'url': url, 'title': str(record.get('title', 'Untitled')),
                     'content': str(record.get('content', '')),
                     'published_at': timestamp.isoformat() if timestamp is not None else None})
    return list({row['url']: row for row in rows}.values())


def json_default(value):
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + '\n')
