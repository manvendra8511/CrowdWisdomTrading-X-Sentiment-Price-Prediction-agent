"""Explicit live integrations. Imports and demo runs never make network calls."""
import time

import httpx
import pandas as pd


class Apify:
    def __init__(self, token, client=None, timeout=900):
        if not token:
            raise ValueError('Set APIFY_TOKEN for live ingestion')
        self.client = client or httpx.Client(timeout=40)
        self.owns_client = client is None
        self.headers = {'Authorization': f'Bearer {token}'}
        self.timeout = timeout
        self.runs = []

    def close(self):
        if self.owns_client:
            self.client.close()

    def request(self, method, path, **kwargs):
        response = self.client.request(method, 'https://api.apify.com/v2/' + path,
                                       headers=self.headers, **kwargs)
        response.raise_for_status()
        return response.json()

    def run(self, actor, payload):
        # Never retry this POST automatically: a retry could start a second paid run.
        run = self.request('POST', f'acts/{actor.replace("/", "~")}/runs', json=payload)['data']
        run_id = run['id']
        deadline = time.monotonic() + self.timeout
        while run['status'] not in {'SUCCEEDED', 'FAILED', 'ABORTED', 'TIMED-OUT'}:
            if time.monotonic() > deadline:
                self.request('POST', f'actor-runs/{run_id}/abort')
                raise TimeoutError(f'Apify run {run_id} exceeded timeout and was aborted')
            run = self.request('GET', f'actor-runs/{run_id}', params={'waitForFinish': 10})['data']
        if run['status'] != 'SUCCEEDED':
            raise RuntimeError(f'Apify run {run_id} ended with {run["status"]}')
        self.runs.append({'actor': actor, 'run_id': run_id,
                          'dataset_id': run['defaultDatasetId'], 'input': payload})
        items, offset = [], 0
        while True:
            page = self.request('GET', f'datasets/{run["defaultDatasetId"]}/items',
                                params={'clean': 'true', 'format': 'json', 'offset': offset, 'limit': 1000})
            if not isinstance(page, list):
                raise ValueError('Apify dataset did not return an array')
            items.extend(page)
            if len(page) < 1000:
                return items
            offset += len(page)

    def posts(self, actor, asset, keywords, start, end, max_items=1000):
        """Search weekly windows so a run-wide cap cannot hide the oldest months."""
        terms = [asset, '$' + asset, *[f'"{word}"' for word in keywords]]
        records, truncated = [], []
        cursor = start.normalize()
        boundary = end.normalize() + pd.Timedelta(days=1)
        while cursor < boundary:
            stop = min(cursor + pd.Timedelta(days=7), boundary)
            query = f'({" OR ".join(terms)}) lang:en since:{cursor.date()} until:{stop.date()}'
            batch = self.run(actor, {'searchTerms': [query], 'sort': 'Latest', 'maxItems': max_items})
            records.extend(batch)
            if len(batch) >= max_items:
                truncated.append(f'{cursor.date()}..{stop.date()}')
            cursor = stop
        return records, truncated


def fetch_news(token, query, client=None):
    if not token:
        raise ValueError('Set TAVILY_API_KEY for live news ingestion')
    owns_client = client is None
    client = client or httpx.Client(timeout=40)
    try:
        response = client.post('https://api.tavily.com/search',
            headers={'Authorization': f'Bearer {token}'},
            json={'query': query, 'topic': 'news', 'time_range': 'week',
                  'max_results': 10, 'include_answer': False, 'include_raw_content': False})
        response.raise_for_status()
        return response.json()['results']
    finally:
        if owns_client:
            client.close()
