"""Regression coverage for lifecycle, inference, and persistent API results."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.api.routes import create_app
from src.core.sentiment.analyzer import SentimentAnalyzer
from src.core.sentiment.ml_model import MLModelAnalyzer
from src.core.sentiment.rule_based import RuleBasedAnalyzer
from src.core.utils.config import Config
from src.storage.repository import Repository


def test_api_lifecycle_reuses_components_and_persists_results():
    app = create_app({'storage': {'path': ':memory:'}})
    with TestClient(app) as client:
        analyzer = app.state.analyzer
        repository = app.state.repository
        assert client.get('/api/v1/health').status_code == 200
        for _ in range(2):
            response = client.post('/api/v1/analyze', json={
                'text': 'I love this excellent product', 'include_details': True,
            })
            assert response.status_code == 200
            body = response.json()
            assert body['correlation_id'] == response.headers['x-correlation-id']
            assert body['details']['rule_based']['positive_words_found'] >= 2
            assert body['details']['ml_model']['probabilities']
            assert app.state.analyzer is analyzer
            assert app.state.repository is repository
        history = client.get('/api/v1/analyze/history').json()['results']
        assert len(history) == 2
        assert all(row['metadata']['processing_time_ms'] > 0 for row in history)
        assert body['correlation_id'] in {row['metadata']['correlation_id'] for row in history}
    assert repository.connection is None
    assert not analyzer._initialized


def test_batch_timing_errors_and_persistence():
    app = create_app({'storage': {'path': ':memory:'}})
    with TestClient(app) as client:
        response = client.post('/api/v1/analyze/batch', json={'texts': ['great', 'bad', ' ']})
        assert response.status_code == 200
        body = response.json()
        assert body['correlation_id'] == response.headers['x-correlation-id']
        assert all(item['processing_time_ms'] > 0 for item in body['results'][:2])
        assert body['results'][2]['sentiment'] == 'error'
        assert client.get('/api/v1/analyze/history').json()['count'] == 2
        assert client.get('/api/v1/analyze/history?limit=-1').status_code == 422
        assert client.get('/api/v1/analyze/history?offset=-1').status_code == 422
        for payload in [{'text': ' '}, {'text': 'good', 'language': 'es'}]:
            assert client.post('/api/v1/analyze', json=payload).status_code == 422


def test_config_sections_reach_components_and_weights_are_normalized(tmp_path):
    lexicons = tmp_path / 'lexicons' / 'en'
    lexicons.mkdir(parents=True)
    (lexicons / 'positive_words.txt').write_text('custompositive\n')
    app = create_app({
        'storage': {'path': ':memory:'},
        'sentiment': {'rule_weight': 4, 'ml_weight': 6, 'max_text_length': 20},
        'rule_based': {'lexicon_path': str(lexicons.parent), 'intensifier_weight': 0.8},
        'ml_model': {'max_features': 5, 'cache_max_size': 0},
        'api': {'cors_origins': ['https://example.test']},
    })
    with TestClient(app) as client:
        analyzer = app.state.analyzer
        assert analyzer.rule_based_analyzer.positive_lexicon == {'custompositive'}
        assert analyzer.rule_based_analyzer.intensifier_weight == 0.8
        assert analyzer.ml_analyzer.max_features == 5
        assert (analyzer.rule_weight, analyzer.ml_weight) == (0.4, 0.6)
        assert client.post('/api/v1/analyze', json={'text': 'x' * 21}).status_code == 422
        assert client.post('/api/v1/analyze', json={'text': 'great'}).status_code == 200
        assert not analyzer.ml_analyzer._prediction_cache
        response = client.get('/api/v1/health', headers={'Origin': 'https://example.test'})
        assert response.headers['access-control-allow-origin'] == 'https://example.test'


def test_empty_repository_uses_lexicon_files_and_handles_contractions():
    async def run():
        repo = Repository({'path': ':memory:'})
        await repo.initialize()
        try:
            rule = RuleBasedAnalyzer({}, repo)
            await rule.initialize()
            assert rule.positive_lexicon and rule.negative_lexicon
            assert (await rule.analyze("I don't love it"))['score'] < 0
            assert (await rule.analyze('I don’t love it'))['score'] < 0
        finally:
            await repo.shutdown()
    asyncio.run(run())


def test_model_save_load_round_trip_and_repository_metadata(tmp_path):
    async def run():
        repo = Repository({'path': ':memory:'})
        await repo.initialize()
        try:
            settings = {'model_path': str(tmp_path), 'model_name': 'custom'}
            original = MLModelAnalyzer(settings, repo)
            await original.initialize()
            expected = await original.analyze('great service')
            assert await original.save_model()
            loaded = MLModelAnalyzer(settings, repo)
            loaded._create_default_model = AsyncMock(side_effect=AssertionError('Unexpected training'))
            await loaded.initialize()
            assert await loaded.analyze('great service') == expected
            assert await repo.store_model_metadata(
                language='en', model_name='custom', model_path=str(tmp_path / 'en/custom.joblib')
            )
            registered = MLModelAnalyzer({**settings, 'model_path': str(tmp_path / 'missing')}, repo)
            registered._create_default_model = AsyncMock(side_effect=AssertionError('Unexpected training'))
            await registered.initialize()
            assert await registered.analyze('great service') == expected
        finally:
            await repo.shutdown()
    asyncio.run(run())


def test_neutral_only_model_has_no_negative_probability():
    async def run():
        model = MLModelAnalyzer({}, Repository({}))
        model.pipeline = model._create_mock_pipeline()
        model._initialized = True
        result = await model.analyze('anything', include_details=True)
        assert result['sentiment'] == 'neutral'
        assert result['score'] == 0
        assert result['details']['probabilities'] == {'negative': 0, 'neutral': 1, 'positive': 0}
    asyncio.run(run())


def test_batch_transaction_rolls_back_and_concurrent_writes_succeed():
    async def run():
        repo = Repository({'path': ':memory:'})
        await repo.initialize()
        row = dict(text='good', sentiment='positive', score=1, confidence=1,
                   processing_time_ms=2, correlation_id='test', language='fr')
        try:
            # The second SQL insert fails after the first insert has executed.
            assert not await repo.store_batch_analyses([row, {**row, 'sentiment': None}])
            assert await repo.get_analysis_history() == []
            assert all(await asyncio.gather(*(repo.store_analysis(**row) for _ in range(10))))
            history = await repo.get_analysis_history()
            assert len(history) == 10
            assert all(r['language'] == 'fr' and r['metadata']['correlation_id'] == 'test' for r in history)
        finally:
            await repo.shutdown()
    asyncio.run(run())


def test_component_failure_falls_back_and_total_failure_is_not_neutral():
    app = create_app({'storage': {'path': ':memory:'}})
    with TestClient(app) as client:
        analyzer = app.state.analyzer
        analyzer.ml_analyzer.analyze = AsyncMock(side_effect=RuntimeError('model failed'))
        response = client.post('/api/v1/analyze', json={'text': 'great', 'include_details': True})
        assert response.status_code == 200
        assert response.json()['details']['source'] == 'rule_only'
        analyzer.rule_based_analyzer.analyze = AsyncMock(side_effect=RuntimeError('rules failed'))
        assert client.post('/api/v1/analyze', json={'text': 'great'}).status_code == 500
        analyzer._initialized = False
        assert client.get('/api/v1/health').status_code == 503


@pytest.mark.parametrize('weights', [(-1, 2), (0, 0), (float('nan'), 1), (float('inf'), 1)])
def test_invalid_weights_fail_explicitly(weights):
    with pytest.raises(ValueError):
        SentimentAnalyzer({'rule_weight': weights[0], 'ml_weight': weights[1]}, Repository({}))


def test_environment_overrides_file_settings(monkeypatch, tmp_path):
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('api:\n  cors_origins: [https://file.test]\nml_model:\n  ngram_range: [1, 1]\n')
    monkeypatch.setenv('API_CORS_ORIGINS', 'https://env.test')
    monkeypatch.setenv('ML_MODEL_NGRAM_RANGE', '[1, 3]')
    monkeypatch.setattr(Config, '_instance', None)
    config = Config(str(config_file))
    assert config.get('api.cors_origins') == ['https://env.test']
    assert config.get('ml_model.ngram_range') == [1, 3]


def test_ml_only_component_is_used():
    async def run():
        analyzer = SentimentAnalyzer({}, Repository({}))
        analyzer._initialized = True
        analyzer.ml_analyzer = AsyncMock()
        analyzer.ml_analyzer.analyze.return_value = {
            'sentiment': 'positive', 'score': 0.8, 'confidence': 0.9,
        }
        result = await analyzer.analyze('great', include_details=True)
        assert result['score'] == 0.8
        assert result['details']['source'] == 'ml_only'
    asyncio.run(run())


def test_failed_startup_closes_database(monkeypatch):
    closed = []
    original_shutdown = Repository.shutdown

    async def shutdown(repository):
        await original_shutdown(repository)
        closed.append(repository.connection is None)

    monkeypatch.setattr(Repository, 'shutdown', shutdown)
    app = create_app({
        'storage': {'path': ':memory:'},
        'sentiment': {'rule_weight': -1},
    })
    with pytest.raises(ValueError):
        with TestClient(app):
            pass
    assert closed == [True]


def test_agent_owns_the_same_resources_as_its_app(monkeypatch, tmp_path):
    import logging
    import src.agent as agent_module

    path = tmp_path / 'agent.yaml'
    path.write_text("storage:\n  path: ':memory:'\n")
    monkeypatch.setattr(Config, '_instance', None)
    monkeypatch.setattr(agent_module, 'setup_logging', lambda settings: logging.getLogger())

    async def run():
        agent = agent_module.SentimentAgent(str(path))
        await agent.initialize()
        repository = agent.repository
        try:
            await agent.initialize()
            assert agent.repository is repository is agent.app.state.repository
            assert agent.analyzer is agent.app.state.analyzer
        finally:
            await agent.shutdown()
            await agent.shutdown()
        assert repository.connection is None
    asyncio.run(run())
