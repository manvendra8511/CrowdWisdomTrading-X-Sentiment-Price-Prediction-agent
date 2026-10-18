"""
API routes for the sentiment analysis agent.
Defines RESTful endpoints for sentiment analysis operations.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from src.core.utils.config import Config
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import time
import asyncio

from src.core.sentiment.analyzer import SentimentAnalyzer
from src.storage.repository import Repository
from src.core.utils.helpers import get_correlation_id

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for request/validation
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'es')")
    include_details: bool = Field(False, description="Include detailed analysis breakdown")


class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Sentiment label (positive/negative/neutral)")
    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score (-1 to 1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    correlation_id: str = Field(..., description="Request correlation ID for tracing")
    details: Optional[Dict[str, Any]] = Field(None, description="Detailed analysis breakdown")


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    language: Optional[str] = Field(None, description="Language code for all texts")
    include_details: bool = Field(False, description="Include detailed analysis breakdown")


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse] = Field(..., description="Analysis results for each text")
    batch_size: int = Field(..., description="Number of texts processed")
    total_processing_time_ms: float = Field(..., description="Total processing time for batch")
    correlation_id: str = Field(..., description="Request correlation ID for tracing")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    timestamp: float = Field(..., description="Current timestamp")
    version: str = Field(..., description="Application version")
    components: Dict[str, str] = Field(..., description="Status of individual components")


# Dependencies share the resources owned by the application lifespan.
async def get_analyzer(request: Request) -> SentimentAnalyzer:
    return request.app.state.analyzer


async def get_repository(request: Request) -> Repository:
    return request.app.state.repository


@router.get("/health", response_model=HealthResponse)
async def health_check(
    repository: Repository = Depends(get_repository),
    analyzer: SentimentAnalyzer = Depends(get_analyzer)
):
    db_ok = await repository.health_check()
    rule = analyzer.rule_based_analyzer
    ml = analyzer.ml_analyzer
    analysis_ok = bool(
        analyzer._initialized and rule and rule._initialized
        and (rule.positive_lexicon or rule.negative_lexicon)
        and ml and ml._initialized and ml.pipeline is not None
    )
    healthy = db_ok and analysis_ok
    return JSONResponse(status_code=200 if healthy else 503, content={
        "status": "healthy" if healthy else "degraded",
        "timestamp": time.time(),
        "version": "1.0.0",
        "components": {
            "database": "healthy" if db_ok else "unhealthy",
            "analyzer": "healthy" if analysis_ok else "unhealthy",
            "api": "healthy",
        },
    })


@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    analyzer: SentimentAnalyzer = Depends(get_analyzer)
):
    """Analyze sentiment of a single text."""
    correlation_id = http_request.state.correlation_id
    start_time = time.time()

    try:
        logger.info(f"Processing sentiment analysis request {correlation_id}")

        # Perform sentiment analysis
        result = await analyzer.analyze(
            text=request.text,
            language=request.language,
            include_details=request.include_details
        )

        processing_time = (time.time() - start_time) * 1000

        # Prepare response
        response = SentimentResponse(
            sentiment=result['sentiment'],
            score=result['score'],
            confidence=result['confidence'],
            processing_time_ms=round(processing_time, 2),
            correlation_id=correlation_id,
            details=result.get('details') if request.include_details else None
        )

        # Store result in background (optional)
        background_tasks.add_task(
            store_analysis_result,
            request.text,
            result,
            processing_time,
            correlation_id,
            analyzer.repository
        )

        logger.info(f"Completed sentiment analysis {correlation_id} in {processing_time:.2f}ms")
        return response

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing sentiment analysis {correlation_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_sentiment_batch(
    request: BatchSentimentRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    analyzer: SentimentAnalyzer = Depends(get_analyzer)
):
    """Analyze sentiment of multiple texts."""
    correlation_id = http_request.state.correlation_id
    start_time = time.time()

    try:
        logger.info(f"Processing batch sentiment analysis {correlation_id} with {len(request.texts)} texts")

        # Process each text
        tasks = []
        for text in request.texts:
            task = analyzer.analyze(
                text=text,
                language=request.language,
                include_details=request.include_details
            )
            tasks.append(task)

        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        sentiment_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing text {i} in batch {correlation_id}: {result}")
                # Create error result
                sentiment_results.append(SentimentResponse(
                    sentiment="error",
                    score=0.0,
                    confidence=0.0,
                    processing_time_ms=0.0,
                    correlation_id=f"{correlation_id}-{i}",
                    details={"error": str(result) if isinstance(result, ValueError) else "Analysis failed"}
                ))
            else:
                sentiment_results.append(SentimentResponse(
                    sentiment=result['sentiment'],
                    score=result['score'],
                    confidence=result['confidence'],
                    processing_time_ms=result.get('processing_time_ms', 0.0),
                    correlation_id=f"{correlation_id}-{i}",
                    details=result.get('details') if request.include_details else None
                ))

        total_processing_time = (time.time() - start_time) * 1000

        # Store results in background
        background_tasks.add_task(
            store_batch_analysis_results,
            request.texts,
            results,
            total_processing_time,
            correlation_id,
            analyzer.repository
        )

        response = BatchSentimentResponse(
            results=sentiment_results,
            batch_size=len(request.texts),
            total_processing_time_ms=round(total_processing_time, 2),
            correlation_id=correlation_id
        )

        logger.info(f"Completed batch sentiment analysis {correlation_id} in {total_processing_time:.2f}ms")
        return response

    except Exception as e:
        logger.error(f"Error processing batch sentiment analysis {correlation_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analyze/history")
async def get_analysis_history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    repository: Repository = Depends(get_repository)
):
    """Get historical analysis results."""
    try:
        results = await repository.get_analysis_history(limit=limit, offset=offset)
        return {
            "results": results,
            "limit": limit,
            "offset": offset,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Error retrieving analysis history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Background tasks use the shared repository, which serializes analysis writes.
async def store_analysis_result(text, result, processing_time, correlation_id, repository):
    stored = await repository.store_analysis(
        text=text, sentiment=result['sentiment'], score=result['score'],
        confidence=result['confidence'], processing_time_ms=processing_time,
        correlation_id=correlation_id, metadata=result.get('details', {}),
        language=result['language'],
    )
    if not stored:
        logger.error("Failed to persist analysis %s", correlation_id)


async def store_batch_analysis_results(texts, results, total_processing_time, correlation_id, repository):
    analyses = [
        dict(text=text, sentiment=result['sentiment'], score=result['score'],
             confidence=result['confidence'],
             processing_time_ms=result['processing_time_ms'],
             correlation_id=f"{correlation_id}-{i}",
             metadata=result.get('details', {}), language=result['language'])
        for i, (text, result) in enumerate(zip(texts, results))
        if not isinstance(result, Exception)
    ]
    if not await repository.store_batch_analyses(analyses):
        logger.error("Failed to persist batch %s", correlation_id)


# Create the FastAPI app instance
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add middleware for correlation ID and logging
async def add_correlation_id(request, call_next):
    """Add correlation ID to requests and log request/response."""
    correlation_id = get_correlation_id()
    request.state.correlation_id = correlation_id

    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000

    # Add headers
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Process-Time-MS"] = str(round(process_time, 2))

    # Log request
    logger.info(
        f"{request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Time: {process_time:.2f}ms "
        f"- CID: {correlation_id}"
    )

    return response

def create_app(config=None):
    """Create an isolated app; load resources once and close them on shutdown."""
    @asynccontextmanager
    async def lifespan(application):
        settings = config if config is not None else Config().to_dict()
        repository = Repository(settings.get('storage', {}))
        analyzer = None
        try:
            await repository.initialize()
            analyzer = SentimentAnalyzer(settings, repository)
            await analyzer.initialize()
            application.state.repository = repository
            application.state.analyzer = analyzer
            yield
        finally:
            try:
                if analyzer is not None:
                    await analyzer.shutdown()
            finally:
                await repository.shutdown()

    application = FastAPI(
        title="CWT X Sentiment Agent API",
        description="RESTful API for sentiment analysis",
        version="1.0.0", lifespan=lifespan,
    )
    # Config is resolved lazily for the module-level app, preserving custom CLI config.
    if config is not None:
        origins = config.get('api', {}).get('cors_origins', ['*'])
    else:
        origins = None
    application.add_middleware(ConfiguredCORSMiddleware, origins=origins)
    application.include_router(router, prefix="/api/v1")
    application.middleware("http")(add_correlation_id)
    return application


class ConfiguredCORSMiddleware:
    def __init__(self, app, origins=None):
        origins = origins if origins is not None else Config().get('api.cors_origins', ['*'])
        self.app = CORSMiddleware(
            app, allow_origins=origins, allow_credentials='*' not in origins,
            allow_methods=['*'], allow_headers=['*'],
        )

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)


app = create_app()
