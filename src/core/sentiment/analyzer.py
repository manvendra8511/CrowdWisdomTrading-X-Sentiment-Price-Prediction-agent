"""
Sentiment analysis core component.
Orchestrates the sentiment analysis pipeline including preprocessing,
rule-based analysis, and machine learning model inference.
"""

import asyncio
import logging
import re
import time
import math
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

from src.core.utils.helpers import normalize_text, load_lexicon
from src.core.sentiment.rule_based import RuleBasedAnalyzer
from src.core.sentiment.ml_model import MLModelAnalyzer
from src.storage.repository import Repository


logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Main sentiment analysis orchestrator."""

    def __init__(self, config: Dict[str, Any], repository: Repository):
        """Initialize the sentiment analyzer.

        Args:
            config: Configuration dictionary
            repository: Storage repository instance
        """
        self.config = config
        config = config.get('sentiment', config)
        self.repository = repository
        self.logger = logging.getLogger(__name__)

        # Component analyzers
        self.rule_based_analyzer: Optional[RuleBasedAnalyzer] = None
        self.ml_analyzer: Optional[MLModelAnalyzer] = None

        # Configuration
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.use_ensemble = config.get('use_ensemble', True)
        self.rule_weight = config.get('rule_weight', 0.4)
        self.ml_weight = config.get('ml_weight', 0.6)
        weights = (self.rule_weight, self.ml_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError('Ensemble weights must be finite, nonnegative, and have a positive sum')
        self.rule_weight, self.ml_weight = (w / sum(weights) for w in weights)

        # Text preprocessing config
        self.max_text_length = config.get('max_text_length', 10000)
        self.language = config.get('language', 'en')

        # Initialization flag
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all analyzer components."""
        try:
            self.logger.info("Initializing Sentiment Analyzer...")

            # Initialize rule-based analyzer
            self.logger.info("Initializing rule-based analyzer...")
            self.rule_based_analyzer = RuleBasedAnalyzer(
                config={**self.config.get('rule_based', {}), 'language': self.language},
                repository=self.repository
            )
            await self.rule_based_analyzer.initialize()

            # Initialize ML analyzer
            self.logger.info("Initializing ML analyzer...")
            self.ml_analyzer = MLModelAnalyzer(
                config={**self.config.get('ml_model', {}), 'language': self.language},
                repository=self.repository
            )
            await self.ml_analyzer.initialize()

            self._initialized = True
            self.logger.info("Sentiment Analyzer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Sentiment Analyzer: {e}")
            raise

    async def analyze(
        self,
        text: str,
        language: Optional[str] = None,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """Analyze sentiment of text.

        Args:
            text: Input text to analyze
            language: Language code (optional)
            include_details: Whether to include detailed breakdown

        Returns:
            Dictionary containing sentiment analysis results
        """
        if not self._initialized:
            raise RuntimeError("SentimentAnalyzer not initialized")

        # Validate input
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text input")

        if len(text) > self.max_text_length:
            raise ValueError(f"Text exceeds maximum length of {self.max_text_length}")

        # Use provided language or default
        lang = language or self.language
        if lang != self.language:
            raise ValueError(f'Unsupported language: {lang}; configured language is {self.language}')
        start_time = time.perf_counter()

        try:
            # Preprocess text
            processed_text = await self._preprocess_text(text, lang)

            components = {
                'rule': self.rule_based_analyzer,
                'ml': self.ml_analyzer,
            }
            active = {name: component for name, component in components.items() if component}
            results = await asyncio.gather(*(
                component.analyze(processed_text, include_details=include_details)
                for component in active.values()
            ), return_exceptions=True)
            successful = {}
            for name, result in zip(active, results):
                if isinstance(result, Exception):
                    self.logger.warning("Analysis component %s failed: %s", name, result)
                else:
                    successful[name] = result
            rule_based_result = successful.get('rule')
            ml_result = successful.get('ml')

            # Fuse results
            final_result = self._fuse_results(
                rule_based_result,
                ml_result,
                include_details=include_details
            )

            # Add metadata
            final_result.update({
                'language': lang,
                'processing_time_ms': (time.perf_counter() - start_time) * 1000,
                'timestamp': datetime.utcnow().isoformat(),
                'original_text_length': len(text),
                'processed_text_length': len(processed_text)
            })

            return final_result

        except Exception as e:
            self.logger.error(f"Error during sentiment analysis: {e}")
            raise

    async def _preprocess_text(self, text: str, language: str) -> str:
        """Preprocess text for analysis.

        Args:
            text: Raw input text
            language: Language code

        Returns:
            Preprocessed text
        """
        # Basic cleaning
        text = text.strip()

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Language-specific preprocessing could go here
        # For now, we'll use a general normalizer

        return text

    def _fuse_results(
        self,
        rule_based_result: Optional[Dict[str, Any]],
        ml_result: Optional[Dict[str, Any]],
        include_details: bool = False
    ) -> Dict[str, Any]:
        """Fuse results from different analysis components.

        Args:
            rule_based_result: Result from rule-based analyzer
            ml_result: Result from ML analyzer
            include_details: Whether to include detailed breakdown

        Returns:
            Fused sentiment analysis result
        """
        details = {} if include_details else None

        # Handle cases where one or both analyzers failed
        if not rule_based_result and not ml_result:
            raise RuntimeError('All sentiment analyzers failed')

        if not rule_based_result:
            return self._format_result(ml_result, 'ml_only', details)

        if not ml_result:
            return self._format_result(rule_based_result, 'rule_only', details)

        # Both results available - perform fusion
        if self.use_ensemble:
            return self._ensemble_fusion(
                rule_based_result,
                ml_result,
                include_details,
                details
            )
        else:
            # Default to ML result if available, otherwise rule-based
            preferred_result = ml_result if ml_result else rule_based_result
            return self._format_result(preferred_result, 'preferred', details)

    def _ensemble_fusion(
        self,
        rule_based_result: Dict[str, Any],
        ml_result: Dict[str, Any],
        include_details: bool,
        details: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform ensemble fusion of rule-based and ML results.

        Args:
            rule_based_result: Result from rule-based analyzer
            ml_result: Result from ML analyzer
            include_details: Whether to include detailed breakdown
            details: Details dictionary to populate

        Returns:
            Fused result
        """
        # Extract scores and sentiments
        rule_score = rule_based_result.get('score', 0.0)
        rule_sentiment = rule_based_result.get('sentiment', 'neutral')
        rule_confidence = rule_based_result.get('confidence', 0.0)

        ml_score = ml_result.get('score', 0.0)
        ml_sentiment = ml_result.get('sentiment', 'neutral')
        ml_confidence = ml_result.get('confidence', 0.0)

        # Store individual results in details
        if include_details:
            details['rule_based'] = {
                **rule_based_result.get('details', {}),
                'sentiment': rule_sentiment,
                'score': rule_score,
                'confidence': rule_confidence
            }
            details['ml_model'] = {
                **ml_result.get('details', {}),
                'sentiment': ml_sentiment,
                'score': ml_score,
                'confidence': ml_confidence
            }

        # Weighted average of scores
        fused_score = (
            self.rule_weight * rule_score +
            self.ml_weight * ml_score
        )

        # Determine fused sentiment based on score
        if fused_score >= 0.1:
            fused_sentiment = 'positive'
        elif fused_score <= -0.1:
            fused_sentiment = 'negative'
        else:
            fused_sentiment = 'neutral'

        # Calculate fused confidence (weighted average)
        fused_confidence = (
            self.rule_weight * rule_confidence +
            self.ml_weight * ml_confidence
        )

        # Boost confidence if both models agree
        if rule_sentiment == ml_sentiment and rule_sentiment != 'neutral':
            agreement_boost = 0.1 * min(rule_confidence, ml_confidence)
            fused_confidence = min(1.0, fused_confidence + agreement_boost)

            if include_details:
                details['agreement_boost'] = agreement_boost

        # Reduce confidence if models disagree significantly
        elif rule_sentiment != ml_sentiment:
            disagreement_penalty = 0.2 * abs(rule_score - ml_score)
            fused_confidence = max(0.1, fused_confidence - disagreement_penalty)

            if include_details:
                details['disagreement_penalty'] = disagreement_penalty

        result = {
            'sentiment': fused_sentiment,
            'score': round(fused_score, 4),
            'confidence': round(fused_confidence, 4)
        }

        if include_details:
            result['details'] = details

        return result

    def _format_result(
        self,
        result: Dict[str, Any],
        source: str,
        details: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format a single analyzer result for output.

        Args:
            result: Result from a single analyzer
            source: Source of the result ('rule_only', 'ml_only', 'preferred')
            details: Details dictionary to populate

        Returns:
            Formatted result
        """
        formatted = {
            'sentiment': result.get('sentiment', 'neutral'),
            'score': round(result.get('score', 0.0), 4),
            'confidence': round(result.get('confidence', 0.0), 4)
        }

        if details is not None:
            details.update(result.get('details', {}))
            details['source'] = source
            formatted['details'] = details

        return formatted

    def _neutral_result(
        self,
        details: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return a neutral sentiment result.

        Args:
            details: Details dictionary to populate

        Returns:
            Neutral sentiment result
        """
        result = {
            'sentiment': 'neutral',
            'score': 0.0,
            'confidence': 0.0
        }

        if details is not None:
            details['reason'] = 'all_analyzers_failed'
            result['details'] = details

        return result

    async def shutdown(self) -> None:
        """Shutdown all analyzer components."""
        self.logger.info("Shutting down Sentiment Analyzer...")

        if self.rule_based_analyzer:
            await self.rule_based_analyzer.shutdown()

        if self.ml_analyzer:
            await self.ml_analyzer.shutdown()

        self._initialized = False
        self.logger.info("Sentiment Analyzer shutdown complete")