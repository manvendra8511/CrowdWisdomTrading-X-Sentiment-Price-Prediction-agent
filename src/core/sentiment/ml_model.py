"""
Machine learning model sentiment analysis component.
Handles loading, preprocessing, and inference with ML models.
"""

import asyncio
import logging
import os
import json
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import hashlib

from src.core.utils.helpers import normalize_text
from src.storage.repository import Repository


logger = logging.getLogger(__name__)


class MLModelAnalyzer:
    """ML-based sentiment analyzer using trained models."""

    def __init__(self, config: Dict[str, Any], repository: Repository):
        """Initialize the ML model analyzer.

        Args:
            config: Configuration dictionary
            repository: Storage repository instance
        """
        self.config = config
        self.repository = repository
        self.logger = logging.getLogger(__name__)

        # Model components
        self.pipeline: Optional[Pipeline] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None

        # Configuration
        self.model_path = config.get('model_path', './data/models')
        self.model_name = config.get('model_name', 'sentiment_model')
        self.vectorizer_name = config.get('vectorizer_name', 'tfidf_vectorizer')
        self.language = config.get('language', 'en')
        self.max_features = config.get('max_features', 10000)
        self.ngram_range = tuple(config.get('ngram_range', [1, 2]))
        self.cache_predictions = config.get('cache_predictions', True)

        # Prediction cache
        self._prediction_cache: Dict[str, Tuple[str, float, float]] = {}
        self._cache_max_size = config.get('cache_max_size', 1000)

        # Initialization flag
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the ML model analyzer by loading or creating models."""
        try:
            self.logger.info("Initializing ML Model Analyzer...")

            # Try to load existing model
            model_loaded = await self._load_model()

            if not model_loaded:
                self.logger.info("No existing model found, creating default model...")
                await self._create_default_model()

            self._initialized = True
            self.logger.info("ML Model Analyzer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize ML Model Analyzer: {e}")
            raise

    async def _load_model(self) -> bool:
        """Load trained model and vectorizer from disk or repository.

        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            # Try to load from repository first
            model_data = await self.repository.get_model(self.language, self.model_name)

            if model_data and 'model_path' in model_data:
                self.pipeline = joblib.load(model_data['model_path'])
                self.logger.info("Loaded model from repository")
                return True

        except Exception as e:
            self.logger.warning(f"Could not load model from repository: {e}")

        # Try to load from filesystem
        try:
            model_file = os.path.join(
                self.model_path,
                self.language,
                f"{self.model_name}.joblib"
            )
            if os.path.exists(model_file):
                self.pipeline = joblib.load(model_file)
                # The saved pipeline contains both vectorizer and classifier.
                self.logger.info(f"Loaded model from {model_file}")
                return True

        except Exception as e:
            self.logger.warning(f"Could not load model from filesystem: {e}")

        return False

    async def _create_default_model(self) -> None:
        """Create a default model as fallback."""
        self.logger.info("Creating default sentiment model...")

        if self.language != 'en':
            raise RuntimeError(f'No trained model available for {self.language}')

        # Create a simple pipeline with TF-IDF and Logistic Regression
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                lowercase=True,
                stop_words='english'
            )),
            ('clf', LogisticRegression(
                random_state=42,
                max_iter=1000
            ))
        ])

        # Train with minimal default data to avoid NotFittedError
        # In practice, this would be replaced with proper training data
        default_texts = [
            "I love this product",  # positive
            "This is terrible",     # negative
            "It is okay",           # neutral
            "Great service",        # positive
            "Poor quality",         # negative
            "Average performance"   # neutral
        ]

        default_labels = [2, 0, 1, 2, 0, 1]  # 0=negative, 1=neutral, 2=positive

        try:
            self.pipeline.fit(default_texts, default_labels)
            self.logger.info("Default model created and fitted with minimal data")
        except Exception as e:
            self.logger.error(f"Failed to create default model: {e}")
            # Create a mock pipeline that returns neutral predictions
            self.pipeline = self._create_mock_pipeline()

    def _create_mock_pipeline(self) -> Pipeline:
        """Create a mock pipeline that returns neutral predictions.

        Returns:
            A pipeline that always predicts neutral sentiment
        """
        from sklearn.dummy import DummyClassifier

        mock_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=100)),
            ('clf', DummyClassifier(strategy='constant', constant=1))  # Always predict neutral (1)
        ])

        # Fit with dummy data
        mock_pipeline.fit(["dummy text"], [1])
        return mock_pipeline

    async def analyze(
        self,
        text: str,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """Analyze sentiment using ML model.

        Args:
            text: Preprocessed input text
            include_details: Whether to include detailed breakdown

        Returns:
            Dictionary containing sentiment analysis results
        """
        if not self._initialized:
            raise RuntimeError("MLModelAnalyzer not initialized")

        if not text or not isinstance(text, str):
            return self._neutral_result(include_details, "Invalid input text")

        try:
            # Check cache first
            if self.cache_predictions:
                text_hash = self._hash_text(text)
                if text_hash in self._prediction_cache and not include_details:
                    sentiment, score, confidence = self._prediction_cache[text_hash]
                    result = {
                        'sentiment': sentiment,
                        'score': round(score, 4),
                        'confidence': round(confidence, 4)
                    }
                    if include_details:
                        result['details'] = {'cached': True}
                    return result

            # Normalize text
            processed_text = normalize_text(text)

            # Make prediction
            # Get prediction probabilities
            probabilities = (await asyncio.to_thread(self.pipeline.predict_proba, [processed_text]))[0]
            prediction = self.pipeline.classes_[int(np.argmax(probabilities))]

            # Map prediction to sentiment and score
            # Assuming classes are [negative, neutral, positive] -> [0, 1, 2]
            classes = self.pipeline.classes_
            by_class = dict(zip(classes, probabilities))
            neg_prob = float(by_class.get(0, 0.0))
            neu_prob = float(by_class.get(1, 0.0))
            pos_prob = float(by_class.get(2, 0.0))

            # Convert to score in [-1, 1] range
            # score = (pos_prob - neg_prob) / (pos_prob + neg_prob + neu_prob) * 2 - 1
            # Simpler approach: score = pos_prob - neg_prob
            score = pos_prob - neg_prob
            score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]

            # Determine sentiment based on score
            if score >= 0.1:
                sentiment = 'positive'
            elif score <= -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            # Confidence is the maximum probability
            confidence = float(max(probabilities))

            result = {
                'sentiment': sentiment,
                'score': round(score, 4),
                'confidence': round(confidence, 4)
            }

            if include_details:
                result['details'] = {
                    'probabilities': {
                        'negative': round(neg_prob, 4),
                        'neutral': round(neu_prob, 4),
                        'positive': round(pos_prob, 4)
                    },
                    'prediction_class': int(prediction),
                    'processed_text': processed_text[:100] + ('...' if len(processed_text) > 100 else ''),
                    'model_type': str(type(self.pipeline.named_steps['clf']).__name__)
                }

            # Cache the result
            if self.cache_predictions:
                self._cache_prediction(text_hash, sentiment, score, confidence)

            return result

        except Exception as e:
            self.logger.error(f"Error in ML model analysis: {e}")
            raise

    def _hash_text(self, text: str) -> str:
        """Create a hash of the text for caching.

        Args:
            text: Input text

        Returns:
            SHA256 hash of the text
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _cache_prediction(
        self,
        text_hash: str,
        sentiment: str,
        score: float,
        confidence: float
    ) -> None:
        """Cache a prediction result.

        Args:
            text_hash: Hash of the input text
            sentiment: Sentiment label
            score: Sentiment score
            confidence: Confidence level
        """
        if self._cache_max_size <= 0:
            return
        # Implement simple LRU-like cache behavior
        if len(self._prediction_cache) >= self._cache_max_size:
            # Remove oldest item (simple FIFO for now)
            oldest_key = next(iter(self._prediction_cache))
            del self._prediction_cache[oldest_key]

        self._prediction_cache[text_hash] = (sentiment, score, confidence)

    def _neutral_result(
        self,
        include_details: bool = False,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return a neutral sentiment result.

        Args:
            include_details: Whether to include details
            error: Optional error message

        Returns:
            Neutral sentiment result
        """
        result = {
            'sentiment': 'neutral',
            'score': 0.0,
            'confidence': 0.0
        }

        if include_details:
            details = {}
            if error:
                details['error'] = error
            result['details'] = details

        return result

    async def train(
        self,
        texts: List[str],
        labels: List[int],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train the ML model on new data.

        Args:
            texts: List of training texts
            labels: List of labels (0=negative, 1=neutral, 2=positive)
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary containing training results
        """
        if not self._initialized:
            raise RuntimeError("MLModelAnalyzer not initialized")

        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report

            self.logger.info(f"Training model on {len(texts)} samples...")

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                texts, labels,
                test_size=validation_split,
                random_state=42,
                stratify=labels
            )

            # Train the pipeline
            self.pipeline.fit(X_train, y_train)

            # Evaluate on validation set
            val_predictions = self.pipeline.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)

            # Get classification report
            report = classification_report(
                y_val, val_predictions,
                target_names=['negative', 'neutral', 'positive'],
                output_dict=True
            )

            # Clear prediction cache after training
            self._prediction_cache.clear()

            training_result = {
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'validation_accuracy': round(val_accuracy, 4),
                'classification_report': report
            }

            self.logger.info(f"Model training completed. Validation accuracy: {val_accuracy:.4f}")
            return training_result

        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise

    async def save_model(self, filepath: Optional[str] = None) -> bool:
        """Save the trained model to disk.

        Args:
            filepath: Optional path to save model to

        Returns:
            True if saved successfully, False otherwise
        """
        if not self._initialized or self.pipeline is None:
            self.logger.warning("No model to save")
            return False

        try:
            if filepath is None:
                # Create directory if it doesn't exist
                model_dir = os.path.join(self.model_path, self.language)
                os.makedirs(model_dir, exist_ok=True)
                filepath = os.path.join(model_dir, f"{self.model_name}.joblib")

            joblib.dump(self.pipeline, filepath)
            self.logger.info(f"Model saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the ML model analyzer."""
        self.logger.info("Shutting down ML Model Analyzer...")

        # Clear cache
        self._prediction_cache.clear()

        # Note: We don't delete the pipeline as it might be used elsewhere
        # In a longer-running service, we might want to clean up more carefully

        self._initialized = False
        self.logger.info("ML Model Analyzer shutdown complete")