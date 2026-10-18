"""
Rule-based sentiment analysis component.
Uses lexicons and linguistic rules to determine sentiment.
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
import json
import os

from src.core.utils.helpers import load_lexicon, normalize_text
from src.storage.repository import Repository


logger = logging.getLogger(__name__)


class RuleBasedAnalyzer:
    """Rule-based sentiment analyzer using lexicons and linguistic rules."""

    def __init__(self, config: Dict[str, Any], repository: Repository):
        """Initialize the rule-based analyzer.

        Args:
            config: Configuration dictionary
            repository: Storage repository instance
        """
        self.config = config
        self.repository = repository
        self.logger = logging.getLogger(__name__)

        # Lexicons
        self.positive_lexicon: set = set()
        self.negative_lexicon: set = set()
        self.intensifier_lexicon: set = set()
        self.diminisher_lexicon: set = set()
        self.negation_lexicon: set = set()

        # Configuration
        self.language = config.get('language', 'en')
        self.case_sensitive = config.get('case_sensitive', False)
        self.intensifier_weight = config.get('intensifier_weight', 0.25)
        self.diminisher_weight = config.get('diminisher_weight', 0.25)
        self.negation_weight = config.get('negation_weight', 0.5)

        # Patterns
        self.sentence_pattern = re.compile(r'[.!?]+')
        self.word_pattern = re.compile(r"\b\w+(?:['’]\w+)*\b")

        # Initialization flag
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the rule-based analyzer by loading lexicons."""
        try:
            self.logger.info("Initializing Rule-Based Analyzer...")

            # Load lexicons from files or repository
            await self._load_lexicons()

            self._initialized = True
            self.logger.info("Rule-Based Analyzer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Rule-Based Analyzer: {e}")
            raise

    async def _load_lexicons(self) -> None:
        """Load sentiment lexicons from various sources."""
        try:
            # Try to load from repository first
            lexicon_data = await self.repository.get_lexicons(self.language)

            if lexicon_data and any(lexicon_data.values()):
                self.positive_lexicon = set(lexicon_data.get('positive', []))
                self.negative_lexicon = set(lexicon_data.get('negative', []))
                self.intensifier_lexicon = set(lexicon_data.get('intensifiers', []))
                self.diminisher_lexicon = set(lexicon_data.get('diminishers', []))
                self.negation_lexicon = set(lexicon_data.get('negations', []))
                self.logger.info(
                    f"Loaded lexicons from repository: "
                    f"{len(self.positive_lexicon)} positive, "
                    f"{len(self.negative_lexicon)} negative, "
                    f"{len(self.intensifier_lexicon)} intensifiers, "
                    f"{len(self.diminisher_lexicon)} diminishers, "
                    f"{len(self.negation_lexicon)} negations"
                )
                return

        except Exception as e:
            self.logger.warning(f"Could not load lexicons from repository: {e}")

        # Fallback to file-based loading
        try:
            base_path = self.config.get('lexicon_path', './data/lexicons')
            lang_path = os.path.join(base_path, self.language)

            # Load positive words
            pos_file = os.path.join(lang_path, 'positive_words.txt')
            if os.path.exists(pos_file):
                with open(pos_file, 'r', encoding='utf-8') as f:
                    self.positive_lexicon = set(
                        line.strip().lower()
                        for line in f
                        if line.strip() and not line.startswith('#')
                    )

            # Load negative words
            neg_file = os.path.join(lang_path, 'negative_words.txt')
            if os.path.exists(neg_file):
                with open(neg_file, 'r', encoding='utf-8') as f:
                    self.negative_lexicon = set(
                        line.strip().lower()
                        for line in f
                        if line.strip() and not line.startswith('#')
                    )

            # Load intensifiers
            int_file = os.path.join(lang_path, 'intensifiers.txt')
            if os.path.exists(int_file):
                with open(int_file, 'r', encoding='utf-8') as f:
                    self.intensifier_lexicon = set(
                        line.strip().lower()
                        for line in f
                        if line.strip() and not line.startswith('#')
                    )

            # Load diminishers
            dim_file = os.path.join(lang_path, 'diminishers.txt')
            if os.path.exists(dim_file):
                with open(dim_file, 'r', encoding='utf-8') as f:
                    self.diminisher_lexicon = set(
                        line.strip().lower()
                        for line in f
                        if line.strip() and not line.startswith('#')
                    )

            # Load negations
            neu_file = os.path.join(lang_path, 'negations.txt')
            if os.path.exists(neu_file):
                with open(neu_file, 'r', encoding='utf-8') as f:
                    self.negation_lexicon = set(
                        line.strip().lower()
                        for line in f
                        if line.strip() and not line.startswith('#')
                    )

            if not self.positive_lexicon and not self.negative_lexicon:
                raise RuntimeError(f'No sentiment lexicons found for {self.language}')

            self.logger.info(
                f"Loaded lexicons from files: "
                f"{len(self.positive_lexicon)} positive, "
                f"{len(self.negative_lexicon)} negative, "
                f"{len(self.intensifier_lexicon)} intensifiers, "
                f"{len(self.diminisher_lexicon)} diminishers, "
                f"{len(self.negation_lexicon)} negations"
            )

        except Exception as e:
            self.logger.error(f"Failed to load lexicons from files: {e}")
            # Initialize with minimal default lexicons to avoid complete failure
            if self.language != 'en':
                raise
            await self._load_default_lexicons()

    async def _load_default_lexicons(self) -> None:
        """Load minimal default lexicons as fallback."""
        self.logger.warning("Loading default minimal lexicons")

        # Basic English sentiment words
        self.positive_lexicon = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'best', 'better', 'love', 'liked', 'enjoy', 'enjoyed', 'happy',
            'pleased', 'satisfied', 'awesome', 'brilliant', 'outstanding',
            'perfect', 'nice', 'fine', 'positive', 'favorable', 'superb'
        }

        self.negative_lexicon = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'worse',
            'hate', 'hated', 'dislike', 'disliked', 'angry', 'sad',
            'unhappy', 'disappointed', 'frustrated', 'annoyed', 'poor',
            'negative', 'unfavorable', 'dreadful', 'abysmal'
        }

        self.intensifier_lexicon = {
            'very', 'extremely', 'incredibly', 'absolutely', 'completely',
            'totally', 'utterly', 'highly', 'remarkably', 'exceptionally',
            'exceptionally', 'extraordinarily', 'particularly', 'especially'
        }

        self.diminisher_lexicon = {
        'slightly', 'somewhat', 'rather', 'quite', 'fairly', 'marginally'
        }

        self.negation_lexicon = {
            'not', 'no', 'never', 'none', 'nothing', 'nowhere',
            'neither', 'nor', 'without', 'cannot', "can't", 'won\'t',
            'don\'t', 'doesn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t'
        }

    async def analyze(
        self,
        text: str,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """Analyze sentiment using rule-based approach.

        Args:
            text: Preprocessed input text
            include_details: Whether to include detailed breakdown

        Returns:
            Dictionary containing sentiment analysis results
        """
        if not self._initialized:
            raise RuntimeError("RuleBasedAnalyzer not initialized")

        try:
            # Normalize text for processing
            text = text.replace("’", "'")
            processed_text = text.lower() if not self.case_sensitive else text

            # Extract words and sentences
            words = self.word_pattern.findall(processed_text)
            sentences = self.sentence_pattern.split(processed_text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not words:
                return self._neutral_result(include_details)

            # Analyze each sentence and aggregate
            sentence_results = []
            total_score = 0.0
            total_confidence = 0.0

            for sentence in sentences:
                sentence_result = await self._analyze_sentence(
                    sentence,
                    include_details=include_details
                )
                sentence_results.append(sentence_result)
                total_score += sentence_result['score']
                total_confidence += sentence_result['confidence']

            # Calculate average scores
            avg_score = total_score / len(sentences) if sentences else 0.0
            avg_confidence = total_confidence / len(sentences) if sentences else 0.0

            # Determine overall sentiment
            if avg_score >= 0.1:
                sentiment = 'positive'
            elif avg_score <= -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            result = {
                'sentiment': sentiment,
                'score': round(avg_score, 4),
                'confidence': round(min(avg_confidence, 1.0), 4)  # Cap confidence at 1.0
            }

            if include_details:
                result['details'] = {
                    'word_count': len(words),
                    'sentence_count': len(sentences),
                    'positive_words_found': self._count_matching_words(
                        words, self.positive_lexicon
                    ),
                    'negative_words_found': self._count_matching_words(
                        words, self.negative_lexicon
                    ),
                    'intensifiers_found': self._count_matching_words(
                        words, self.intensifier_lexicon
                    ),
                    'diminishers_found': self._count_matching_words(
                        words, self.diminisher_lexicon
                    ),
                    'negations_found': self._count_matching_words(
                        words, self.negation_lexicon
                    ),
                    'sentence_analysis': sentence_results if include_details else None,
                    'language': self.language
                }

            return result

        except Exception as e:
            self.logger.error(f"Error in rule-based analysis: {e}")
            raise

    async def _analyze_sentence(
        self,
        sentence: str,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """Analyze sentiment of a single sentence.

        Args:
            sentence: Input sentence to analyze
            include_details: Whether to include detailed breakdown

        Returns:
            Dictionary containing sentence-level sentiment analysis
        """
        words = self.word_pattern.findall(sentence)

        if not words:
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'word_count': 0
            }

        # Initialize scoring variables
        base_score = 0.0
        word_scores = [] if include_details else None
        modifiers_applied = [] if include_details else None

        # Process each word with context
        for i, word in enumerate(words):
            word_score = 0.0
            modifiers = []

            # Check if word is in sentiment lexicons
            if word in self.positive_lexicon:
                word_score = 1.0
            elif word in self.negative_lexicon:
                word_score = -1.0

            if word_score != 0.0:
                # Check for intensifiers/diminishers in window
                start_idx = max(0, i - 3)
                end_idx = min(len(words), i + 3)
                context_words = words[start_idx:end_idx]

                # Check for negations (typically before the word)
                negation_found = False
                for j in range(max(0, i - 3), i):
                    if words[j] in self.negation_lexicon:
                        negation_found = True
                        if include_details:
                            modifiers.append(f"negation:{words[j]}")
                        break

                # Apply negation
                if negation_found:
                    word_score = -word_score

                # Check for intensifiers
                intensifier_count = 0
                for j in range(max(0, i - 2), i):  # Look before the word
                    if words[j] in self.intensifier_lexicon:
                        intensifier_count += 1
                        if include_details:
                            modifiers.append(f"intensifier:{words[j]}")

                # Check for diminishers
                diminisher_count = 0
                for j in range(max(0, i - 2), i):  # Look before the word
                    if words[j] in self.diminisher_lexicon:
                        diminisher_count += 1
                        if include_details:
                            modifiers.append(f"diminisher:{words[j]}")

                # Apply modifiers
                if intensifier_count > 0:
                    intensifier_boost = 1 + (self.intensifier_weight * intensifier_count)
                    word_score *= intensifier_boost

                if diminisher_count > 0:
                    diminisher_reduction = 1 - (self.diminisher_weight * diminisher_count)
                    word_score *= max(0.1, diminisher_reduction)  # Don't go below 0.1

                base_score += word_score

                if include_details:
                    word_scores.append({
                        'word': word,
                        'base_score': 1.0 if word in self.positive_lexicon else -1.0 if word in self.negative_lexicon else 0.0,
                        'final_score': round(word_score, 4),
                        'modifiers': modifiers.copy()
                    })
                    modifiers_applied.extend(modifiers)

        # Normalize score to [-1, 1] range based on word count
        max_possible_score = len(words)  # If all words were max sentiment
        normalized_score = base_score / max_possible_score if max_possible_score > 0 else 0.0
        normalized_score = max(-1.0, min(1.0, normalized_score))  # Clamp to [-1, 1]

        # Calculate confidence based on proportion of sentiment words found
        sentiment_words = sum(1 for w in words if w in self.positive_lexicon or w in self.negative_lexicon)
        confidence = min(1.0, sentiment_words / len(words)) if words else 0.0

        # Boost confidence if we found strong sentiment indicators
        if abs(normalized_score) > 0.5:
            confidence = min(1.0, confidence + 0.2)

        result = {
            'sentiment': self._score_to_sentiment(normalized_score),
            'score': round(normalized_score, 4),
            'confidence': round(confidence, 4),
            'word_count': len(words)
        }

        if include_details:
            result['details'] = {
                'word_scores': word_scores,
                'modifiers_applied': list(set(modifiers_applied)) if modifiers_applied else [],
                'sentiment_words_found': sentiment_words,
                'raw_score': round(base_score, 4)
            }

        return result

    def _score_to_sentiment(self, score: float) -> str:
        """Convert numerical score to sentiment label.

        Args:
            score: Numerical sentiment score

        Returns:
            Sentiment label string
        """
        if score >= 0.1:
            return 'positive'
        elif score <= -0.1:
            return 'negative'
        else:
            return 'neutral'

    def _count_matching_words(self, words: List[str], lexicon: set) -> int:
        """Count how many words match a given lexicon.

        Args:
            words: List of words to check
            lexicon: Lexicon set to match against

        Returns:
            Count of matching words
        """
        return sum(1 for word in words if word in lexicon)

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

    async def shutdown(self) -> None:
        """Shutdown the rule-based analyzer."""
        self.logger.info("Shutting down Rule-Based Analyzer...")

        # Clear lexicons to free memory
        self.positive_lexicon.clear()
        self.negative_lexicon.clear()
        self.intensifier_lexicon.clear()
        self.diminisher_lexicon.clear()
        self.negation_lexicon.clear()

        self._initialized = False
        self.logger.info("Rule-Based Analyzer shutdown complete")