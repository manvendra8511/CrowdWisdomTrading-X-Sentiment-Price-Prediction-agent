"""
Storage repository component.
Handles data persistence and retrieval for the sentiment analysis agent.
"""

import asyncio
import logging
import json
import sqlite3
import aiosqlite
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class Repository:
    """Data storage repository for sentiment analysis agent."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the repository.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Database configuration
        self.db_type = config.get('type', 'sqlite')
        self.db_path = config.get('path', './data/sentiment_agent.db')
        self.echo = config.get('echo', False)

        # Connection
        self.connection: Optional[aiosqlite.Connection] = None
        self._initialized = False
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the database connection and create tables."""
        try:
            self.logger.info("Initializing Storage Repository...")

            if self.db_type != 'sqlite':
                raise ValueError(f'Unsupported storage type: {self.db_type}')

            # Ensure data directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)

            # Create connection
            self.connection = await aiosqlite.connect(self.db_path)

            # Enable foreign key constraints
            await self.connection.execute("PRAGMA foreign_keys = ON")

            # Create tables
            await self._create_tables()

            self._initialized = True
            self.logger.info("Storage Repository initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Storage Repository: {e}")
            await self.shutdown()
            raise

    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            # Analysis results table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_hash TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    processed_text TEXT,
                    sentiment TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    language TEXT DEFAULT 'en',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT  -- JSON stored as text
                )
            """)

            # Create indexes for better query performance
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_results_sentiment
                ON analysis_results(sentiment)
            """)

            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at
                ON analysis_results(created_at)
            """)

            # Lexicons table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS lexicons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    language TEXT NOT NULL,
                    lexicon_type TEXT NOT NULL,  -- 'positive', 'negative', 'intensifier', etc.
                    word TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    UNIQUE(language, lexicon_type, word)
                )
            """)

            # Create index for lexicon lookups
            await self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_lexicons_language_type
                ON lexicons(language, lexicon_type)
            """)

            # Models table (for storing model metadata)
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    language TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    version TEXT DEFAULT '1.0.0',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,  -- JSON stored as text
                    UNIQUE(language, model_name)
                )
            """)

            # Training data table
            await self.connection.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    label INTEGER NOT NULL,  -- 0=negative, 1=neutral, 2=positive
                    language TEXT NOT NULL,
                    source TEXT,  -- Where the data came from
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(text, language)  -- Avoid duplicates
                )
            """)

            # Commit all table creations
            await self.connection.commit()

            self.logger.info("Database tables created/verified successfully")

        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise

    async def store_analysis(
        self,
        text: str,
        sentiment: str,
        score: float,
        confidence: float,
        processing_time_ms: float,
        correlation_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        language: str = 'en'
    ) -> bool:
        """Store an analysis result in the database.

        Args:
            text: Original input text
            sentiment: Sentiment label ('positive', 'negative', 'neutral')
            score: Sentiment score (-1.0 to 1.0)
            confidence: Confidence level (0.0 to 1.0)
            processing_time_ms: Processing time in milliseconds
            correlation_id: Request correlation ID
            metadata: Additional metadata to store
            language: Language code

        Returns:
            True if stored successfully, False otherwise
        """
        return await self.store_batch_analyses([{
            'text': text, 'sentiment': sentiment, 'score': score,
            'confidence': confidence, 'processing_time_ms': processing_time_ms,
            'correlation_id': correlation_id, 'metadata': metadata,
            'language': language,
        }])

    async def store_batch_analyses(self, analyses: List[Dict[str, Any]]) -> bool:
        """Persist a complete batch atomically, serializing concurrent writers."""
        if not self._initialized:
            self.logger.warning("Repository not initialized")
            return False
        if not analyses:
            return True

        import hashlib
        async with self._write_lock:
            try:
                rows = []
                for analysis in analyses:
                    text = analysis['text']
                    metadata = dict(analysis.get('metadata') or {})
                    metadata.update({
                        'processing_time_ms': analysis.get('processing_time_ms', 0.0),
                        'correlation_id': analysis.get('correlation_id', 'unknown'),
                    })
                    rows.append((
                        hashlib.sha256(text.encode('utf-8')).hexdigest(), text,
                        analysis['sentiment'], analysis['score'], analysis['confidence'],
                        analysis.get('language', 'en'), json.dumps(metadata),
                    ))
                await self.connection.executemany(
                    """INSERT INTO analysis_results
                    (text_hash, original_text, sentiment, score, confidence, language, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""", rows,
                )
                await self.connection.commit()
                return True
            except Exception:
                self.logger.exception("Error storing analysis batch")
                await self.connection.rollback()
                return False

    async def get_analysis_history(
        self,
        limit: int = 100,
        offset: int = 0,
        sentiment: Optional[str] = None,
        language: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve analysis history from the database.

        Args:
            limit: Maximum number of results to return
            offset: Number of results to skip
            sentiment: Filter by sentiment ('positive', 'negative', 'neutral')
            language: Filter by language
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)

        Returns:
            List of analysis result dictionaries
        """
        if not self._initialized:
            self.logger.warning("Repository not initialized")
            return []

        try:
            # Build query
            query = """
                SELECT id, text_hash, original_text, processed_text, sentiment,
                       score, confidence, language, created_at, metadata
                FROM analysis_results
                WHERE 1=1
            """
            params = []

            if sentiment:
                query += " AND sentiment = ?"
                params.append(sentiment)

            if language:
                query += " AND language = ?"
                params.append(language)

            if start_date:
                query += " AND created_at >= ?"
                params.append(start_date)

            if end_date:
                query += " AND created_at <= ?"
                params.append(end_date)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            # Execute query
            cursor = await self.connection.execute(query, params)
            rows = await cursor.fetchall()

            # Convert to list of dictionaries
            results = []
            for row in rows:
                result = {
                    'id': row[0],
                    'text_hash': row[1],
                    'original_text': row[2],
                    'processed_text': row[3],
                    'sentiment': row[4],
                    'score': row[5],
                    'confidence': row[6],
                    'language': row[7],
                    'created_at': row[8],
                    'metadata': json.loads(row[9]) if row[9] else {}
                }
                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Error retrieving analysis history: {e}")
            return []

    async def get_lexicons(
        self,
        language: str = 'en'
    ) -> Dict[str, List[str]]:
        """Get sentiment lexicons for a given language.

        Args:
            language: Language code

        Returns:
            Dictionary with lexicon types as keys and word lists as values
        """
        if not self._initialized:
            self.logger.warning("Repository not initialized")
            return {
                'positive': [],
                'negative': [],
                'intensifiers': [],
                'diminishers': [],
                'negations': []
            }

        try:
            lexicon_types = ['positive', 'negative', 'intensifiers', 'diminishers', 'negations']
            result = {}

            for lexicon_type in lexicon_types:
                cursor = await self.connection.execute(
                    """
                    SELECT word FROM lexicons
                    WHERE language = ? AND lexicon_type = ?
                    ORDER BY word
                    """,
                    (language, lexicon_type)
                )
                rows = await cursor.fetchall()
                result[lexicon_type] = [row[0] for row in rows]

            return result

        except Exception as e:
            self.logger.error(f"Error retrieving lexicons: {e}")
            return {
                'positive': [],
                'negative': [],
                'intensifiers': [],
                'diminishers': [],
                'negations': []
            }

    async def store_lexicons(
        self,
        lexicons: Dict[str, List[str]],
        language: str = 'en'
    ) -> bool:
        """Store sentiment lexicons for a given language.

        Args:
            lexicons: Dictionary with lexicon types as keys and word lists as values
            language: Language code

        Returns:
            True if stored successfully, False otherwise
        """
        if not self._initialized:
            self.logger.warning("Repository not initialized")
            return False

        try:
            await self.connection.execute("BEGIN TRANSACTION")

            # Clear existing lexicons for this language
            await self.connection.execute(
                "DELETE FROM lexicons WHERE language = ?",
                (language,)
            )

            # Insert new lexicons
            total_inserted = 0
            for lexicon_type, words in lexicons.items():
                for word in words:
                    await self.connection.execute(
                        """
                        INSERT OR IGNORE INTO lexicons
                        (language, lexicon_type, word, weight)
                        VALUES (?, ?, ?, 1.0)
                        """,
                        (language, lexicon_type, word.strip())
                    )
                    total_inserted += 1

            await self.connection.commit()

            self.logger.info(f"Stored {total_inserted} lexicon entries for language {language}")
            return True

        except Exception as e:
            self.logger.error(f"Error storing lexicons: {e}")
            await self.connection.rollback()
            return False

    async def get_model(
        self,
        language: str = 'en',
        model_name: str = 'sentiment_model'
    ) -> Optional[Dict[str, Any]]:
        """Get model metadata from the database.

        Args:
            language: Language code
            model_name: Name of the model

        Returns:
            Dictionary containing model metadata or None if not found
        """
        if not self._initialized:
            self.logger.warning("Repository not initialized")
            return None

        try:
            cursor = await self.connection.execute(
                """
                SELECT id, model_path, version, created_at, metadata
                FROM models
                WHERE language = ? AND model_name = ?
                """,
                (language, model_name)
            )
            row = await cursor.fetchone()

            if row:
                return {
                    'id': row[0],
                    'model_path': row[1],
                    'version': row[2],
                    'created_at': row[3],
                    'metadata': json.loads(row[4]) if row[4] else {}
                }
            return None

        except Exception as e:
            self.logger.error(f"Error retrieving model: {e}")
            return None

    async def store_model_metadata(
        self,
        model_path: str,
        language: str = 'en',
        model_name: str = 'sentiment_model',
        version: str = '1.0.0',
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store model metadata in the database.

        Args:
            model_path: Path to the model file
            language: Language code
            model_name: Name of the model
            version: Version string
            metadata: Additional metadata to store

        Returns:
            True if stored successfully, False otherwise
        """
        if not self._initialized:
            self.logger.warning("Repository not initialized")
            return False

        try:
            await self.connection.execute("""
                INSERT OR REPLACE INTO models
                (language, model_name, model_path, version, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                language,
                model_name,
                model_path,
                version,
                json.dumps(metadata or {})
            ))

            await self.connection.commit()

            self.logger.info(f"Stored model metadata for {language}/{model_name} v{version}")
            return True

        except Exception as e:
            self.logger.error(f"Error storing model metadata: {e}")
            await self.connection.rollback()
            return False

    async def store_training_data(
        self,
        texts: List[str],
        labels: List[int],
        language: str = 'en',
        source: Optional[str] = None
    ) -> bool:
        """Store training data for future model training.

        Args:
            texts: List of training texts
            labels: List of labels (0=negative, 1=neutral, 2=positive)
            language: Language code
            source: Source of the training data

        Returns:
            True if stored successfully, False otherwise
        """
        if not self._initialized:
            self.logger.warning("Repository not initialized")
            return False

        if len(texts) != len(labels):
            self.logger.error("Number of texts and labels must match")
            return False

        try:
            await self.connection.execute("BEGIN TRANSACTION")

            inserted_count = 0
            for text, label in zip(texts, labels):
                try:
                    await self.connection.execute(
                        """
                        INSERT OR IGNORE INTO training_data
                        (text, label, language, source)
                        VALUES (?, ?, ?, ?)
                        """,
                        (text, label, language, source)
                    )
                    inserted_count += 1
                except Exception as e:
                    self.logger.warning(f"Error inserting training data item: {e}")

            await self.connection.commit()

            self.logger.info(f"Stored {inserted_count}/{len(texts)} training data items")
            return inserted_count == len(texts)

        except Exception as e:
            self.logger.error(f"Error storing training data: {e}")
            await self.connection.rollback()
            return False

    async def get_training_data(
        self,
        language: str = 'en',
        limit: Optional[int] = None
    ) -> Tuple[List[str], List[int]]:
        """Retrieve training data from the database.

        Args:
            language: Language code
            limit: Maximum number of samples to return

        Returns:
            Tuple of (texts, labels) lists
        """
        if not self._initialized:
            self.logger.warning("Repository not initialized")
            return [], []

        try:
            query = """
                SELECT text, label FROM training_data
                WHERE language = ?
                ORDER BY created_at DESC
            """
            params = [language]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor = await self.connection.execute(query, params)
            rows = await cursor.fetchall()

            texts = [row[0] for row in rows]
            labels = [row[1] for row in rows]

            return texts, labels

        except Exception as e:
            self.logger.error(f"Error retrieving training data: {e}")
            return [], []

    async def get_analysis_statistics(
        self,
        language: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get analysis statistics from the database.

        Args:
            language: Filter by language
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)

        Returns:
            Dictionary containing analysis statistics
        """
        if not self._initialized:
            self.logger.warning("Repository not initialized")
            return {}

        try:
            # Build WHERE clause
            where_conditions = []
            params = []

            if language:
                where_conditions.append("language = ?")
                params.append(language)

            if start_date:
                where_conditions.append("created_at >= ?")
                params.append(start_date)

            if end_date:
                where_conditions.append("created_at <= ?")
                params.append(end_date)

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM analysis_results {where_clause}
            """
            cursor = await self.connection.execute(count_query, params)
            total_count = (await cursor.fetchone())[0]

            # Get sentiment distribution
            sentiment_query = f"""
                SELECT sentiment, COUNT(*) as count
                FROM analysis_results {where_clause}
                GROUP BY sentiment
                ORDER BY count DESC
            """
            cursor = await self.connection.execute(sentiment_query, params)
            sentiment_rows = await cursor.fetchall()
            sentiment_dist = {row[0]: row[1] for row in sentiment_rows}

            # Get average score and confidence
            avg_query = f"""
                SELECT AVG(score) as avg_score, AVG(confidence) as avg_confidence
                FROM analysis_results {where_clause}
            """
            cursor = await self.connection.execute(avg_query, params)
            avg_row = await cursor.fetchone()
            avg_score = avg_row[0] if avg_row[0] is not None else 0.0
            avg_confidence = avg_row[1] if avg_row[1] is not None else 0.0

            # Get date range
            date_query = f"""
                SELECT MIN(created_at) as earliest, MAX(created_at) as latest
                FROM analysis_results {where_clause}
            """
            cursor = await self.connection.execute(date_query, params)
            date_row = await cursor.fetchone()
            earliest = date_row[0] if date_row[0] is not None else None
            latest = date_row[1] if date_row[1] is not None else None

            return {
                'total_analyses': total_count,
                'sentiment_distribution': sentiment_dist,
                'average_score': round(avg_score, 4),
                'average_confidence': round(avg_confidence, 4),
                'date_range': {
                    'earliest': earliest,
                    'latest': latest
                },
                'language': language,
                'query_period': {
                    'start_date': start_date,
                    'end_date': end_date
                }
            }

        except Exception as e:
            self.logger.error(f"Error retrieving analysis statistics: {e}")
            return {}

    async def health_check(self) -> bool:
        """Check if the repository is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        if not self._initialized or self.connection is None:
            return False

        try:
            # Execute a simple query to check connectivity
            await self.connection.execute("SELECT 1")
            return True
        except Exception as e:
            self.logger.error(f"Repository health check failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the repository and close connections."""
        self.logger.info("Shutting down Storage Repository...")

        if self.connection:
            try:
                await self.connection.close()
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}")

        self.connection = None
        self._initialized = False
        self.logger.info("Storage Repository shutdown complete")