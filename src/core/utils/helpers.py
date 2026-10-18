"""
Utility helper functions.
Common functions used across the sentiment analysis agent.
"""

import re
import hashlib
import uuid
import logging
import logging.config
from typing import List, Set, Optional, Dict, Any
from datetime import datetime
import os

logger = logging.getLogger(__name__)


def setup_logging(logging_config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration.

    Args:
        logging_config: Logging configuration dictionary

    Returns:
        Configured logger instance
    """
    # Use default config if none provided
    if not logging_config:
        logging_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_path': './logs/app.log',
            'max_size_mb': 10,
            'backup_count': 5,
            'console_output': True
        }

    # Create logs directory if it doesn't exist
    log_file_path = logging_config.get('file_path', './logs/app.log')
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': logging_config.get('level', 'INFO'),
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
        },
        'root': {
            'level': logging_config.get('level', 'INFO'),
            'handlers': ['console'] if logging_config.get('console_output', True) else []
        }
    }

    # File output is independent of console output.
    if logging_config.get('file_path'):
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': logging_config.get('level', 'INFO'),
            'formatter': 'standard',
            'filename': log_file_path,
            'maxBytes': logging_config.get('max_size_mb', 10) * 1024 * 1024,  # Convert MB to bytes
            'backupCount': logging_config.get('backup_count', 5),
            'encoding': 'utf8'
        }
        config['root']['handlers'].append('file')

    # Apply configuration
    logging.config.dictConfig(config)

    # Return the root logger
    return logging.getLogger()


def normalize_text(text: str) -> str:
    """Normalize text for processing.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def load_lexicon(file_path: str) -> Set[str]:
    """Load a lexicon from a text file.

    Args:
        file_path: Path to lexicon file (one word per line)

    Returns:
        Set of words from the lexicon
    """
    lexicon = set()

    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):  # Skip comments and empty lines
                        lexicon.add(word.lower())
        else:
            logger.warning(f"Lexicon file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error loading lexicon from {file_path}: {e}")

    return lexicon


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for request tracing.

    Returns:
        Unique correlation ID string
    """
    return str(uuid.uuid4())


def get_correlation_id() -> str:
    """Get or generate a correlation ID for the current context.

    In a more sophisticated implementation, this might retrieve
    an ID from thread-local storage or request context.

    Returns:
        Correlation ID string
    """
    # For simplicity, we'll generate a new one each time
    # In a web framework context, this would be stored in request state
    return generate_correlation_id()


def hash_text(text: str) -> str:
    """Generate a SHA256 hash of text.

    Args:
        text: Input text

    Returns:
        SHA256 hex digest
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length.

    Args:
        text: Input text
        max_length: Maximum length before truncation
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if not isinstance(text, str):
        return ""

    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def parse_language_code(language_str: str) -> str:
    """Parse and normalize a language code.

    Args:
        language_str: Language code string (e.g., 'en', 'en-US', 'english')

    Returns:
        Normalized language code (ISO 639-1 format)
    """
    if not isinstance(language_str, str):
        return 'en'

    # Convert to lowercase and extract first part
    lang = language_str.lower().split('-')[0].split('_')[0]

    # Handle common language names
    language_map = {
        'english': 'en',
        'spanish': 'es',
        'french': 'fr',
        'german': 'de',
        'italian': 'it',
        'portuguese': 'pt',
        'russian': 'ru',
        'chinese': 'zh',
        'japanese': 'ja',
        'korean': 'ko'
    }

    return language_map.get(lang, lang if len(lang) == 2 else 'en')


def is_valid_language_code(language_code: str) -> bool:
    """Check if a language code is valid.

    Args:
        language_code: Language code to check

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(language_code, str):
        return False

    # Basic validation: 2 lowercase letters
    return bool(re.match(r'^[a-z]{2}$', language_code))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning a default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return if denominator is zero

    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between a minimum and maximum.

    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def exponential_decay(initial_value: float, decay_rate: float, steps: int) -> float:
    """Calculate exponential decay.

    Args:
        initial_value: Starting value
        decay_rate: Decay rate (0-1)
        steps: Number of steps

    Returns:
        Value after decay
    """
    return initial_value * ((1 - decay_rate) ** steps)


def calculate_confidence_boost(
    base_confidence: float,
    agreement_factor: float,
    max_boost: float = 0.2
) -> float:
    """Calculate confidence boost based on agreement between models.

    Args:
        base_confidence: Base confidence level
        agreement_factor: Factor representing agreement (0-1)
        max_boost: Maximum boost to apply

    Returns:
        Boosted confidence value
    """
    boost = agreement_factor * max_boost
    return min(1.0, base_confidence + boost)


def calculate_confidence_penalty(
    base_confidence: float,
    disagreement_factor: float,
    min_confidence: float = 0.1
) -> float:
    """Calculate confidence penalty based on disagreement between models.

    Args:
        base_confidence: Base confidence level
        disagreement_factor: Factor representing disagreement (0-1)
        min_confidence: Minimum confidence to allow

    Returns:
        Penalized confidence value
    """
    penalty = disagreement_factor * (base_confidence - min_confidence)
    return max(min_confidence, base_confidence - penalty)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    if chunk_size <= 0:
        return [lst]

    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists into a single list.

    Args:
        nested_list: List of lists to flatten

    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def deduplicate_list(lst: List[Any]) -> List[Any]:
    """Remove duplicates from a list while preserving order.

    Args:
        lst: List to deduplicate

    Returns:
        Deduplicated list
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def retry_async(
    func,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying async functions.

    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff_factor: Multiplier for delay after each attempt
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorated function
    """
    import asyncio
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        last_exception = None
        current_delay = delay

        for attempt in range(max_attempts):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor

        # If we got here, all attempts failed
        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
        raise last_exception

    return wrapper


def timer_async(func):
    """Decorator to time async function execution.

    Args:
        func: Async function to time

    Returns:
        Decorated function that logs execution time
    """
    import asyncio
    import functools
    import time

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {elapsed_time:.4f} seconds")
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.debug(f"{func.__name__} failed after {elapsed_time:.4f} seconds: {e}")
            raise

    return wrapper


def singleton(cls):
    """Decorator to implement singleton pattern.

    Args:
        cls: Class to make singleton

    Returns:
        Singleton class
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def validate_text_input(text: Any, max_length: int = 10000) -> Optional[str]:
    """Validate and sanitize text input.

    Args:
        text: Input to validate
        max_length: Maximum allowed length

    Returns:
        Validated text string or None if invalid
    """
    if not isinstance(text, str):
        return None

    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '').replace('�', '')

    # Check length
    if len(text) > max_length:
        logger.warning(f"Text input too long ({len(text)} chars), truncating to {max_length}")
        text = text[:max_length]

    # Check if empty after sanitization
    if not text.strip():
        return None

    return text


def get_timestamp() -> str:
    """Get current timestamp in ISO format.

    Returns:
        Current timestamp string
    """
    return datetime.utcnow().isoformat() + 'Z'


class SimpleCache:
    """Simple in-memory cache with LRU-like eviction."""

    def __init__(self, max_size: int = 1000):
        """Initialize the cache.

        Args:
            max_size: Maximum number of items to store
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []

    def get(self, key: str) -> Any:
        """Get an item from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            # Update access order (move to end)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set an item in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing item
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new item
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]

            self.cache[key] = value
            self.access_order.append(key)

    def delete(self, key: str) -> bool:
        """Delete an item from the cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)
            return True
        return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        self.cache.clear()
        self.access_order.clear()

    def size(self) -> int:
        """Get the current number of items in the cache.

        Returns:
            Number of items in cache
        """
        return len(self.cache)