"""
Configuration management component.
Handles loading, parsing, and providing access to application configuration.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the sentiment analysis agent."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str] = None):
        """Ensure singleton pattern for Config."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.

        Args:
            config_path: Path to configuration file
        """
        # Initialization happens in __new__
        pass

    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from file and environment variables.

        Args:
            config_path: Path to configuration file
        """
        # Start with default configuration
        self._config = self._get_default_config()

        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self._config = self._merge_configs(
                self._config,
                self._load_config_file(config_path)
            )
        else:
            # Try to load from default locations
            default_paths = [
                './configs/config.yaml',
                './configs/config.yml',
                './config.yaml',
                './config.yml',
                './config.json',
                '/etc/cwt-sentiment-agent/config.yaml'
            ]

            for path in default_paths:
                if os.path.exists(path):
                    self._config = self._merge_configs(
                        self._config,
                        self._load_config_file(path)
                    )
                    break

        # Override with environment variables
        self._config = self._apply_env_overrides(self._config)

        logger.debug(f"Loaded configuration: {json.dumps(self._config, indent=2)}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values.

        Returns:
            Default configuration dictionary
        """
        return {
            'app': {
                'name': 'CWT X Sentiment Agent',
                'version': '1.0.0',
                'description': 'Sentiment analysis agent for CrowdWisdom Trading',
                'environment': os.getenv('ENVIRONMENT', 'development')
            },
            'api': {
                'host': os.getenv('API_HOST', '0.0.0.0'),
                'port': int(os.getenv('API_PORT', '8000')),
                'debug': os.getenv('API_DEBUG', 'false').lower() == 'true',
                'cors_origins': os.getenv('API_CORS_ORIGINS', '*').split(','),
                'workers': int(os.getenv('API_WORKERS', '1'))
            },
            'sentiment': {
                'language': os.getenv('SENTIMENT_LANGUAGE', 'en'),
                'confidence_threshold': float(os.getenv('SENTIMENT_CONFIDENCE_THRESHOLD', '0.6')),
                'use_ensemble': os.getenv('SENTIMENT_USE_ENSEMBLE', 'true').lower() == 'true',
                'rule_weight': float(os.getenv('SENTIMENT_RULE_WEIGHT', '0.4')),
                'ml_weight': float(os.getenv('SENTIMENT_ML_WEIGHT', '0.6')),
                'max_text_length': int(os.getenv('SENTIMENT_MAX_TEXT_LENGTH', '10000')),
                'cache_predictions': os.getenv('SENTIMENT_CACHE_PREDICTIONS', 'true').lower() == 'true'
            },
            'rule_based': {
                'case_sensitive': os.getenv('RULE_BASED_CASE_SENSITIVE', 'false').lower() == 'true',
                'intensifier_weight': float(os.getenv('RULE_BASED_INTENSIFIER_WEIGHT', '0.25')),
                'diminisher_weight': float(os.getenv('RULE_BASED_DIMINISHER_WEIGHT', '0.25')),
                'negation_weight': float(os.getenv('RULE_BASED_NEGATION_WEIGHT', '0.5')),
                'lexicon_path': os.getenv('RULE_BASED_LEXICON_PATH', './data/lexicons')
            },
            'ml_model': {
                'model_path': os.getenv('ML_MODEL_PATH', './data/models'),
                'model_name': os.getenv('ML_MODEL_NAME', 'sentiment_model'),
                'vectorizer_name': os.getenv('ML_VECTORIZER_NAME', 'tfidf_vectorizer'),
                'max_features': int(os.getenv('ML_MODEL_MAX_FEATURES', '10000')),
                'ngram_range': json.loads(os.getenv('ML_MODEL_NGRAM_RANGE', '[1, 2]')),
                'cache_predictions': os.getenv('ML_MODEL_CACHE_PREDICTIONS', 'true').lower() == 'true',
                'cache_max_size': int(os.getenv('ML_MODEL_CACHE_MAX_SIZE', '1000'))
            },
            'storage': {
                'type': os.getenv('STORAGE_TYPE', 'sqlite'),
                'path': os.getenv('STORAGE_PATH', './data/sentiment_agent.db'),
                'echo': os.getenv('STORAGE_ECHO', 'false').lower() == 'true'
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'format': os.getenv(
                    'LOG_FORMAT',
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ),
                'file_path': os.getenv('LOG_FILE_PATH', './logs/app.log'),
                'max_size_mb': int(os.getenv('LOG_MAX_SIZE_MB', '10')),
                'backup_count': int(os.getenv('LOG_BACKUP_COUNT', '5')),
                'console_output': os.getenv('LOG_CONSOLE_OUTPUT', 'true').lower() == 'true'
            }
        }

    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from a file.

        Args:
            file_path: Path to configuration file

        Returns:
            Configuration dictionary from file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif file_path.endswith('.json'):
                    config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path}")

            return config if config is not None else {}
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            return {}

    def _merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration.

        Args:
            config: Base configuration dictionary

        Returns:
            Configuration with environment overrides applied
        """
        # Create a copy to avoid modifying the original
        result = json.loads(json.dumps(config))

        # Define environment variable mappings
        env_mappings = {
            # App settings
            'APP_NAME': ['app', 'name'],
            'APP_VERSION': ['app', 'version'],
            'APP_ENV': ['app', 'environment'],

            # API settings
            'API_HOST': ['api', 'host'],
            'API_PORT': ['api', 'port'],
            'API_DEBUG': ['api', 'debug'],
            'API_WORKERS': ['api', 'workers'],
            'API_CORS_ORIGINS': ['api', 'cors_origins'],

            # Sentiment analysis settings
            'SENTIMENT_LANGUAGE': ['sentiment', 'language'],
            'SENTIMENT_CONFIDENCE_THRESHOLD': ['sentiment', 'confidence_threshold'],
            'SENTIMENT_USE_ENSEMBLE': ['sentiment', 'use_ensemble'],
            'SENTIMENT_RULE_WEIGHT': ['sentiment', 'rule_weight'],
            'SENTIMENT_ML_WEIGHT': ['sentiment', 'ml_weight'],
            'SENTIMENT_MAX_TEXT_LENGTH': ['sentiment', 'max_text_length'],

            # Rule-based settings
            'RULE_BASED_CASE_SENSITIVE': ['rule_based', 'case_sensitive'],
            'RULE_BASED_INTENSIFIER_WEIGHT': ['rule_based', 'intensifier_weight'],
            'RULE_BASED_DIMINISHER_WEIGHT': ['rule_based', 'diminisher_weight'],
            'RULE_BASED_NEGATION_WEIGHT': ['rule_based', 'negation_weight'],
            'RULE_BASED_LEXICON_PATH': ['rule_based', 'lexicon_path'],

            # ML model settings
            'ML_MODEL_PATH': ['ml_model', 'model_path'],
            'ML_MODEL_NAME': ['ml_model', 'model_name'],
            'ML_VECTORIZER_NAME': ['ml_model', 'vectorizer_name'],
            'ML_MODEL_MAX_FEATURES': ['ml_model', 'max_features'],
            'ML_MODEL_NGRAM_RANGE': ['ml_model', 'ngram_range'],
            'ML_MODEL_CACHE_PREDICTIONS': ['ml_model', 'cache_predictions'],
            'ML_MODEL_CACHE_MAX_SIZE': ['ml_model', 'cache_max_size'],

            # Storage settings
            'STORAGE_TYPE': ['storage', 'type'],
            'STORAGE_PATH': ['storage', 'path'],
            'STORAGE_ECHO': ['storage', 'echo'],

            # Logging settings
            'LOG_LEVEL': ['logging', 'level'],
            'LOG_FILE_PATH': ['logging', 'file_path'],
            'LOG_MAX_SIZE_MB': ['logging', 'max_size_mb'],
            'LOG_BACKUP_COUNT': ['logging', 'backup_count'],
            'LOG_CONSOLE_OUTPUT': ['logging', 'console_output']
        }

        # Apply environment variable overrides
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Navigate to the correct nested location
                current = result
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

                # Convert value to appropriate type
                final_key = config_path[-1]
                existing_value = current.get(final_key)

                # Try to preserve the original type
                if isinstance(existing_value, bool):
                    current[final_key] = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(existing_value, int):
                    try:
                        current[final_key] = int(value)
                    except ValueError:
                        logger.warning(f"Could not convert {env_var}={value} to int, keeping default")
                elif isinstance(existing_value, float):
                    try:
                        current[final_key] = float(value)
                    except ValueError:
                        logger.warning(f"Could not convert {env_var}={value} to float, keeping default")
                elif isinstance(existing_value, list):
                    # Try to parse as JSON array
                    try:
                        current[final_key] = json.loads(value)
                    except json.JSONDecodeError:
                        # Fall back to comma-separated string
                        current[final_key] = [item.strip() for item in value.split(',')]
                else:
                    current[final_key] = value

        return result

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'api.host')
            default: Default value to return if key is not found

        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            current = self._config

            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default

            return current
        except Exception:
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section.

        Args:
            section: Section name (e.g., 'api', 'sentiment')

        Returns:
            Configuration section dictionary
        """
        return self.get(section, {})

    def set(self, key_path: str, value: Any) -> None:
        """Set a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        current = self._config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Get the entire configuration as a dictionary.

        Returns:
            Complete configuration dictionary
        """
        return json.loads(json.dumps(self._config))

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration sections."""
        return self.get_section(key)

    def __contains__(self, key: str) -> bool:
        """Check if a configuration section exists."""
        return self.get(key) is not None


# Convenience function for getting config instance
def get_config(config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance.

    Args:
        config_path: Path to configuration file (only used on first call)

    Returns:
        Global Config instance
    """
    return Config(config_path)