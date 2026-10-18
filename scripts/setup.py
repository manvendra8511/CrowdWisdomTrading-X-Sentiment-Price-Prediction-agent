#!/usr/bin/env python3
"""
Setup script for the CWT X Sentiment Agent.
Initializes data directories and basic files.
"""

import os
import sys
import logging
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
project_dir = script_dir.parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary data directories."""
    directories = [
        "data/lexicons/en",
        "data/models",
        "data/datasets",
        "logs",
        "configs"
    ]

    for directory in directories:
        dir_path = project_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def setup_default_lexicons():
    """Ensure default lexicons exist."""
    lexicon_dir = project_dir / "data/lexicons/en"

    # Default lexicons (in case files are missing)
    default_lexicons = {
        "positive_words.txt": ["good", "great", "excellent", "happy", "love", "like", "best", "better", "awesome", "fantastic"],
        "negative_words.txt": ["bad", "terrible", "awful", "hate", "dislike", "angry", "sad", "worst", "worse", "horrible"],
        "intensifiers.txt": ["very", "extremely", "incredibly", "absolutely", "completely", "totally", "utterly"],
        "diminishers.txt": ["slightly", "somewhat", "rather", "quite", "fairly", "marginally"],
        "negations.txt": ["not", "no", "never", "none", "nothing", "nowhere", "neither", "nor", "without"]
    }

    for filename, words in default_lexicons.items():
        filepath = lexicon_dir / filename
        if not filepath.exists():
            with open(filepath, 'w', encoding='utf-8') as f:
                for word in words:
                    f.write(word + '\n')
            logger.info(f"Created default lexicon: {filepath}")
        else:
            logger.info(f"Lexicon already exists: {filepath}")

def create_sample_config():
    """Create a sample configuration file if it doesn't exist."""
    config_file = project_dir / "configs" / "config.yaml"
    if not config_file.exists():
        # Create a basic config file
        config_content = """# Application Configuration
app:
  name: "CWT X Sentiment Agent"
  version: "1.0.0"
  description: "Sentiment analysis agent for CrowdWisdom Trading"
  environment: "development"

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  cors_origins:
    - "*"
  workers: 1

# Sentiment Analysis Configuration
sentiment:
  language: "en"
  confidence_threshold: 0.6
  use_ensemble: true
  rule_weight: 0.4
  ml_weight: 0.6
  max_text_length: 10000
  cache_predictions: true

# Rule-based Analyzer Configuration
rule_based:
  case_sensitive: false
  intensifier_weight: 0.25
  diminisher_weight: 0.25
  negation_weight: 0.5
  lexicon_path: "./data/lexicons"

# ML Model Configuration
ml_model:
  model_path: "./data/models"
  model_name: "sentiment_model"
  vectorizer_name: "tfidf_vectorizer"
  max_features: 10000
  ngram_range: [1, 2]
  cache_predictions: true
  cache_max_size: 1000

# Storage Configuration
storage:
  type: "sqlite"
  path: "./data/sentiment_agent.db"
  echo: false

# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: "./logs/app.log"
  max_size_mb: 10
  backup_count: 5
  console_output: true
"""
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        logger.info(f"Created configuration file: {config_file}")
    else:
        logger.info(f"Configuration file already exists: {config_file}")

def main():
    """Main setup function."""
    logger.info("Starting setup of CWT X Sentiment Agent...")

    try:
        setup_directories()
        setup_default_lexicons()
        create_sample_config()

        logger.info("Setup completed successfully!")
        logger.info("You can now run the agent with: python src/agent.py")
        logger.info("Note: The first run may take a moment as it initializes the database and loads models.")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()