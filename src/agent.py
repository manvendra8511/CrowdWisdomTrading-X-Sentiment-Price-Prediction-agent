#!/usr/bin/env python3
"""
Main agent orchestrator for the CWT X Sentiment Agent.
Initializes all components and handles application lifecycle.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Add the project root to the Python path so we can import src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.routes import create_app
from src.core.utils.config import Config
from src.core.utils.helpers import setup_logging
from src.core.sentiment.analyzer import SentimentAnalyzer
from src.storage.repository import Repository


logger = logging.getLogger(__name__)


class SentimentAgent:
    """Main sentiment analysis agent orchestrator."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the sentiment agent.

        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.logger = setup_logging(self.config.get('logging', {}))
        self.repository: Optional[Repository] = None
        self.analyzer: Optional[SentimentAnalyzer] = None
        self.app = create_app(self.config.to_dict())
        self._lifespan = None

    async def initialize(self) -> None:
        """Use the same lifecycle as direct ASGI deployment."""
        if self._lifespan is not None:
            return
        context = self.app.router.lifespan_context(self.app)
        await context.__aenter__()
        self._lifespan = context
        self.repository = self.app.state.repository
        self.analyzer = self.app.state.analyzer

    async def shutdown(self) -> None:
        """Close the resources owned by this agent."""
        if self._lifespan is not None:
            context, self._lifespan = self._lifespan, None
            await context.__aexit__(None, None, None)

    async def run_api_server(self) -> None:
        """Run the API server."""
        import uvicorn

        host = self.config.get('api.host', '0.0.0.0')
        port = self.config.get('api.port', 8000)

        self.logger.info(f"Starting API server on {host}:{port}")

        config = uvicorn.Config(
            self.app,
            lifespan="off",
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )

        server = uvicorn.Server(config)

        # Run server until shutdown signal
        await server.serve()

    async def run(self) -> None:
        """Run the agent."""
        try:
            await self.initialize()

            # Run the API server (this will block until shutdown)
            await self.run_api_server()

        except Exception as e:
            self.logger.error(f"Error running Sentiment Agent: {e}")
            raise
        finally:
            await self.shutdown()


def main() -> None:
    """Main entry point."""
    agent = SentimentAgent()

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()