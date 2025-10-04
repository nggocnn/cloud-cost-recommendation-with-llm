"""
Main entry point for the LLM cost recommendation package.
This allows running the package with: python -m llm_cost_recommendation
"""

import asyncio
import sys
from .cli import main
from .api import run_server
from .utils.logging import get_logger

if __name__ == "__main__":
    try:
        result = asyncio.run(main())

        # Handle special case for serve mode to avoid asyncio conflicts
        if isinstance(result, tuple) and result[0] == "serve":

            logger = get_logger(__name__)
            config = result[1]

            logger.info(
                "Starting API server mode",
                **{k: v for k, v in config.items() if k != "log_format"},
            )

            run_server(**config)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
