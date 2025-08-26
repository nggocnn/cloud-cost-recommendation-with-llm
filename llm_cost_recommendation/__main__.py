"""
Main entry point for the LLM cost recommendation package.
This allows running the package with: python -m llm_cost_recommendation
"""
import asyncio
import sys
from .cli import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
