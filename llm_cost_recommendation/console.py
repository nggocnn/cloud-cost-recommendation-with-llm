"""
Console script entry point for the LLM cost recommendation package.
"""
import asyncio
import sys

def main_sync():
    """Synchronous wrapper for the async main function"""
    try:
        from llm_cost_recommendation.cli import main
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_sync()
