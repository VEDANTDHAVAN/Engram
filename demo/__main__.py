"""Top-level entry point: ``python -m demo``.

Delegates to the interactive Engram demo CLI in ``packages.demo``.
"""

from packages.demo.__main__ import main

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())