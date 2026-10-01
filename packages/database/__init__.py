from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession


async def get_db() -> AsyncGenerator[AsyncSession]:
    from .session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        yield session
