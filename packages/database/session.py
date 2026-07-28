from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from packages.core.config import settings

engine = create_async_engine(
    settings.database_url, echo=settings.debug,
    future=True, pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession,
    autoflush=False, expire_on_commit=False,
)