from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.database.base import Base

T = TypeVar("T", bound=Base)

class BaseRepository(Generic[T]):
    def __init__(self, session: AsyncSession, model: type[T]):
        self.session = session
        self.model = model

    async def get(self, id):
        return await self.session.get(self.model, id)

    async def list(self):
        result = await self.session.execute(select(self.model))

        return result.scalars().all()

    async def create(self, obj: T):
        self.session.add(obj)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj

    async def delete(self, obj: T):
        await self.session.delete(obj)