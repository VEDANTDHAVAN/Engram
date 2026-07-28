from sqlalchemy import select

from packages.database.models.memory import Memory
from packages.database.repositories.base import BaseRepository

class MemoryRepository(BaseRepository[Memory]):
    def __init__(self, session):
        super().__init__(session, Memory)

    async def by_namespace(self, namespace: str):
        result = await self.session.execute(
            select(Memory).where(
                Memory.namespace == namespace
            )
        )

        return result.scalars().all()