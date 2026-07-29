from sqlalchemy.ext.asyncio import AsyncSession

from packages.core.schemas.memory import CreateMemoryRequest
from packages.database.models.memory import Memory
from packages.database.repositories.memory import MemoryRepository

class MemoryService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = MemoryRepository(session)

    async def create(self, request: CreateMemoryRequest) -> Memory:
        memory = Memory(
            namespace=request.namespace,
            content=request.content,
            embedding=request.embedding,
            confidence=request.confidence,
            importance=request.importance,
        )

        await self.repository.create(memory)
        await self.session.commit()
        await self.session.refresh(memory)

        return memory

    async def get(self, memory_id):
        return await self.repository.get(memory_id)

    async def list(self):
        return await self.repository.list()