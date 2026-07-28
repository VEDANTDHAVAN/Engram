from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from packages.database import get_db

app = FastAPI(
    title="Engram",
    description="Temporal Memory Infrastructure for AI Applications",
    version="0.1.0",
)

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }

@app.get("/health/db")
async def db_health(db: AsyncSession = Depends(get_db)):
    await db.execute(text("SELECT 1"))

    return {
        "database": "connected"
    }