from fastapi import FastAPI

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