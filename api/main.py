"""
api rest para el sistema de trading
endpoints: market data, signals, backtest, pipeline
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes import market, signals, backtest, pipeline, dashboard

app = FastAPI(
    title="Trading API",
    description="API para sistema de trading algoritmico",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market.router, prefix="/api", tags=["Market Data"])
app.include_router(signals.router, prefix="/api", tags=["Signals"])
app.include_router(backtest.router, prefix="/api", tags=["Backtest"])
app.include_router(pipeline.router, prefix="/api", tags=["Pipeline"])
app.include_router(dashboard.router, prefix="/api", tags=["Dashboard"])


@app.get("/")
async def root():
    return {"message": "Trading API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok"}


FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_PATH.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_PATH / "assets"), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = FRONTEND_PATH / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_PATH / "index.html")
