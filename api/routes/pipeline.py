"""
endpoints para ejecutar el pipeline
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import sys

router = APIRouter()

RUNNING = False


class PipelineConfig(BaseModel):
    skip_screener: bool = False
    skip_download: bool = False
    max_tickers: int = 25


class PipelineStatus(BaseModel):
    running: bool
    message: str


def run_pipeline_task(config: PipelineConfig):
    global RUNNING
    RUNNING = True
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.main import run_pipeline
        run_pipeline(
            skip_screener=config.skip_screener,
            skip_download=config.skip_download
        )
    except Exception as e:
        print(f"Error en pipeline: {e}")
    finally:
        RUNNING = False


@router.get("/pipeline/status")
async def get_pipeline_status():
    """estado del pipeline"""
    return {"running": RUNNING}


@router.post("/pipeline/run")
async def run_pipeline(config: PipelineConfig, background_tasks: BackgroundTasks):
    """ejecuta el pipeline en background"""
    global RUNNING
    
    if RUNNING:
        raise HTTPException(status_code=409, detail="Pipeline ya en ejecucion")
    
    background_tasks.add_task(run_pipeline_task, config)
    
    return {"message": "Pipeline iniciado", "config": config.dict()}


@router.get("/pipeline/data-status")
async def get_data_status():
    """estado de los datos disponibles"""
    DATA_PATH = Path(__file__).parent.parent.parent / "data"
    
    files = {
        "screener": (DATA_PATH / "raw" / "screener.csv").exists(),
        "market_data": (DATA_PATH / "raw" / "market_data.parquet").exists(),
        "features": (DATA_PATH / "processed" / "features.parquet").exists(),
        "model": (DATA_PATH / "processed" / "models" / "sklearn_model.pkl").exists(),
        "backtest_report": (Path(__file__).parent.parent.parent / "backtest_report.json").exists(),
        "dashboard": (Path(__file__).parent.parent.parent / "dashboard.html").exists(),
    }
    
    return {
        "files": files,
        "ready": all([files["features"], files["model"], files["backtest_report"]])
    }
