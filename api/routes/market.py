"""
endpoints de datos de mercado
"""

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import polars as pl
import json

router = APIRouter()

DATA_PATH = Path(__file__).parent.parent.parent / "data"
FEATURES_PATH = DATA_PATH / "processed" / "features.parquet"
MARKET_PATH = DATA_PATH / "raw" / "market_data.parquet"


def load_market_data() -> Optional[pl.DataFrame]:
    if FEATURES_PATH.exists():
        return pl.read_parquet(FEATURES_PATH)
    if MARKET_PATH.exists():
        return pl.read_parquet(MARKET_PATH)
    return None


@router.get("/tickers")
async def get_tickers():
    """lista de tickers disponibles"""
    df = load_market_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No hay datos de mercado")
    
    tickers = df["symbol"].unique().sort().to_list()
    return {"tickers": tickers, "count": len(tickers)}


@router.get("/market/{ticker}")
async def get_market_data(
    ticker: str,
    start_date: Optional[str] = Query(None, description="Fecha inicio YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Fecha fin YYYY-MM-DD")
):
    """datos ohlcv de un ticker"""
    df = load_market_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No hay datos de mercado")
    
    ticker_df = df.filter(pl.col("symbol") == ticker.upper())
    if ticker_df.is_empty():
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} no encontrado")
    
    if start_date:
        ticker_df = ticker_df.filter(pl.col("date") >= datetime.fromisoformat(start_date))
    if end_date:
        ticker_df = ticker_df.filter(pl.col("date") <= datetime.fromisoformat(end_date))
    
    ticker_df = ticker_df.sort("date")
    
    cols = ["date", "open", "high", "low", "close", "volume"]
    available = [c for c in cols if c in ticker_df.columns]
    
    data = ticker_df.select(available).to_dicts()
    for row in data:
        if "date" in row:
            row["date"] = str(row["date"])[:10]
    
    return {
        "ticker": ticker.upper(),
        "count": len(data),
        "data": data
    }


@router.get("/market")
async def get_all_market_data(limit: int = Query(100, le=1000)):
    """datos de todos los tickers"""
    df = load_market_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No hay datos de mercado")
    
    latest = df.sort("date", descending=True).group_by("symbol").head(1)
    
    data = latest.to_dicts()
    for row in data:
        if "date" in row:
            row["date"] = str(row["date"])[:10]
    
    return {"count": len(data), "data": data[:limit]}
