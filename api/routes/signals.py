"""
endpoints de señales de trading
"""

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import Optional, List
import polars as pl
import pickle
import numpy as np

router = APIRouter()

DATA_PATH = Path(__file__).parent.parent.parent / "data"
FEATURES_PATH = DATA_PATH / "processed" / "features.parquet"
MODEL_PATH = DATA_PATH / "processed" / "models" / "sklearn_model.pkl"


def get_feature_cols(df: pl.DataFrame) -> List[str]:
    exclude = ["date", "symbol", "target", "signal", "probability", "predicted_probability"]
    numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
    return [c for c in df.columns if c not in exclude and df[c].dtype in numeric_types]


def load_signals_data() -> Optional[pl.DataFrame]:
    if not FEATURES_PATH.exists():
        return None
    
    df = pl.read_parquet(FEATURES_PATH)
    
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model_info = pickle.load(f)
        feature_cols = model_info.get("feature_cols") or get_feature_cols(df)
        X = np.nan_to_num(df.select(feature_cols).to_numpy(), nan=0)
        probs = model_info["model"].predict_proba(X)[:, 1]
        df = df.with_columns([pl.Series("probability", probs)])
    else:
        df = df.with_columns([pl.lit(0.5).alias("probability")])
    
    if "sma_50" not in df.columns:
        df = df.sort(["symbol", "date"])
        df = df.with_columns([pl.col("close").rolling_mean(50).over("symbol").alias("sma_50")])
        df = df.with_columns([
            pl.when(pl.col("sma_50").is_null()).then(pl.col("close")).otherwise(pl.col("sma_50")).alias("sma_50")
        ])
    
    df = df.with_columns([
        pl.when((pl.col("probability") > 0.70) & (pl.col("close") > pl.col("sma_50"))).then(pl.lit("STRONG_BUY"))
        .when(pl.col("probability") > 0.60).then(pl.lit("BUY"))
        .when(pl.col("probability") < 0.40).then(pl.lit("SELL"))
        .otherwise(pl.lit("HOLD")).alias("signal")
    ])
    
    return df


@router.get("/signals/{ticker}")
async def get_signals(
    ticker: str,
    signal_type: Optional[str] = Query(None, description="BUY, SELL, HOLD, STRONG_BUY"),
    limit: int = Query(100, le=500)
):
    """señales de un ticker"""
    df = load_signals_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No hay datos de señales")
    
    ticker_df = df.filter(pl.col("symbol") == ticker.upper())
    if ticker_df.is_empty():
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} no encontrado")
    
    if signal_type:
        ticker_df = ticker_df.filter(pl.col("signal") == signal_type.upper())
    
    ticker_df = ticker_df.sort("date", descending=True).head(limit)
    
    cols = ["date", "close", "probability", "signal"]
    available = [c for c in cols if c in ticker_df.columns]
    
    data = ticker_df.select(available).to_dicts()
    for row in data:
        if "date" in row:
            row["date"] = str(row["date"])[:10]
        if "probability" in row:
            row["probability"] = round(row["probability"], 4)
    
    signal_counts = ticker_df.group_by("signal").agg(pl.count().alias("count")).to_dicts()
    
    return {
        "ticker": ticker.upper(),
        "count": len(data),
        "signal_distribution": {s["signal"]: s["count"] for s in signal_counts},
        "signals": data
    }


@router.get("/signals")
async def get_all_signals(
    signal_type: Optional[str] = Query(None, description="BUY, SELL, STRONG_BUY"),
    limit: int = Query(50, le=200)
):
    """ultimas señales de todos los tickers"""
    df = load_signals_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No hay datos de señales")
    
    if signal_type:
        df = df.filter(pl.col("signal") == signal_type.upper())
    else:
        df = df.filter(pl.col("signal") != "HOLD")
    
    df = df.sort("date", descending=True).head(limit)
    
    cols = ["date", "symbol", "close", "probability", "signal"]
    available = [c for c in cols if c in df.columns]
    
    data = df.select(available).to_dicts()
    for row in data:
        if "date" in row:
            row["date"] = str(row["date"])[:10]
        if "probability" in row:
            row["probability"] = round(row["probability"], 4)
    
    return {"count": len(data), "signals": data}


@router.get("/signals/{ticker}/latest")
async def get_latest_signal(ticker: str):
    """ultima señal de un ticker"""
    df = load_signals_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No hay datos")
    
    ticker_df = df.filter(pl.col("symbol") == ticker.upper()).sort("date", descending=True)
    if ticker_df.is_empty():
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} no encontrado")
    
    latest = ticker_df.head(1).to_dicts()[0]
    latest["date"] = str(latest["date"])[:10]
    if "probability" in latest:
        latest["probability"] = round(latest["probability"], 4)
    
    return {"ticker": ticker.upper(), "latest": latest}
