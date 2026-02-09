"""
endpoint agregado para el dashboard frontend
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Optional, List
import polars as pl
import pickle
import numpy as np
import json

router = APIRouter()

DATA_PATH = Path(__file__).parent.parent.parent / "data"
FEATURES_PATH = DATA_PATH / "processed" / "features.parquet"
MODEL_PATH = DATA_PATH / "processed" / "models" / "sklearn_model.pkl"
REPORT_PATH = Path(__file__).parent.parent.parent / "backtest_report.json"


def get_feature_cols(df: pl.DataFrame) -> List[str]:
    exclude = ["date", "symbol", "target", "signal", "probability", "predicted_probability"]
    numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
    return [c for c in df.columns if c not in exclude and df[c].dtype in numeric_types]


@router.get("/dashboard")
async def get_dashboard():
    """datos agregados para el dashboard completo"""
    
    if not FEATURES_PATH.exists():
        raise HTTPException(status_code=404, detail="No hay datos. Ejecuta el pipeline primero.")
    
    df = pl.read_parquet(FEATURES_PATH)
    
    model_info = {"name": "No disponible", "auc": 0}
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        feature_cols = model_data.get("feature_cols") or get_feature_cols(df)
        X = np.nan_to_num(df.select(feature_cols).to_numpy(), nan=0)
        probs = model_data["model"].predict_proba(X)[:, 1]
        df = df.with_columns([pl.Series("probability", probs)])
        model_info = {
            "name": model_data.get("model_name", "sklearn"),
            "auc": model_data.get("auc", 0)
        }
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
    
    tickers = df["symbol"].unique().sort().to_list()
    
    latest_signals = []
    for ticker in tickers:
        ticker_df = df.filter(pl.col("symbol") == ticker).sort("date", descending=True).head(1)
        if ticker_df.height > 0:
            row = ticker_df.to_dicts()[0]
            latest_signals.append({
                "symbol": row["symbol"],
                "date": str(row["date"])[:10],
                "price": round(row.get("close", 0), 2),
                "probability": round(row.get("probability", 0.5), 4),
                "signal": row.get("signal", "HOLD"),
                "sma_50": round(row.get("sma_50", 0), 2) if row.get("sma_50") else None
            })
    
    buy_signals = [s for s in latest_signals if s["signal"] in ["BUY", "STRONG_BUY"]]
    best_opportunity = None
    if buy_signals:
        best_opportunity = max(buy_signals, key=lambda x: x["probability"])
    elif latest_signals:
        best_opportunity = max(latest_signals, key=lambda x: x["probability"])
    
    historics = {}
    for ticker in tickers:
        ticker_df = df.filter(pl.col("symbol") == ticker).sort("date")
        cols = ["date", "open", "high", "low", "close", "probability", "signal"]
        available = [c for c in cols if c in ticker_df.columns]
        data = ticker_df.select(available).to_dicts()
        for row in data:
            row["date"] = str(row["date"])[:10]
            if "probability" in row:
                row["probability"] = round(row["probability"], 4)
        historics[ticker] = data
    
    backtest_stats = None
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r") as f:
            report = json.load(f)
        backtest_stats = report.get("backtest", {}).get("global", {})
    
    return {
        "tickers": tickers,
        "total_tickers": len(tickers),
        "signals": latest_signals,
        "best_opportunity": best_opportunity,
        "historics": historics,
        "model_info": model_info,
        "backtest": backtest_stats
    }


@router.post("/run-pipeline")
async def trigger_pipeline():
    """endpoint para ejecutar pipeline desde frontend"""
    from fastapi import BackgroundTasks
    import sys
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.main import run_pipeline
        run_pipeline(skip_screener=False, skip_download=False)
        return {"status": "ok", "message": "Pipeline ejecutado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
