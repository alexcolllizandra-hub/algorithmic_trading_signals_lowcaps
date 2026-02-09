"""
endpoints de backtesting
"""

from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import Optional
import json

router = APIRouter()

REPORT_PATH = Path(__file__).parent.parent.parent / "backtest_report.json"


def load_backtest_report() -> Optional[dict]:
    if not REPORT_PATH.exists():
        return None
    with open(REPORT_PATH, "r") as f:
        return json.load(f)


@router.get("/backtest")
async def get_backtest_summary():
    """resumen global del backtest"""
    report = load_backtest_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No hay resultados de backtest")
    
    global_stats = report.get("backtest", {}).get("global", {})
    summary = report.get("summary", {})
    
    return {
        "global_stats": global_stats,
        "model_summary": summary,
        "tickers_analyzed": len(report.get("model_metrics", {}))
    }


@router.get("/backtest/kpis")
async def get_kpis():
    """kpis principales para dashboard"""
    report = load_backtest_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No hay resultados de backtest")
    
    global_stats = report.get("backtest", {}).get("global", {})
    
    return {
        "total_return": global_stats.get("total_pnl", 0),
        "win_rate": global_stats.get("win_rate", 0),
        "total_trades": global_stats.get("total_trades", 0),
        "max_drawdown": global_stats.get("max_drawdown", 0),
        "sharpe_ratio": global_stats.get("sharpe_ratio", 0),
        "avg_pnl": global_stats.get("avg_pnl", 0),
        "by_signal": global_stats.get("by_signal", {})
    }


@router.get("/backtest/trades/{ticker}")
async def get_ticker_trades(ticker: str, limit: int = Query(50, le=200)):
    """trades ejecutados para un ticker"""
    report = load_backtest_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No hay resultados de backtest")
    
    ticker_upper = ticker.upper()
    ticker_data = report.get("backtest", {}).get("by_ticker", {}).get(ticker_upper)
    
    if not ticker_data:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} no encontrado")
    
    trades = ticker_data.get("trades", [])[:limit]
    
    return {
        "ticker": ticker_upper,
        "total_trades": ticker_data.get("total_trades", 0),
        "win_rate": ticker_data.get("win_rate", 0),
        "total_pnl": ticker_data.get("total_pnl", 0),
        "trades": trades
    }


@router.get("/backtest/{ticker}")
async def get_backtest_ticker(ticker: str):
    """resultados de backtest para un ticker"""
    report = load_backtest_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No hay resultados de backtest")
    
    ticker_upper = ticker.upper()
    
    backtest_data = report.get("backtest", {}).get("by_ticker", {}).get(ticker_upper)
    model_data = report.get("model_metrics", {}).get(ticker_upper)
    
    if not backtest_data and not model_data:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} no encontrado en backtest")
    
    return {
        "ticker": ticker_upper,
        "backtest": backtest_data,
        "model_metrics": model_data
    }


@router.get("/model/metrics")
async def get_model_metrics():
    """metricas del modelo ml"""
    report = load_backtest_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No hay resultados")
    
    model_metrics = report.get("model_metrics", {})
    summary = report.get("summary", {})
    
    ticker_results = []
    for ticker, data in model_metrics.items():
        ticker_results.append({
            "ticker": ticker,
            "best_model": data.get("best_model"),
            "auc": data.get("best_auc", 0),
            "train_samples": data.get("train_samples", 0),
            "test_samples": data.get("test_samples", 0)
        })
    
    ticker_results.sort(key=lambda x: x["auc"], reverse=True)
    
    return {
        "avg_auc": summary.get("avg_best_auc", 0),
        "model_distribution": summary.get("model_distribution", {}),
        "by_ticker": ticker_results
    }
