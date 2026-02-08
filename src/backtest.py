"""
backtesting y metricas de trading
evalua señales con reglas tp/sl
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TP_PERCENT = 0.05
SL_PERCENT = 0.03
MAX_HOLD_DAYS = 10

MODELS = {
    "GradientBoosting": Pipeline([("scaler", StandardScaler()), 
        ("clf", GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42))]),
    "RandomForest": Pipeline([("scaler", StandardScaler()), 
        ("clf", RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42))]),
    "LogisticRegression": Pipeline([("scaler", StandardScaler()), 
        ("clf", LogisticRegression(max_iter=500, random_state=42))])
}


def get_feature_cols(df: pl.DataFrame) -> List[str]:
    exclude = ["date", "symbol", "target", "signal", "probability", "predicted_probability"]
    numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
    return [c for c in df.columns if c not in exclude and df[c].dtype in numeric_types]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_proba)
    except:
        metrics["auc"] = 0.5
    return metrics


def evaluate_model_per_ticker(df: pl.DataFrame, train_ratio: float = 0.8) -> Dict[str, Dict[str, Any]]:
    """evalua modelos por ticker"""
    feature_cols = get_feature_cols(df)
    results = {}
    
    for symbol in df["symbol"].unique().to_list():
        symbol_df = df.filter(pl.col("symbol") == symbol).sort("date")
        if symbol_df.height < 50:
            continue
        
        split_idx = int(len(symbol_df) * train_ratio)
        train_df = symbol_df.head(split_idx)
        test_df = symbol_df.tail(len(symbol_df) - split_idx)
        
        if test_df.height < 10:
            continue
        
        X_train = np.nan_to_num(train_df.select(feature_cols).to_numpy(), nan=0)
        y_train = train_df["target"].to_numpy()
        X_test = np.nan_to_num(test_df.select(feature_cols).to_numpy(), nan=0)
        y_test = test_df["target"].to_numpy()
        
        model_results = {}
        best_model_name, best_auc = None, 0
        
        for model_name, model in MODELS.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics = calculate_metrics(y_test, y_pred, y_proba)
                model_results[model_name] = metrics
                if metrics["auc"] > best_auc:
                    best_auc = metrics["auc"]
                    best_model_name = model_name
            except Exception as e:
                model_results[model_name] = {"error": str(e)}
        
        results[symbol] = {
            "best_model": best_model_name, "best_auc": best_auc,
            "all_models": model_results,
            "train_samples": train_df.height, "test_samples": test_df.height
        }
        logger.info(f"  {symbol}: {best_model_name} (AUC={best_auc:.4f})")
    
    return results


def simulate_trade(prices: List[float], entry_idx: int, signal: str, 
                  tp_pct: float = TP_PERCENT, sl_pct: float = SL_PERCENT, 
                  max_hold: int = MAX_HOLD_DAYS) -> Dict[str, Any]:
    """simula trade individual con tp/sl"""
    if entry_idx >= len(prices) - 1:
        return {"status": "no_data", "pnl_pct": 0}
    
    entry_price = prices[entry_idx]
    direction = 1 if signal in ["BUY", "STRONG_BUY"] else -1
    
    if direction == 1:
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)
    
    exit_price, exit_reason, hold_days = entry_price, "max_hold", 0
    
    for i in range(1, min(max_hold + 1, len(prices) - entry_idx)):
        current_price = prices[entry_idx + i]
        hold_days = i
        
        if direction == 1:
            if current_price >= tp_price:
                exit_price, exit_reason = current_price, "take_profit"
                break
            elif current_price <= sl_price:
                exit_price, exit_reason = current_price, "stop_loss"
                break
        else:
            if current_price <= tp_price:
                exit_price, exit_reason = current_price, "take_profit"
                break
            elif current_price >= sl_price:
                exit_price, exit_reason = current_price, "stop_loss"
                break
        exit_price = current_price
    
    pnl_pct = (exit_price - entry_price) / entry_price * 100 if direction == 1 else (entry_price - exit_price) / entry_price * 100
    
    if exit_reason == "stop_loss" and pnl_pct < -sl_pct * 100 * 2:
        pnl_pct = -sl_pct * 100
    if exit_reason == "take_profit" and pnl_pct > tp_pct * 100 * 2:
        pnl_pct = tp_pct * 100
    
    return {
        "status": exit_reason, "entry_price": entry_price, "exit_price": exit_price,
        "pnl_pct": round(pnl_pct, 2), "hold_days": hold_days,
        "direction": "LONG" if direction == 1 else "SHORT"
    }


def backtest_signals(df: pl.DataFrame, tp_pct: float = TP_PERCENT, sl_pct: float = SL_PERCENT, 
                    max_hold: int = MAX_HOLD_DAYS) -> Dict[str, Any]:
    """backtesting de todas las señales"""
    logger.info("ejecutando backtest...")
    
    signals_df = df.filter(pl.col("signal").is_in(["BUY", "STRONG_BUY", "SELL"]))
    if signals_df.is_empty():
        return {"error": "no signals"}
    
    results_by_ticker = {}
    all_trades = []
    
    for symbol in signals_df["symbol"].unique().to_list():
        symbol_df = df.filter(pl.col("symbol") == symbol).sort("date")
        symbol_signals = signals_df.filter(pl.col("symbol") == symbol)
        
        prices = symbol_df["close"].to_list()
        dates = symbol_df["date"].to_list()
        trades = []
        
        for row in symbol_signals.iter_rows(named=True):
            try:
                entry_idx = dates.index(row["date"])
            except ValueError:
                continue
            
            trade = simulate_trade(prices, entry_idx, row["signal"], tp_pct, sl_pct, max_hold)
            trade.update({"symbol": symbol, "signal": row["signal"], "date": str(row["date"])[:10]})
            trades.append(trade)
            all_trades.append(trade)
        
        if trades:
            pnls = [t["pnl_pct"] for t in trades if t["status"] != "no_data"]
            wins = [t for t in trades if t["pnl_pct"] > 0]
            results_by_ticker[symbol] = {
                "total_trades": len(trades), "wins": len(wins), "losses": len(trades) - len(wins),
                "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
                "total_pnl": round(sum(pnls), 2), "avg_pnl": round(np.mean(pnls), 2) if pnls else 0,
                "trades": trades
            }
    
    all_pnls = [t["pnl_pct"] for t in all_trades if t["status"] != "no_data"]
    all_wins = [t for t in all_trades if t["pnl_pct"] > 0]
    
    by_signal = {}
    for signal_type in ["BUY", "STRONG_BUY", "SELL"]:
        signal_trades = [t for t in all_trades if t["signal"] == signal_type]
        signal_pnls = [t["pnl_pct"] for t in signal_trades if t["status"] != "no_data"]
        signal_wins = [t for t in signal_trades if t["pnl_pct"] > 0]
        by_signal[signal_type] = {
            "total_trades": len(signal_trades), "wins": len(signal_wins),
            "win_rate": round(len(signal_wins) / len(signal_trades) * 100, 1) if signal_trades else 0,
            "total_pnl": round(sum(signal_pnls), 2) if signal_pnls else 0,
            "avg_pnl": round(np.mean(signal_pnls), 2) if signal_pnls else 0
        }
    
    max_drawdown, sharpe_ratio = 0.0, 0.0
    if all_pnls:
        factors = [1 + (pnl / 100) for pnl in all_pnls]
        equity = [100]
        for f in factors:
            equity.append(equity[-1] * f)
        equity = np.array(equity)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max * 100
        max_drawdown = round(float(np.max(drawdowns)), 2)
        
        if len(all_pnls) > 1:
            avg_ret, std_ret = np.mean(all_pnls), np.std(all_pnls)
            if std_ret > 0:
                sharpe_ratio = round((avg_ret / std_ret) * np.sqrt(min(len(all_pnls), 250)), 2)
    
    global_stats = {
        "total_trades": len(all_trades), "wins": len(all_wins), "losses": len(all_trades) - len(all_wins),
        "win_rate": round(len(all_wins) / len(all_trades) * 100, 1) if all_trades else 0,
        "total_pnl": round(sum(all_pnls), 2), "avg_pnl": round(np.mean(all_pnls), 2) if all_pnls else 0,
        "max_win": round(max(all_pnls), 2) if all_pnls else 0,
        "max_loss": round(min(all_pnls), 2) if all_pnls else 0,
        "max_drawdown": max_drawdown, "sharpe_ratio": sharpe_ratio,
        "by_signal": by_signal,
        "config": {"tp_pct": tp_pct * 100, "sl_pct": sl_pct * 100, "max_hold_days": max_hold}
    }
    
    logger.info(f"backtest: {len(all_trades)} trades, WR={global_stats['win_rate']}%, P&L={global_stats['total_pnl']}%")
    return {"global": global_stats, "by_ticker": results_by_ticker}


def generate_full_report(df: pl.DataFrame, output_path: Optional[str] = None) -> Dict[str, Any]:
    """reporte completo con metricas y backtesting"""
    logger.info("=" * 60)
    logger.info("GENERANDO REPORTE")
    logger.info("=" * 60)
    
    model_metrics = evaluate_model_per_ticker(df)
    backtest_results = backtest_signals(df)
    
    report = {
        "model_metrics": model_metrics,
        "backtest": backtest_results,
        "summary": {
            "total_tickers": len(model_metrics),
            "avg_best_auc": round(np.mean([m["best_auc"] for m in model_metrics.values() if m.get("best_auc")]), 4),
            "model_distribution": {},
        }
    }
    
    for ticker, data in model_metrics.items():
        best = data.get("best_model", "N/A")
        report["summary"]["model_distribution"][best] = report["summary"]["model_distribution"].get(best, 0) + 1
    
    if output_path:
        import json
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"guardado: {output_path}")
    
    return report


if __name__ == "__main__":
    import pickle
    
    data_path = Path("../data")
    features_path = data_path / "processed" / "features.parquet"
    
    if features_path.exists():
        df = pl.read_parquet(features_path)
    else:
        print("no se encontro features.parquet")
        exit(1)
    
    model_path = data_path / "processed" / "models" / "sklearn_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
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
    
    generate_full_report(df, output_path="../backtest_report.json")
