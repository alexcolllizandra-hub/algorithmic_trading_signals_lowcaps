"""
pipeline principal de trading algoritmico

pasos:
1. screener - obtener tickers
2. market data - descargar historicos (yfinance)
3. eda - analisis exploratorio
4. features - polars lazy evaluation
5. modelo - sklearn con timeseriessplit
6. señales - generar buy/sell/hold
7. backtest - evaluar estrategia
8. dashboard - visualizacion html

arquitectura por capas:
- ingestion/      -> screener.py, market_data.py
- processing/     -> eda.py, features.py
- modeling/       -> model.py, signals.py, backtest.py
- persistence/    -> database.py
- presentation/   -> visualization.py
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import polars as pl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MONGO_ENABLED = False
try:
    from persistence.database import init_mongodb, save_market_data, save_signals, save_backtest_results
    MONGO_ENABLED = True
except ImportError:
    pass


class PipelineError(Exception):
    def __init__(self, step: str, message: str, original_error: Optional[Exception] = None):
        self.step, self.message, self.original_error = step, message, original_error
        super().__init__(f"[{step}] {message}")


def setup_directories(base_path: Path) -> Dict[str, Path]:
    dirs = {
        'data_raw': base_path / 'data' / 'raw',
        'data_processed': base_path / 'data' / 'processed',
        'models': base_path / 'data' / 'processed' / 'models'
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def run_step(step_name: str, step_func, *args, **kwargs) -> Any:
    try:
        logger.info(f"\n{'='*60}\n[{step_name}] iniciando...\n{'='*60}")
        result = step_func(*args, **kwargs)
        logger.info(f"[{step_name}] ok")
        return result
    except PipelineError:
        raise
    except Exception as e:
        logger.error(f"[{step_name}] error: {e}")
        raise PipelineError(step_name, str(e), e)


def step_1_screener(dirs: Dict[str, Path], skip: bool) -> 'pl.DataFrame':
    from ingestion.screener import get_small_cap_screener, save_screener_results, ScreenerError
    
    screener_path = dirs['data_raw'] / 'screener.csv'
    
    if skip and screener_path.exists():
        logger.info("cargando screener existente...")
        return pl.read_csv(screener_path)
    
    try:
        screener_df = get_small_cap_screener(min_price=0.0, max_price=10000.0, use_fallback=True)
    except ScreenerError as e:
        raise PipelineError("SCREENER", str(e), e)
    
    if screener_df.is_empty():
        raise PipelineError("SCREENER", "0 stocks")
    
    save_screener_results(screener_df, str(screener_path))
    logger.info(f"stocks: {screener_df['symbol'].to_list()}")
    return screener_df


def step_2_market_data(dirs: Dict[str, Path], symbols: list) -> 'pl.DataFrame':
    from ingestion.market_data import download_multiple_stocks, save_market_data
    
    market_df = download_multiple_stocks(symbols, period="1y", interval="1d")
    if market_df.is_empty():
        raise PipelineError("MARKET DATA", f"sin datos para: {symbols}")
    
    save_market_data(market_df, str(dirs['data_raw'] / 'market_data.parquet'))
    return market_df


def step_2b_eda(market_df: 'pl.DataFrame', dirs: Dict[str, Path]) -> Dict[str, Any]:
    from processing.eda import run_full_eda, DataExplorer, TargetDefinition
    
    logger.info("ejecutando eda...")
    
    target_def = TargetDefinition(forward_days=5, threshold_pct=0.0)
    logger.info(f"target: {target_def.get_description()}")
    
    explorer = DataExplorer(market_df)
    stats = explorer.get_basic_stats()
    logger.info(f"perimetro: {stats['n_rows']} filas, {stats['n_cols']} columnas")
    
    df_with_target = target_def.create_target(market_df)
    return run_full_eda(df_with_target, str(dirs['data_processed'] / 'eda_report.json'))


def step_3_features(market_df: 'pl.DataFrame', dirs: Dict[str, Path]) -> tuple:
    from processing.features import generate_features_lazy, prepare_model_data, save_features
    
    features_df = generate_features_lazy(market_df, forward_days=5)
    if features_df.is_empty():
        raise PipelineError("FEATURES", "no se pudieron generar features")
    
    logger.info(f"features: {features_df.shape}, columnas: {features_df.columns}")
    save_features(features_df, str(dirs['data_processed'] / 'features.parquet'))
    
    train_df, test_df = prepare_model_data(features_df)
    if train_df.is_empty() or test_df.is_empty():
        raise PipelineError("FEATURES", "train o test vacio")
    
    return train_df, test_df


def step_4_model(train_df: 'pl.DataFrame', dirs: Dict[str, Path], skip_training: bool):
    from modeling.model import ModelTrainer, ModelPersistence
    
    if skip_training:
        model_files = list(dirs['models'].glob("*"))
        if not model_files:
            raise PipelineError("MODELO", "no hay modelo guardado")
        return ModelPersistence.load(str(model_files[0]))
    
    logger.info("entrenando con timeseriessplit...")
    trainer = ModelTrainer(n_splits=5)
    model_info, metrics = trainer.train(train_df)
    logger.info(f"mejor: {metrics['model_name']} (AUC={metrics['auc']:.4f})")
    
    ModelPersistence.save(model_info, str(dirs['models']))
    return model_info


def step_5_signals(model_info, test_df: 'pl.DataFrame', dirs: Dict[str, Path]) -> 'pl.DataFrame':
    from modeling.model import predict
    from modeling.signals import generate_signals, save_signals
    
    predictions_df = predict(model_info, test_df)
    if predictions_df.is_empty():
        raise PipelineError("SEÑALES", "sin predicciones")
    
    signals_df = generate_signals(predictions_df, buy_threshold=0.6, sell_threshold=0.4)
    save_signals(signals_df, str(dirs['data_processed'] / 'signals.csv'))
    return signals_df


def step_5b_backtest(dirs: Dict[str, Path], base_path: Path) -> None:
    from modeling.backtest import generate_full_report, get_feature_cols
    import pickle
    import numpy as np
    
    features_path = dirs['data_processed'] / 'features.parquet'
    model_path = dirs['models'] / 'sklearn_model.pkl'
    
    if not features_path.exists():
        logger.warning("sin features.parquet")
        return
    
    df = pl.read_parquet(features_path)
    
    if model_path.exists():
        with open(model_path, "rb") as f:
            model_info = pickle.load(f)
        feature_cols = model_info.get("feature_cols") or get_feature_cols(df)
        X = np.nan_to_num(df.select(feature_cols).to_numpy(), nan=0)
        df = df.with_columns([pl.Series("probability", model_info["model"].predict_proba(X)[:, 1])])
    else:
        df = df.with_columns([pl.lit(0.5).alias("probability")])
    
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
    
    report = generate_full_report(df, output_path=str(base_path / 'backtest_report.json'))
    
    if "backtest" in report and "global" in report["backtest"]:
        bt = report["backtest"]["global"]
        logger.info(f"backtest: {bt['total_trades']} trades, WR={bt['win_rate']}%, P&L={bt['total_pnl']}%")


def step_6_dashboard(dirs: Dict[str, Path], base_path: Path) -> Path:
    from presentation.visualization import create_dashboard
    
    dashboard_path = base_path / 'dashboard.html'
    data_path = base_path / 'data'
    
    if not (data_path / 'processed' / 'training_data.parquet').exists() and \
       not (data_path / 'processed' / 'features.parquet').exists():
        raise PipelineError("DASHBOARD", f"sin datos en {data_path / 'processed'}")
    
    create_dashboard(str(data_path), str(dashboard_path))
    return dashboard_path


def run_pipeline(base_path: Optional[Path] = None, skip_screener: bool = False, skip_training: bool = False) -> bool:
    start_time = datetime.now()
    
    if base_path is None:
        base_path = Path(__file__).parent.parent
    
    logger.info("=" * 60)
    logger.info("PIPELINE DE TRADING")
    logger.info("=" * 60)
    
    try:
        dirs = setup_directories(base_path)
        
        mongo_connected = init_mongodb() if MONGO_ENABLED else False
        
        screener_df = run_step("PASO 1 - SCREENER", step_1_screener, dirs, skip_screener)
        symbols = screener_df['symbol'].to_list()
        
        market_df = run_step("PASO 2 - MARKET DATA", step_2_market_data, dirs, symbols)
        if mongo_connected:
            save_market_data(market_df)
        
        run_step("PASO 2b - EDA", step_2b_eda, market_df, dirs)
        
        train_df, test_df = run_step("PASO 3 - FEATURES", step_3_features, market_df, dirs)
        
        model_info = run_step("PASO 4 - MODELO", step_4_model, train_df, dirs, skip_training)
        
        signals_df = run_step("PASO 5 - SEÑALES", step_5_signals, model_info, test_df, dirs)
        if mongo_connected:
            save_signals(signals_df)
        
        run_step("PASO 5b - BACKTEST", step_5b_backtest, dirs, base_path)
        
        dashboard_path = run_step("PASO 6 - DASHBOARD", step_6_dashboard, dirs, base_path)
        
        elapsed = datetime.now() - start_time
        logger.info(f"\n{'='*60}\nPIPELINE OK - {elapsed}\n{'='*60}")
        logger.info(f"dashboard: {dashboard_path}")
        
        return True
        
    except PipelineError as e:
        logger.error(f"\n{'='*60}\nPIPELINE ERROR\n{'='*60}")
        logger.error(f"paso: {e.step}, mensaje: {e.message}")
        return False
    except Exception as e:
        logger.error(f"error inesperado: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = run_pipeline(
        skip_screener="--skip-screener" in sys.argv,
        skip_training="--skip-training" in sys.argv
    )
    sys.exit(0 if success else 1)
