"""
feature engineering con polars lazy evaluation
genera: retornos, volatilidad, momentum, volumen relativo, target
"""

import polars as pl
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_returns(df: pl.DataFrame, window: int) -> pl.DataFrame:
    return df.with_columns([pl.col('close').pct_change(window).alias(f'return_{window}d')])


def compute_volatility(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    if 'return_1d' not in df.columns:
        df = compute_returns(df, 1)
    return df.with_columns([pl.col('return_1d').rolling_std(window).alias(f'volatility_{window}d')])


def compute_momentum(df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
    return df.with_columns([
        ((pl.col('close') / pl.col('close').rolling_mean(window)) - 1.0).alias(f'momentum_{window}d')
    ])


def compute_relative_volume(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    return df.with_columns([
        (pl.col('volume') / pl.col('volume').rolling_mean(window)).alias(f'relative_volume_{window}d')
    ])


def create_target_variable(df: pl.DataFrame, forward_days: int = 5) -> pl.DataFrame:
    return df.with_columns([
        (pl.col('close').shift(-forward_days) > pl.col('close')).cast(pl.Int8).alias('target')
    ])


def generate_features(df: pl.DataFrame, forward_days: int = 5) -> pl.DataFrame:
    """genera features tecnicas (version estandar)"""
    logger.info("generando features...")
    
    df = df.sort(['symbol', 'date'])
    df = create_target_variable(df, forward_days)
    df = compute_returns(df, 1)
    df = compute_returns(df, 5)
    df = compute_volatility(df, window=20)
    df = compute_momentum(df, window=10)
    df = compute_relative_volume(df, window=20)
    
    initial = len(df)
    df = df.drop_nulls()
    logger.info(f"filas: {initial} -> {len(df)}")
    
    return df


def generate_features_lazy(df: pl.DataFrame, forward_days: int = 5) -> pl.DataFrame:
    """genera features con lazy evaluation de polars"""
    logger.info("generando features con lazy evaluation...")
    
    lf = df.lazy()
    lf = lf.sort(['symbol', 'date'])
    
    # target
    lf = lf.with_columns([
        (pl.col('close').shift(-forward_days) > pl.col('close')).cast(pl.Int8).alias('target')
    ])
    
    # retornos
    lf = lf.with_columns([
        pl.col('close').pct_change(1).alias('return_1d'),
        pl.col('close').pct_change(5).alias('return_5d')
    ])
    
    # volatilidad
    lf = lf.with_columns([pl.col('return_1d').rolling_std(20).alias('volatility_20d')])
    
    # momentum
    lf = lf.with_columns([
        ((pl.col('close') / pl.col('close').rolling_mean(10)) - 1.0).alias('momentum_10d')
    ])
    
    # volumen relativo
    lf = lf.with_columns([
        (pl.col('volume') / pl.col('volume').rolling_mean(20)).alias('relative_volume_20d')
    ])
    
    logger.info("ejecutando plan lazy...")
    result = lf.collect()
    
    initial = len(result)
    result = result.drop_nulls()
    logger.info(f"filas: {initial} -> {len(result)}")
    
    return result


def prepare_model_data(df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """split temporal 80/20"""
    df = df.sort('date')
    cutoff_idx = int(len(df) * 0.8)
    cutoff_date = df['date'][cutoff_idx]
    
    train_df = df.filter(pl.col('date') < cutoff_date)
    test_df = df.filter(pl.col('date') >= cutoff_date)
    
    logger.info(f"train: {len(train_df)}, test: {len(test_df)}")
    return train_df, test_df


def save_features(df: pl.DataFrame, output_path: str) -> None:
    df.write_parquet(output_path)
    logger.info(f"guardado: {output_path}")


def get_feature_columns(df: pl.DataFrame, exclude: List[str] = None) -> List[str]:
    if exclude is None:
        exclude = ['target', 'date', 'symbol']
    numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
    return [col for col, dtype in zip(df.columns, df.dtypes) if dtype in numeric_types and col not in exclude]
