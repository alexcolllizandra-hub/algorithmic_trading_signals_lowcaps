"""
generacion de señales de trading
convierte probabilidades en BUY, SELL, HOLD
"""

import polars as pl
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_signals(df: pl.DataFrame, buy_threshold: float = 0.6, sell_threshold: float = 0.4, 
                    prob_col: str = 'predicted_probability') -> pl.DataFrame:
    """genera señales basadas en probabilidades"""
    logger.info(f"señales: BUY>{buy_threshold}, SELL<{sell_threshold}")
    
    df = df.with_columns([
        pl.when(pl.col(prob_col) >= buy_threshold).then(pl.lit('BUY'))
        .when(pl.col(prob_col) <= sell_threshold).then(pl.lit('SELL'))
        .otherwise(pl.lit('HOLD')).alias('signal')
    ])
    
    signal_counts = df.group_by('signal').agg(pl.count().alias('count'))
    logger.info(f"distribucion: {signal_counts.to_dicts()}")
    
    return df


def get_signals_summary(df: pl.DataFrame) -> pl.DataFrame:
    """resumen de señales por simbolo"""
    return df.group_by(['symbol', 'signal']).agg([
        pl.count().alias('count'),
        pl.col('predicted_probability').mean().alias('avg_probability'),
        pl.col('date').max().alias('last_date')
    ]).sort(['symbol', 'signal'])


def save_signals(df: pl.DataFrame, output_path: str) -> None:
    output_cols = ['date', 'symbol', 'predicted_probability', 'signal']
    available_cols = [col for col in output_cols if col in df.columns]
    df.select(available_cols).write_csv(output_path)
    logger.info(f"guardado: {output_path}")
