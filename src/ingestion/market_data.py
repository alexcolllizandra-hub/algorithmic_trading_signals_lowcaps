"""
descarga de datos historicos con yfinance
ohlcv diarios del ultimo año
"""

import yfinance as yf
import polars as pl
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_stock_history(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pl.DataFrame]:
    """descarga datos historicos de un stock"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"sin datos para {symbol}")
            return None
        
        df_polars = pl.from_pandas(df.reset_index())
        
        if 'Date' in df_polars.columns:
            df_polars = df_polars.rename({'Date': 'date'})
        elif 'Datetime' in df_polars.columns:
            df_polars = df_polars.rename({'Datetime': 'date'})
        
        df_polars = df_polars.with_columns(pl.col('date').cast(pl.Datetime))
        
        columns_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        for old_col, new_col in columns_map.items():
            if old_col in df_polars.columns:
                df_polars = df_polars.rename({old_col: new_col})
        
        df_polars = df_polars.with_columns(pl.lit(symbol).alias('symbol'))
        
        final_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        df_polars = df_polars.select([col for col in final_columns if col in df_polars.columns])
        
        return df_polars
        
    except Exception as e:
        logger.error(f"error descargando {symbol}: {e}")
        return None


def download_multiple_stocks(symbols: List[str], period: str = "1y", interval: str = "1d", delay: float = 0.5) -> pl.DataFrame:
    """descarga datos para multiples stocks"""
    all_data = []
    logger.info(f"descargando datos para {len(symbols)} stocks")
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"procesando {symbol} ({i}/{len(symbols)})")
        df = download_stock_history(symbol, period, interval)
        
        if df is not None and not df.is_empty():
            all_data.append(df)
        
        if i < len(symbols):
            time.sleep(delay)
    
    if not all_data:
        logger.warning("no se descargaron datos")
        return pl.DataFrame()
    
    result = pl.concat(all_data)
    logger.info(f"total registros: {len(result)}")
    return result


def save_market_data(df: pl.DataFrame, output_path: str) -> None:
    df.write_parquet(output_path)
    logger.info(f"guardado: {output_path}")
