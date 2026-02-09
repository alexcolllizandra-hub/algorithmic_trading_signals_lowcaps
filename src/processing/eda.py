"""
modulo de eda - analisis exploratorio de datos
implementa: variable objetivo, perimetro, eda, limpieza, analisis estadistico
usa polars con lazy evaluation
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TargetDefinition:
    """define la variable objetivo binaria: 1 si sube, 0 si no"""
    
    def __init__(self, forward_days: int = 5, threshold_pct: float = 0.0, target_col: str = 'target'):
        self.forward_days = forward_days
        self.threshold_pct = threshold_pct
        self.target_col = target_col
        logger.info(f"target: sube >{threshold_pct}% en {forward_days} dias")
    
    def create_target(self, df: pl.DataFrame) -> pl.DataFrame:
        threshold_factor = 1 + (self.threshold_pct / 100)
        return df.with_columns([
            (pl.col('close').shift(-self.forward_days) > pl.col('close') * threshold_factor)
            .cast(pl.Int8).alias(self.target_col)
        ])
    
    def get_description(self) -> str:
        return f"clasificacion binaria: predice si el precio subira mas de {self.threshold_pct}% en los proximos {self.forward_days} dias"


class DataExplorer:
    """analisis exploratorio: estadisticas, nulls, distribucion target, correlaciones"""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
    
    def get_basic_stats(self) -> Dict[str, Any]:
        stats = {
            'n_rows': self.df.height,
            'n_cols': self.df.width,
            'columns': self.df.columns,
            'dtypes': {col: str(dtype) for col, dtype in zip(self.df.columns, self.df.dtypes)},
            'memory_mb': self.df.estimated_size() / (1024 * 1024)
        }
        logger.info(f"dataset: {stats['n_rows']} filas x {stats['n_cols']} columnas")
        return stats
    
    def get_numeric_stats(self) -> pl.DataFrame:
        numeric_cols = [col for col, dtype in zip(self.df.columns, self.df.dtypes)
                       if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]]
        if not numeric_cols:
            return pl.DataFrame()
        
        stats_list = []
        for col in numeric_cols:
            col_data = self.df[col].drop_nulls()
            stats_list.append({
                'column': col, 'count': col_data.len(),
                'null_count': self.df[col].null_count(),
                'null_pct': round(self.df[col].null_count() / self.df.height * 100, 2),
                'mean': round(col_data.mean(), 4) if col_data.len() > 0 else None,
                'std': round(col_data.std(), 4) if col_data.len() > 0 else None,
                'min': round(col_data.min(), 4) if col_data.len() > 0 else None,
                'q25': round(col_data.quantile(0.25), 4) if col_data.len() > 0 else None,
                'median': round(col_data.median(), 4) if col_data.len() > 0 else None,
                'q75': round(col_data.quantile(0.75), 4) if col_data.len() > 0 else None,
                'max': round(col_data.max(), 4) if col_data.len() > 0 else None
            })
        return pl.DataFrame(stats_list)
    
    def get_missing_analysis(self) -> pl.DataFrame:
        missing_data = []
        for col in self.df.columns:
            null_count = self.df[col].null_count()
            missing_data.append({
                'column': col, 'null_count': null_count,
                'null_pct': round(null_count / self.df.height * 100, 2),
                'has_nulls': null_count > 0
            })
        return pl.DataFrame(missing_data).sort('null_count', descending=True)
    
    def get_target_distribution(self, target_col: str = 'target') -> Dict[str, Any]:
        if target_col not in self.df.columns:
            return {'error': f'columna {target_col} no encontrada'}
        
        distribution = self.df.group_by(target_col).agg(pl.count().alias('count')).sort(target_col)
        total = self.df.height
        dist_dict = {}
        
        for row in distribution.iter_rows(named=True):
            label = row[target_col]
            count = row['count']
            dist_dict[f'class_{label}'] = {'count': count, 'pct': round(count / total * 100, 2)}
        
        if len(dist_dict) == 2:
            counts = [v['count'] for v in dist_dict.values()]
            dist_dict['balance_ratio'] = round(min(counts) / max(counts), 3)
            dist_dict['is_balanced'] = dist_dict['balance_ratio'] > 0.4
        
        logger.info(f"distribucion del target:")
        for k, v in dist_dict.items():
            if isinstance(v, dict):
                logger.info(f"  {k}: {v['count']} ({v['pct']}%)")
        return dist_dict
    
    def get_correlations(self, target_col: str = 'target') -> pl.DataFrame:
        numeric_cols = [col for col, dtype in zip(self.df.columns, self.df.dtypes)
                       if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32] and col != target_col]
        
        if target_col not in self.df.columns:
            return pl.DataFrame()
        
        correlations = []
        target_series = self.df[target_col].to_numpy()
        
        for col in numeric_cols:
            col_series = self.df[col].to_numpy()
            mask = ~(np.isnan(target_series) | np.isnan(col_series))
            if mask.sum() > 10:
                corr = np.corrcoef(target_series[mask], col_series[mask])[0, 1]
                correlations.append({'feature': col, 'correlation': round(corr, 4), 'abs_correlation': round(abs(corr), 4)})
        
        return pl.DataFrame(correlations).sort('abs_correlation', descending=True)
    
    def get_temporal_analysis(self, date_col: str = 'date') -> Dict[str, Any]:
        if date_col not in self.df.columns:
            return {'error': f'columna {date_col} no encontrada'}
        dates = self.df[date_col].drop_nulls()
        return {
            'min_date': str(dates.min()), 'max_date': str(dates.max()),
            'n_days': dates.n_unique(),
            'date_range_days': (dates.max() - dates.min()).days if hasattr(dates.max() - dates.min(), 'days') else None
        }
    
    def get_symbol_analysis(self, symbol_col: str = 'symbol') -> Dict[str, Any]:
        if symbol_col not in self.df.columns:
            return {'error': f'columna {symbol_col} no encontrada'}
        symbol_counts = self.df.group_by(symbol_col).agg(pl.count().alias('count')).sort('count', descending=True)
        counts = symbol_counts['count'].to_list()
        return {
            'n_symbols': self.df[symbol_col].n_unique(),
            'avg_rows_per_symbol': round(np.mean(counts), 1),
            'min_rows_per_symbol': min(counts), 'max_rows_per_symbol': max(counts),
            'symbols': symbol_counts.head(10).to_dicts()
        }
    
    def generate_report(self, target_col: str = 'target') -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("GENERANDO REPORTE DE ANALISIS EXPLORATORIO (EDA)")
        logger.info("=" * 60)
        report = {
            'basic_stats': self.get_basic_stats(),
            'numeric_stats': self.get_numeric_stats().to_dicts(),
            'missing_analysis': self.get_missing_analysis().to_dicts(),
            'target_distribution': self.get_target_distribution(target_col),
            'correlations': self.get_correlations(target_col).to_dicts(),
            'temporal_analysis': self.get_temporal_analysis(),
            'symbol_analysis': self.get_symbol_analysis()
        }
        logger.info("reporte de eda generado correctamente")
        return report


class DataCleaner:
    """limpieza: nulls, duplicados, precios invalidos, ordenamiento"""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.cleaning_log = []
    
    def remove_nulls(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        initial = self.df.height
        self.df = self.df.drop_nulls(subset=subset) if subset else self.df.drop_nulls()
        self.cleaning_log.append(f"nulls eliminados: {initial - self.df.height}")
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        initial = self.df.height
        self.df = self.df.unique(subset=subset) if subset else self.df.unique()
        self.cleaning_log.append(f"duplicados eliminados: {initial - self.df.height}")
        return self
    
    def filter_valid_prices(self, price_col: str = 'close', min_price: float = 0.01) -> 'DataCleaner':
        initial = self.df.height
        self.df = self.df.filter(pl.col(price_col) >= min_price)
        self.cleaning_log.append(f"precios invalidos: {initial - self.df.height}")
        return self
    
    def sort_temporal(self, date_col: str = 'date', symbol_col: str = 'symbol') -> 'DataCleaner':
        self.df = self.df.sort([symbol_col, date_col])
        self.cleaning_log.append("ordenado temporalmente")
        return self
    
    def get_result(self) -> pl.DataFrame:
        return self.df
    
    def get_cleaning_log(self) -> List[str]:
        return self.cleaning_log


class LazyFeatureEngineer:
    """feature engineering con polars lazy evaluation"""
    
    def __init__(self, df: pl.DataFrame):
        self.lf = df.lazy()
        self.feature_log = []
        logger.info("lazy evaluation activada")
    
    def add_returns(self, windows: List[int] = [1, 5]) -> 'LazyFeatureEngineer':
        for w in windows:
            self.lf = self.lf.with_columns([pl.col('close').pct_change(w).alias(f'return_{w}d')])
            self.feature_log.append(f'return_{w}d')
        return self
    
    def add_volatility(self, window: int = 20) -> 'LazyFeatureEngineer':
        if 'return_1d' not in self.lf.columns:
            self.add_returns([1])
        self.lf = self.lf.with_columns([pl.col('return_1d').rolling_std(window).alias(f'volatility_{window}d')])
        self.feature_log.append(f'volatility_{window}d')
        return self
    
    def add_momentum(self, window: int = 10) -> 'LazyFeatureEngineer':
        self.lf = self.lf.with_columns([
            ((pl.col('close') / pl.col('close').rolling_mean(window)) - 1.0).alias(f'momentum_{window}d')
        ])
        self.feature_log.append(f'momentum_{window}d')
        return self
    
    def add_relative_volume(self, window: int = 20) -> 'LazyFeatureEngineer':
        self.lf = self.lf.with_columns([
            (pl.col('volume') / pl.col('volume').rolling_mean(window)).alias(f'relative_volume_{window}d')
        ])
        self.feature_log.append(f'relative_volume_{window}d')
        return self
    
    def add_target(self, forward_days: int = 5, threshold_pct: float = 0.0) -> 'LazyFeatureEngineer':
        threshold_factor = 1 + (threshold_pct / 100)
        self.lf = self.lf.with_columns([
            (pl.col('close').shift(-forward_days) > pl.col('close') * threshold_factor)
            .cast(pl.Int8).alias('target')
        ])
        self.feature_log.append('target')
        return self
    
    def collect(self) -> pl.DataFrame:
        logger.info(f"ejecutando plan lazy, features: {self.feature_log}")
        return self.lf.collect()
    
    def get_feature_log(self) -> List[str]:
        return self.feature_log


class StatisticalAnalyzer:
    """analisis estadistico: estacionariedad, distribucion de retornos"""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
    
    def test_stationarity(self, col: str, symbol: str = None) -> Dict[str, Any]:
        data = self.df.filter(pl.col('symbol') == symbol)[col].to_numpy() if symbol else self.df[col].to_numpy()
        data = data[~np.isnan(data)]
        if len(data) < 30:
            return {'error': 'datos insuficientes'}
        
        n = len(data)
        first_half = np.mean(data[:n//2])
        second_half = np.mean(data[n//2:])
        mean_diff = abs(second_half - first_half) / abs(first_half) * 100
        
        return {
            'column': col, 'symbol': symbol, 'n_observations': n,
            'overall_mean': round(np.mean(data), 4), 'overall_std': round(np.std(data), 4),
            'first_half_mean': round(first_half, 4), 'second_half_mean': round(second_half, 4),
            'mean_change_pct': round(mean_diff, 2), 'likely_stationary': mean_diff < 20
        }
    
    def analyze_returns_distribution(self, return_col: str = 'return_1d') -> Dict[str, Any]:
        if return_col not in self.df.columns:
            return {'error': f'columna {return_col} no encontrada'}
        
        returns = self.df[return_col].drop_nulls().to_numpy()
        if len(returns) < 10:
            return {'error': 'datos insuficientes'}
        
        mean, std = np.mean(returns), np.std(returns)
        skewness = np.mean(((returns - mean) / std) ** 3) if std > 0 else 0
        kurtosis = np.mean(((returns - mean) / std) ** 4) - 3 if std > 0 else 0
        
        return {
            'column': return_col, 'n_observations': len(returns),
            'mean': round(mean, 6), 'std': round(std, 6),
            'skewness': round(skewness, 4), 'kurtosis': round(kurtosis, 4),
            'p1': round(np.percentile(returns, 1), 4), 'p99': round(np.percentile(returns, 99), 4),
            'is_normal_like': abs(skewness) < 0.5 and abs(kurtosis) < 1
        }
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'returns_distribution': self.analyze_returns_distribution(),
            'stationarity_check': self.test_stationarity('close')
        }


def run_full_eda(df: pl.DataFrame, output_path: Optional[str] = None) -> Dict[str, Any]:
    """ejecuta pipeline completo de eda"""
    logger.info("=" * 60)
    logger.info("ANALISIS EXPLORATORIO COMPLETO")
    logger.info("=" * 60)
    
    target_def = TargetDefinition(forward_days=5, threshold_pct=0.0)
    explorer = DataExplorer(df)
    analyzer = StatisticalAnalyzer(df)
    
    full_report = {
        'target_definition': {
            'description': target_def.get_description(),
            'forward_days': target_def.forward_days,
            'threshold_pct': target_def.threshold_pct
        },
        'exploratory_analysis': explorer.generate_report(),
        'statistical_analysis': analyzer.get_summary()
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        logger.info(f"reporte guardado: {output_path}")
    
    return full_report
