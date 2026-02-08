"""
modelado con sklearn - usa timeseriessplit para evitar data leakage
modelos: GradientBoosting, RandomForest, LogisticRegression
"""

import polars as pl
from typing import Optional, Tuple, Any, Dict, List
import logging
import pickle
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """entrena modelos con timeseriessplit"""
    
    def __init__(self, target_col: str = 'target', n_splits: int = 5, random_state: int = 42):
        self.target_col = target_col
        self.n_splits = n_splits
        self.random_state = random_state
        self.model_info = None
    
    def _get_models(self) -> Dict[str, Any]:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        return {
            'GradientBoosting': Pipeline([('scaler', StandardScaler()), 
                ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=self.random_state))]),
            'RandomForest': Pipeline([('scaler', StandardScaler()), 
                ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1))]),
            'LogisticRegression': Pipeline([('scaler', StandardScaler()), 
                ('clf', LogisticRegression(max_iter=1000, random_state=self.random_state))])
        }
    
    def _prepare_data(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        exclude = [self.target_col, 'date', 'symbol']
        numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
        feature_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                       if dtype in numeric_types and col not in exclude]
        
        X = df.select(feature_cols).to_numpy()
        y = df[self.target_col].to_numpy()
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        return X[mask], y[mask], feature_cols
    
    def train(self, train_df: pl.DataFrame) -> Tuple[Dict[str, Any], Dict[str, float]]:
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        
        logger.info("entrenando con timeseriessplit...")
        train_df = train_df.sort('date')
        X, y, feature_cols = self._prepare_data(train_df)
        logger.info(f"datos: {X.shape[0]} x {X.shape[1]}")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        models = self._get_models()
        
        best_model, best_score, best_name = None, 0, ""
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
                logger.info(f"  {name}: AUC={scores.mean():.4f} (+/-{scores.std():.4f})")
                if scores.mean() > best_score:
                    best_score, best_model, best_name = scores.mean(), model, name
            except Exception as e:
                logger.warning(f"  {name}: error - {e}")
        
        logger.info(f"mejor: {best_name} (AUC={best_score:.4f})")
        best_model.fit(X, y)
        
        self.model_info = {
            'model': best_model, 'feature_cols': feature_cols,
            'model_name': best_name, 'auc': best_score
        }
        return self.model_info, {'auc': best_score, 'model_name': best_name}


class ModelPredictor:
    """genera predicciones"""
    
    def __init__(self, model_info: Dict[str, Any]):
        self.model = model_info['model']
        self.feature_cols = model_info['feature_cols']
        self.model_name = model_info.get('model_name', 'unknown')
    
    def predict(self, test_df: pl.DataFrame) -> pl.DataFrame:
        X = test_df.select(self.feature_cols).to_numpy()
        nan_mask = np.isnan(X).any(axis=1)
        X_clean = np.nan_to_num(X, nan=0)
        probs = self.model.predict_proba(X_clean)[:, 1]
        probs[nan_mask] = 0.5
        return test_df.with_columns([pl.Series('predicted_probability', probs)])


class ModelPersistence:
    """guarda/carga modelos"""
    
    @staticmethod
    def save(model_info: Dict[str, Any], model_path: str) -> None:
        path = Path(model_path)
        if path.suffix == '':
            path.mkdir(parents=True, exist_ok=True)
            model_file = path / 'sklearn_model.pkl'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            model_file = path
        with open(model_file, 'wb') as f:
            pickle.dump(model_info, f)
        logger.info(f"modelo guardado: {model_file}")
    
    @staticmethod
    def load(model_path: str) -> Optional[Dict[str, Any]]:
        try:
            path = Path(model_path)
            model_file = path / 'sklearn_model.pkl' if path.is_dir() else path
            if not model_file.exists():
                logger.error(f"no encontrado: {model_file}")
                return None
            with open(model_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"error: {e}")
            return None


# api compatible
def train_sklearn_model(train_df: pl.DataFrame, target_col: str = 'target') -> Tuple[Any, Dict[str, float]]:
    return ModelTrainer(target_col=target_col).train(train_df)

def predict_sklearn(model_info: Dict, test_df: pl.DataFrame, target_col: str = 'target') -> pl.DataFrame:
    return ModelPredictor(model_info).predict(test_df)

def save_sklearn_model(model_info: Dict, model_path: str) -> None:
    ModelPersistence.save(model_info, model_path)

def load_sklearn_model(model_path: str) -> Optional[Dict]:
    return ModelPersistence.load(model_path)

def train_model(train_df: pl.DataFrame, target_col: str = 'target', **kwargs) -> Tuple[Any, Any]:
    return ModelTrainer(target_col=target_col).train(train_df)

def predict(model: Any, test_df: pl.DataFrame, target_col: str = 'target') -> pl.DataFrame:
    return ModelPredictor(model).predict(test_df)

def save_model(model: Any, model_path: str) -> None:
    ModelPersistence.save(model, model_path)

def load_model(model_path: str) -> Optional[Any]:
    return ModelPersistence.load(model_path)
