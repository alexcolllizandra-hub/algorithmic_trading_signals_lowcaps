# capa de procesamiento - eda y feature engineering
from .eda import (
    TargetDefinition, DataExplorer, DataCleaner, 
    LazyFeatureEngineer, StatisticalAnalyzer, run_full_eda
)
from .features import (
    generate_features, generate_features_lazy, 
    prepare_model_data, save_features, get_feature_columns
)
