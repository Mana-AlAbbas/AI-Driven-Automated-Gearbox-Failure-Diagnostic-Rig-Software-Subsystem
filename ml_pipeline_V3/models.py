"""Model-building helpers."""

import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .config import ModelConfig
except ImportError:
    from config import ModelConfig

warnings.filterwarnings("ignore")


def build_random_forest_pipeline(n_features: int, model_config: ModelConfig) -> Pipeline:
    k = int(min(max(1, model_config.k_best), n_features))
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=model_config.n_estimators,
                    max_depth=model_config.max_depth,
                    min_samples_split=model_config.min_samples_split,
                    min_samples_leaf=model_config.min_samples_leaf,
                    random_state=model_config.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
