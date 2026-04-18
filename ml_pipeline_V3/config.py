"""Configuration objects and default constants for the gearbox pipeline."""

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_FS = 25600
DEFAULT_TRIM_START = 0.2
DEFAULT_WINDOW_SIZE = 12800
DEFAULT_OVERLAP = 0.0
DEFAULT_VIB_SOURCE = "Z"
DEFAULT_USE_MAGNITUDE = False
DEFAULT_ENABLE_GEAR_SPECIFIC = False
DEFAULT_K_BEST = 30
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
DEFAULT_DATA_PATH = r"C:/other senior project folder/raw data"


@dataclass
class DataConfig:
    data_path: str = DEFAULT_DATA_PATH
    fs: int = DEFAULT_FS
    trim_start: float = DEFAULT_TRIM_START
    vib_source: str = DEFAULT_VIB_SOURCE
    use_magnitude: bool = DEFAULT_USE_MAGNITUDE
    max_files: Optional[int] = None
    test_mode: bool = False


@dataclass
class WindowConfig:
    window_size: int = DEFAULT_WINDOW_SIZE
    overlap: float = DEFAULT_OVERLAP
    fs: int = DEFAULT_FS


@dataclass
class FeatureConfig:
    fs: int = DEFAULT_FS
    enable_gear_specific: bool = DEFAULT_ENABLE_GEAR_SPECIFIC
    estimate_gmf: bool = True


@dataclass
class ModelConfig:
    k_best: int = DEFAULT_K_BEST
    random_state: int = DEFAULT_RANDOM_STATE
    n_estimators: int = 200
    max_depth: int = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 2


@dataclass
class LeaveOneConditionOutConfig:
    condition: str = "speed"
    k_best: int = DEFAULT_K_BEST
    exclude_time_varying: bool = True
    verbose: bool = True
    compute_run_level: bool = True
    permutation_test: bool = False
    permutation_mode: str = "run"
    permutation_repeats: int = 1
    rng_seed: int = DEFAULT_RANDOM_STATE


@dataclass
class GroupKFoldConfig:
    k_best: int = DEFAULT_K_BEST
    n_splits: int = DEFAULT_N_SPLITS
    verbose: bool = True


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loso_speed: LeaveOneConditionOutConfig = field(
        default_factory=lambda: LeaveOneConditionOutConfig(
            condition="speed",
            k_best=DEFAULT_K_BEST,
            exclude_time_varying=True,
            verbose=True,
            compute_run_level=True,
            permutation_test=True,
            permutation_mode="run",
            permutation_repeats=100,
            rng_seed=DEFAULT_RANDOM_STATE,
        )
    )
    loso_load: LeaveOneConditionOutConfig = field(
        default_factory=lambda: LeaveOneConditionOutConfig(
            condition="load",
            k_best=DEFAULT_K_BEST,
            exclude_time_varying=True,
            verbose=True,
        )
    )
    loso_speed_load: LeaveOneConditionOutConfig = field(
        default_factory=lambda: LeaveOneConditionOutConfig(
            condition="speed_load",
            k_best=DEFAULT_K_BEST,
            exclude_time_varying=True,
            verbose=True,
        )
    )
    group_kfold: GroupKFoldConfig = field(default_factory=GroupKFoldConfig)
