"""Dataset assembly utilities for feature-matrix creation."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .config import DataConfig, FeatureConfig, WindowConfig
    from .data_loader import DataLoader
    from .features import GearboxFeatureExtractor
    from .windowing import create_windows
except ImportError:
    from config import DataConfig, FeatureConfig, WindowConfig
    from data_loader import DataLoader
    from features import GearboxFeatureExtractor
    from windowing import create_windows


def process_hust_dataset(
    data_config: DataConfig,
    window_config: WindowConfig,
    feature_config: FeatureConfig,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    print("=" * 60)
    print("GEARBOX FAULT DIAGNOSIS PIPELINE")
    print("=" * 60)

    print("\n1. Loading data...")
    loader = DataLoader(
        data_path=data_config.data_path,
        fs=data_config.fs,
        trim_start=data_config.trim_start,
        vib_source=data_config.vib_source,
        use_magnitude=data_config.use_magnitude,
    )

    max_files = data_config.max_files
    if data_config.test_mode:
        max_files = 5 if max_files is None else min(5, max_files)

    all_signals, all_metadata = loader.load_all_files(max_files=max_files)
    print(f"   Loaded {len(all_signals)} files")

    print("\n2. Creating windows...")
    all_windows: List[np.ndarray] = []
    all_window_metadata: List[Dict] = []
    for signal, metadata in tqdm(
        zip(all_signals, all_metadata),
        total=len(all_signals),
        desc="Windowing",
    ):
        windows, window_meta = create_windows(
            signal,
            metadata,
            window_size=window_config.window_size,
            overlap=window_config.overlap,
            fs=window_config.fs,
        )
        all_windows.extend(windows)
        all_window_metadata.extend(window_meta)
    print(f"   Created {len(all_windows)} windows from {len(all_signals)} files")

    print("\n3. Extracting features...")
    extractor = GearboxFeatureExtractor(
        fs=feature_config.fs,
        enable_gear_specific=feature_config.enable_gear_specific,
        estimate_gmf=feature_config.estimate_gmf,
    )
    all_features: List[Dict[str, float]] = []
    for window, metadata in tqdm(
        zip(all_windows, all_window_metadata),
        total=len(all_windows),
        desc="Feature extraction",
    ):
        all_features.append(extractor.extract_all_features(window, metadata))

    print("\n4. Creating feature matrix...")
    x_df = pd.DataFrame(all_features)
    y = np.asarray([meta["fault_state"] for meta in all_window_metadata])
    groups = np.asarray([meta["filename"] for meta in all_window_metadata])
    metadata_df = pd.DataFrame(all_window_metadata)

    expected = {"Healthy", "Broken", "Missing"}
    actual = set(np.unique(y).tolist())
    if not actual.issubset(expected):
        raise ValueError(f"Unexpected labels found: {sorted(actual)}")

    print(f"   Feature matrix shape: {x_df.shape}")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Samples per class: {pd.Series(y).value_counts().to_dict()}")
    return x_df, y, groups, metadata_df
