"""Data loading utilities for the HUST gearbox dataset."""

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


class DataLoader:
    """Loader for the gearbox dataset."""

    def __init__(
        self,
        data_path: str,
        fs: int = 25600,
        trim_start: float = 0.2,
        vib_source: str = "Z",
        use_magnitude: bool = False,
    ):
        self.data_path = data_path
        self.fs = fs
        self.trim_samples = int(trim_start * fs)
        self.vib_source = vib_source.upper()
        self.use_magnitude = bool(use_magnitude)
        self.state_mapping = {"H": "Healthy", "B": "Broken", "M": "Missing"}

        if (not self.use_magnitude) and (self.vib_source not in {"X", "Y", "Z"}):
            raise ValueError("vib_source must be one of {'X','Y','Z'} or set use_magnitude=True.")

    def parse_filename(self, filename: str) -> Optional[Dict]:
        basename = os.path.basename(filename).replace(".txt", "")
        parts = basename.split("_")
        if len(parts) != 3:
            return None

        fault_code, speed_str, load_str = parts
        speed = speed_str if "-" in speed_str else float(speed_str)
        load_index = int(load_str)
        load_nm = [0, 0.113, 0.226, 0.339, 0.452][load_index]
        fault_state = self.state_mapping.get(fault_code, fault_code)

        return {
            "filename": basename,
            "fault_code": fault_code,
            "fault_state": fault_state,
            "speed": speed,
            "speed_str": speed_str,
            "load_index": load_index,
            "load_nm": load_nm,
            "is_time_varying": "-" in speed_str,
        }

    def load_single_file(self, filepath: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        metadata = self.parse_filename(filepath)
        if metadata is None:
            return None, None

        with open(filepath, "r", encoding="utf-8", errors="ignore") as file_obj:
            lines = file_obj.readlines()

        data_start = 0
        for index, line in enumerate(lines):
            if "Time (seconds) and Data Channels" in line:
                data_start = index + 2
                break

        if data_start == 0:
            return None, None

        rows: List[List[float]] = []
        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values = list(map(float, stripped.split()))
            except Exception:
                continue
            if len(values) >= 5:
                rows.append(values[:5])

        if not rows:
            return None, None

        arr = np.asarray(rows, dtype=float)
        time_col = arr[:, 0]
        speed_col = arr[:, 1]
        ax = arr[:, 2]
        ay = arr[:, 3]
        az = arr[:, 4]

        if self.use_magnitude:
            vibration = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
        elif self.vib_source == "X":
            vibration = ax
        elif self.vib_source == "Y":
            vibration = ay
        else:
            vibration = az

        if len(vibration) > self.trim_samples:
            vibration = vibration[self.trim_samples:]
            time_col = time_col[self.trim_samples:] - time_col[self.trim_samples]
            speed_col = speed_col[self.trim_samples:]

        metadata["original_length"] = len(vibration)
        metadata["duration"] = len(vibration) / self.fs
        metadata["has_speed_channel"] = True
        metadata["speed_channel_mean"] = float(np.mean(speed_col)) if len(speed_col) else np.nan
        metadata["start_time_after_trim"] = float(time_col[0]) if len(time_col) else 0.0
        return vibration, metadata

    def load_all_files(self, max_files: Optional[int] = None) -> Tuple[List[np.ndarray], List[Dict]]:
        all_files = sorted(glob.glob(os.path.join(self.data_path, "*.txt")))
        if max_files:
            all_files = all_files[:max_files]

        print(f"Found {len(all_files)} files in {self.data_path}")
        all_signals: List[np.ndarray] = []
        all_metadata: List[Dict] = []

        for filepath in tqdm(all_files, desc="Loading files"):
            vibration, metadata = self.load_single_file(filepath)
            if vibration is not None and metadata is not None:
                all_signals.append(vibration)
                all_metadata.append(metadata)

        return all_signals, all_metadata
