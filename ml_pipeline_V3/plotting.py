"""Plotting helpers for the gearbox pipeline."""

import glob
import os
from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from .data_loader import DataLoader
except ImportError:
    from data_loader import DataLoader


def visualize_sample_signals(
    data_folder_path: str,
    num_samples: int = 3,
    vib_source: str = "Z",
    use_magnitude: bool = False,
    fs: int = 25600,
):
    loader = DataLoader(
        data_folder_path,
        fs=fs,
        trim_start=0.2,
        vib_source=vib_source,
        use_magnitude=use_magnitude,
    )
    all_files = sorted(glob.glob(os.path.join(data_folder_path, "*.txt")))
    fault_files = {"Healthy": [], "Broken": [], "Missing": []}

    for filepath in all_files:
        metadata = loader.parse_filename(filepath)
        if metadata and metadata["fault_state"] in fault_files:
            fault_files[metadata["fault_state"]].append((filepath, metadata))

    fig, axes = plt.subplots(3, num_samples, figsize=(15, 10), squeeze=False)
    for row, (fault_type, files) in enumerate(fault_files.items()):
        for col, (filepath, metadata) in enumerate(files[:num_samples]):
            vibration, _ = loader.load_single_file(filepath)
            if vibration is None:
                continue

            axis = axes[row, col]
            time_axis = np.arange(len(vibration)) / fs
            plot_samples = min(len(vibration), 12800)
            axis.plot(time_axis[:plot_samples], vibration[:plot_samples])
            axis.set_xlabel("Time (s)")
            axis.set_ylabel("Amplitude (g)")
            axis.set_title(f"{fault_type}\n{metadata['speed_str']}Hz, Load{metadata['load_index']}")
            axis.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sample_signals.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    output_path: str,
    show: bool = False,
):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
