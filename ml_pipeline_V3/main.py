"""Main entry point for the refactored gearbox diagnosis pipeline."""

import os
import warnings

import numpy as np

try:
    from .config import PipelineConfig
    from .dataset import process_hust_dataset
    from .evaluation import evaluate_groupkfold_by_run, evaluate_leave_one_condition_out
    from .plotting import plot_confusion_matrix
except ImportError:
    from config import PipelineConfig
    from dataset import process_hust_dataset
    from evaluation import evaluate_groupkfold_by_run, evaluate_leave_one_condition_out
    from plotting import plot_confusion_matrix

warnings.filterwarnings("ignore")
np.random.seed(42)


def main() -> None:
    config = PipelineConfig()
    data_path = config.data.data_path

    if not os.path.exists(data_path):
        print(f"Error: data path '{data_path}' does not exist.")
        print("Update config.py or PipelineConfig() in main.py so it points to the 'raw data' folder.")
        raise SystemExit(1)

    print(f"Data path found: {data_path}")

    x_df, y, groups, metadata = process_hust_dataset(
        data_config=config.data,
        window_config=config.window,
        feature_config=config.features,
    )

    results_speed, mean_imp_speed, cm_speed, label_encoder, diag_speed = evaluate_leave_one_condition_out(
        x_df,
        y,
        metadata,
        eval_config=config.loso_speed,
        model_config=config.model,
    )

    results_load, mean_imp_load, cm_load, label_encoder_load, diag_load = evaluate_leave_one_condition_out(
        x_df,
        y,
        metadata,
        eval_config=config.loso_load,
        model_config=config.model,
    )

    results_speed_load, mean_imp_speed_load, cm_speed_load, label_encoder_speed_load, diag_speed_load = evaluate_leave_one_condition_out(
        x_df,
        y,
        metadata,
        eval_config=config.loso_speed_load,
        model_config=config.model,
    )

    accs, cm_cv, label_encoder_cv = evaluate_groupkfold_by_run(
        x_df,
        y,
        groups,
        eval_config=config.group_kfold,
        model_config=config.model,
    )

    plot_confusion_matrix(
        cm=cm_speed,
        class_names=label_encoder.classes_,
        title="Aggregated Confusion Matrix (LOSO Speed)",
        output_path="hust_gearbox_loso_speed_cm.png",
        show=False,
    )

    if isinstance(diag_speed, dict) and ("cm_run_norm" in diag_speed):
        plot_confusion_matrix(
            cm=diag_speed["cm_run_norm"],
            class_names=label_encoder.classes_,
            title="Aggregated Confusion Matrix (LOSO Speed, Run-level Majority Vote)",
            output_path="hust_gearbox_loso_speed_cm_runlevel.png",
            show=False,
        )

    results_speed.to_csv("loso_speed_results.csv", index=False)
    if isinstance(diag_speed, dict) and ("run_votes_df" in diag_speed):
        diag_speed["run_votes_df"].to_csv("loso_speed_run_votes.csv", index=False)

    print("\nSaved: loso_speed_results.csv, hust_gearbox_loso_speed_cm.png")
    if os.path.exists("hust_gearbox_loso_speed_cm_runlevel.png"):
        print("Saved: hust_gearbox_loso_speed_cm_runlevel.png")
    if os.path.exists("loso_speed_run_votes.csv"):
        print("Saved: loso_speed_run_votes.csv")

    _ = (
        mean_imp_speed,
        mean_imp_load,
        mean_imp_speed_load,
        cm_load,
        cm_speed_load,
        label_encoder_load,
        label_encoder_speed_load,
        label_encoder_cv,
        diag_load,
        diag_speed_load,
        accs,
        cm_cv,
        results_load,
        results_speed_load,
    )


if __name__ == "__main__":
    main()
