"""Evaluation routines for gearbox classification experiments."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

try:
    from .config import GroupKFoldConfig, LeaveOneConditionOutConfig, ModelConfig
    from .models import build_random_forest_pipeline
except ImportError:
    from config import GroupKFoldConfig, LeaveOneConditionOutConfig, ModelConfig
    from models import build_random_forest_pipeline


def _majority_vote_int(labels_1d: np.ndarray) -> int:
    labels_1d = np.asarray(labels_1d, dtype=int)
    if labels_1d.size == 0:
        return 0
    vals, counts = np.unique(labels_1d, return_counts=True)
    max_count = counts.max()
    return int(vals[counts == max_count].min())


def _run_level_votes_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_ids: np.ndarray,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    run_ids = np.asarray(run_ids).astype(str)

    if not (len(y_true) == len(y_pred) == len(run_ids)):
        raise ValueError("Length mismatch in run-level voting inputs.")

    df = pd.DataFrame({"run": run_ids, "true": y_true, "pred": y_pred})
    nunique = df.groupby("run")["true"].nunique()
    if (nunique > 1).any():
        bad_runs = nunique[nunique > 1].index.tolist()[:5]
        raise ValueError(f"Inconsistent true labels in runs: {bad_runs}")

    return (
        df.groupby("run", sort=False)
        .agg(
            true=("true", "first"),
            pred=("pred", lambda s: _majority_vote_int(s.values)),
            n_windows=("pred", "size"),
        )
        .reset_index()
    )


def _permute_labels_by_run(
    y_train: np.ndarray,
    run_ids_train: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=int)
    run_ids_train = np.asarray(run_ids_train).astype(str)
    df = pd.DataFrame({"run": run_ids_train, "y": y_train})

    nunique = df.groupby("run")["y"].nunique()
    if (nunique > 1).any():
        bad_runs = nunique[nunique > 1].index.tolist()[:5]
        raise ValueError(f"Inconsistent training labels in runs: {bad_runs}")

    run_to_label = df.groupby("run", sort=False)["y"].first()
    permuted_labels = rng.permutation(run_to_label.values)
    mapping = dict(zip(run_to_label.index.tolist(), permuted_labels))
    return np.asarray([mapping[run] for run in run_ids_train], dtype=int)


def evaluate_leave_one_condition_out(
    x_df: pd.DataFrame,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    eval_config: LeaveOneConditionOutConfig,
    model_config: ModelConfig,
) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray, LabelEncoder, Dict]:
    rng = np.random.RandomState(int(eval_config.rng_seed))
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    if eval_config.exclude_time_varying:
        keep_mask = ~metadata_df["is_time_varying"].astype(bool).values
        x_use = x_df.loc[keep_mask].reset_index(drop=True)
        y_use = y_enc[keep_mask]
        meta_use = metadata_df.loc[keep_mask].reset_index(drop=True)
    else:
        x_use = x_df.reset_index(drop=True)
        y_use = y_enc
        meta_use = metadata_df.reset_index(drop=True)

    if eval_config.condition == "speed":
        cond_series = meta_use["speed_str"].astype(str)
    elif eval_config.condition == "load":
        cond_series = meta_use["load_index"].astype(int).astype(str)
    elif eval_config.condition == "speed_load":
        cond_series = meta_use["speed_str"].astype(str) + "__" + meta_use["load_index"].astype(int).astype(str)
    else:
        raise ValueError("condition must be 'speed', 'load', or 'speed_load'")

    unique_conditions = sorted(cond_series.unique().tolist())

    if eval_config.verbose:
        print("\n" + "=" * 60)
        print(f"LEAVE-ONE-{eval_config.condition.upper()}-OUT EVALUATION")
        print("=" * 60)
        print(f"Conditions ({len(unique_conditions)}): {unique_conditions}")
        if eval_config.permutation_test:
            print(
                f"Permutation check: mode={eval_config.permutation_mode}, "
                f"repeats={eval_config.permutation_repeats}"
            )

    window_accs: List[float] = []
    run_accs: List[float] = []
    perm_accs_window: List[float] = []
    perm_accs_run: List[float] = []
    rows: List[Dict] = []
    all_true: List[int] = []
    all_pred: List[int] = []
    all_run_ids: List[str] = []

    importance_sum = {column: 0.0 for column in x_use.columns}
    importance_count = 0
    base_pipe = build_random_forest_pipeline(n_features=x_use.shape[1], model_config=model_config)

    for cond in unique_conditions:
        test_mask = cond_series.values == cond
        train_mask = ~test_mask

        x_train = x_use.loc[train_mask]
        y_train = y_use[train_mask]
        x_test = x_use.loc[test_mask]
        y_test = y_use[test_mask]

        pipe = clone(base_pipe)
        pipe.fit(x_train, y_train)
        y_hat = pipe.predict(x_test)
        acc = float(accuracy_score(y_test, y_hat))
        window_accs.append(acc)

        row = {
            "held_out_condition": cond,
            "n_train_windows": int(len(x_train)),
            "n_test_windows": int(len(x_test)),
            "window_accuracy": acc,
        }

        run_acc = np.nan
        n_test_runs = np.nan
        run_ids_test = meta_use.loc[test_mask, "filename"].astype(str).values

        if eval_config.compute_run_level:
            run_df_split = _run_level_votes_df(y_test, y_hat, run_ids_test)
            run_acc = float(accuracy_score(run_df_split["true"], run_df_split["pred"]))
            n_test_runs = int(run_df_split.shape[0])
            run_accs.append(run_acc)
            row["n_test_runs"] = n_test_runs
            row["run_accuracy"] = run_acc
            all_run_ids.extend(run_ids_test.tolist())

        all_true.extend(y_test.tolist())
        all_pred.extend(y_hat.tolist())

        if eval_config.permutation_test:
            split_perm_accs_window: List[float] = []
            split_perm_accs_run: List[float] = []
            run_ids_train = meta_use.loc[train_mask, "filename"].astype(str).values

            for _ in range(int(eval_config.permutation_repeats)):
                if eval_config.permutation_mode == "run":
                    y_train_perm = _permute_labels_by_run(y_train, run_ids_train, rng)
                elif eval_config.permutation_mode == "window":
                    y_train_perm = rng.permutation(y_train)
                else:
                    raise ValueError("permutation_mode must be 'run' or 'window'")

                pipe_perm = clone(base_pipe)
                pipe_perm.fit(x_train, y_train_perm)
                y_hat_perm = pipe_perm.predict(x_test)
                split_perm_accs_window.append(float(accuracy_score(y_test, y_hat_perm)))

                if eval_config.compute_run_level:
                    run_df_perm = _run_level_votes_df(y_test, y_hat_perm, run_ids_test)
                    split_perm_accs_run.append(float(accuracy_score(run_df_perm["true"], run_df_perm["pred"])))

            perm_acc_w = float(np.mean(split_perm_accs_window))
            perm_accs_window.append(perm_acc_w)
            row["perm_accuracy_window"] = perm_acc_w

            if eval_config.compute_run_level:
                perm_acc_r = float(np.mean(split_perm_accs_run))
                perm_accs_run.append(perm_acc_r)
                row["perm_accuracy_run"] = perm_acc_r

        selector = pipe.named_steps["selector"]
        model = pipe.named_steps["model"]
        selected_names = x_use.columns[selector.get_support()].tolist()
        importances = model.feature_importances_.tolist()
        for name, value in zip(selected_names, importances):
            importance_sum[name] += float(value)
        importance_count += 1

        rows.append(row)

        if eval_config.verbose:
            message = (
                f"  Held out {eval_config.condition}={cond:>8s}  |  "
                f"window_acc={acc:.4f}  |  n_test_windows={len(x_test)}"
            )
            if eval_config.compute_run_level:
                message += f"  |  run_acc={run_acc:.4f}  |  n_test_runs={n_test_runs}"
            if eval_config.permutation_test:
                message += f"  |  perm_acc_w={perm_acc_w:.4f}"
                if eval_config.compute_run_level:
                    message += f"  |  perm_acc_r={perm_acc_r:.4f}"
            print(message)

    results_df = pd.DataFrame(rows).sort_values("held_out_condition").reset_index(drop=True)
    mean_acc = float(np.mean(window_accs)) if window_accs else 0.0
    std_acc = float(np.std(window_accs)) if window_accs else 0.0

    if eval_config.verbose:
        print("\nWindow-level summary:")
        print(f"  Mean accuracy: {mean_acc:.4f}")
        print(f"  Std  accuracy: {std_acc:.4f}")
        print("\nAggregated WINDOW-level classification report (all held-out splits combined):")
        print(classification_report(all_true, all_pred, target_names=label_encoder.classes_))

    cm_window = confusion_matrix(all_true, all_pred)
    cm_window_norm = cm_window.astype(float) / (cm_window.sum(axis=1, keepdims=True) + 1e-12)
    mean_importance = {name: value / max(1, importance_count) for name, value in importance_sum.items()}

    diagnostics: Dict = {
        "window_mean_accuracy": mean_acc,
        "window_std_accuracy": std_acc,
    }

    if eval_config.compute_run_level:
        run_votes_all = _run_level_votes_df(
            np.asarray(all_true, dtype=int),
            np.asarray(all_pred, dtype=int),
            np.asarray(all_run_ids, dtype=str),
        )
        run_acc_all = float(accuracy_score(run_votes_all["true"], run_votes_all["pred"]))
        cm_run = confusion_matrix(run_votes_all["true"], run_votes_all["pred"])
        cm_run_norm = cm_run.astype(float) / (cm_run.sum(axis=1, keepdims=True) + 1e-12)

        if eval_config.verbose:
            mean_run = float(np.mean(run_accs)) if run_accs else 0.0
            std_run = float(np.std(run_accs)) if run_accs else 0.0
            print("\nRun-level summary (majority vote per file/run):")
            print(f"  Mean run accuracy across held-out conditions: {mean_run:.4f}")
            print(f"  Std  run accuracy across held-out conditions: {std_run:.4f}")
            print(f"  Aggregated run accuracy (all runs combined): {run_acc_all:.4f}")
            print("\nAggregated RUN-level classification report:")
            print(
                classification_report(
                    run_votes_all["true"],
                    run_votes_all["pred"],
                    target_names=label_encoder.classes_,
                )
            )

        diagnostics.update(
            {
                "cm_run_norm": cm_run_norm,
                "run_accuracy": run_acc_all,
                "run_votes_df": run_votes_all,
                "run_mean_accuracy": float(np.mean(run_accs)) if run_accs else np.nan,
                "run_std_accuracy": float(np.std(run_accs)) if run_accs else np.nan,
            }
        )

    if eval_config.permutation_test:
        perm_mean_w = float(np.mean(perm_accs_window)) if perm_accs_window else np.nan
        perm_std_w = float(np.std(perm_accs_window)) if perm_accs_window else np.nan
        if eval_config.verbose:
            print("\nPermutation sanity (window-level):")
            print(f"  Mean perm accuracy: {perm_mean_w:.4f}")
            print(f"  Std  perm accuracy: {perm_std_w:.4f}")
        diagnostics.update(
            {
                "perm_accs_window": perm_accs_window,
                "perm_mean_window": perm_mean_w,
                "perm_std_window": perm_std_w,
            }
        )

        if eval_config.compute_run_level:
            perm_mean_r = float(np.mean(perm_accs_run)) if perm_accs_run else np.nan
            perm_std_r = float(np.std(perm_accs_run)) if perm_accs_run else np.nan
            if eval_config.verbose:
                print("\nPermutation sanity (run-level):")
                print(f"  Mean perm run accuracy: {perm_mean_r:.4f}")
                print(f"  Std  perm run accuracy: {perm_std_r:.4f}")
            diagnostics.update(
                {
                    "perm_accs_run": perm_accs_run,
                    "perm_mean_run": perm_mean_r,
                    "perm_std_run": perm_std_r,
                    "perm_mode": eval_config.permutation_mode,
                    "perm_repeats": int(eval_config.permutation_repeats),
                }
            )

    return results_df, mean_importance, cm_window_norm, label_encoder, diagnostics


def evaluate_groupkfold_by_run(
    x_df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    eval_config: GroupKFoldConfig,
    model_config: ModelConfig,
) -> Tuple[List[float], np.ndarray, LabelEncoder]:
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    cv = GroupKFold(n_splits=eval_config.n_splits)
    base_pipe = build_random_forest_pipeline(n_features=x_df.shape[1], model_config=model_config)

    accs: List[float] = []
    all_true: List[int] = []
    all_pred: List[int] = []

    if eval_config.verbose:
        print("\n" + "=" * 60)
        print("GROUPKFOLD BY RUN EVALUATION")
        print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(cv.split(x_df, y_enc, groups=groups), start=1):
        pipe = clone(base_pipe)
        pipe.fit(x_df.iloc[train_idx], y_enc[train_idx])
        y_hat = pipe.predict(x_df.iloc[test_idx])
        acc = float(accuracy_score(y_enc[test_idx], y_hat))
        accs.append(acc)
        all_true.extend(y_enc[test_idx].tolist())
        all_pred.extend(y_hat.tolist())

        if eval_config.verbose:
            print(
                f"  Fold {fold:>2d}: acc={acc:.4f}  |  "
                f"n_test={len(test_idx)}  |  n_groups_test={len(np.unique(groups[test_idx]))}"
            )

    cm = confusion_matrix(all_true, all_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    if eval_config.verbose:
        print("\nSummary:")
        print(f"  Mean accuracy: {float(np.mean(accs)):.4f}")
        print(f"  Std  accuracy: {float(np.std(accs)):.4f}")

    return accs, cm_norm, label_encoder
