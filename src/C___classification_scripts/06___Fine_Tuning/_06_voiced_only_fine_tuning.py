"""_06_voiced_only_fine_tuning.py
===============================

Copyright © 2026 Till Preidt (GitHub: MrNemesis98).
This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License
(CC BY-NC 4.0). Reuse and adaptation are permitted for non-commercial purposes,
provided appropriate credit is given."""

from __future__ import annotations

import platform
import shutil
import sys
from itertools import product
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support

SCRIPT_DIR = Path(__file__).resolve().parent
CLASSIFICATION_SCRIPTS_DIR = SCRIPT_DIR.parent
SRC_DIR = CLASSIFICATION_SCRIPTS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CLASSIFICATION_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_SCRIPTS_DIR))

import _00_config as config
from _00_classification_utils import (
    PLACE_FEATURE_COLUMNS,
    ensure_directory,
    evaluate_predictions,
    find_feature_table,
    fit_scaler,
    load_table,
    prepare_dataset,
    probability_frame,
    safe_print,
    save_confusion_matrix,
    save_evaluation_workbook,
    save_json,
    timestamp,
    transform_features,
)


# -----------------------------------------------------------------------------
# Fine-tuning configuration
# -----------------------------------------------------------------------------
# balance_strength = 0.00: empirical class distribution / no class weighting
# balance_strength = 1.00: equal priors / sklearn's fully balanced class weights
# Values between 0 and 1 interpolate between these two endpoints.
# Supported operation modes:
# TRAIN=True,  VALIDATE=True,  TEST=False -> train, validate, and copy bundles
# TRAIN=False, VALIDATE=True,  TEST=False -> load bundles and validate
# TRAIN=False, VALIDATE=False, TEST=True  -> load the locked bundle and test once
TRAIN = True
VALIDATE = True
TEST = False

# A separate recovery stage protects all Stage 01/02 outputs and locked bundles.
CONFIG_STAGE = "voiced_only_fine_tuning_stage_03"
MODEL_BUNDLE_COMPRESSION = 3
RESUME_COMPLETED_RUNS = True
MINIMUM_FREE_DISK_GB = 5.0
MODEL_STORAGE_ROOT = (
    PROJECT_ROOT / "models" / "06___Fine_Tuning_voiced_only" / CONFIG_STAGE
)

FINE_TUNING_CONFIG = {
    "lr": {
        "enabled": True,
        "frequency_hz": 2000,
        "balance_strengths": [0.75, 1.00],
        "parameters": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": "lbfgs",
            "max_iter": 2000,
            "tol": 0.0001,
            "random_state": 42,
        },
        "final_model_bundle": None,
        "final_model_bundle_directory": None,
    },
    "rf": {
        "enabled": True,
        "frequency_hz": 1000,
        "balance_strengths": [0.75, 1.00],
        "parameters": {
            "n_estimators": [300, 500],
            "max_depth": [10, 15, 30],
            "min_samples_leaf": [1, 5],
            "min_samples_split": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        "final_model_bundle": None,
        "final_model_bundle_directory": None,
    },
    "svm": {
        "enabled": False,
        "frequency_hz": 1000,
        "balance_strengths": [1.00],
        "parameters": {
            "C": [3.0, 10.0, 30.0],
            "kernel": "rbf",
            "gamma": [0.05, 0.1, 0.2],
        },
        # VALIDATE/TEST automatically load the single bundle from this folder.
        "final_model_bundle": None,
        "final_model_bundle_directory": (
            MODEL_STORAGE_ROOT
            / "svm_1000Hz_balance100_cfg007"
        ),
    },
    "qda": {
        "enabled": False,
        "frequency_hz": 2000,
        "balance_strengths": [0.75, 1.00],
        "parameter_grid": {
            "reg_param": [0.0, 0.01, 0.05, 0.1, 0.2],
            },
        "final_model_bundle": None,
        "final_model_bundle_directory": None,
    },
    "lda": {
        "enabled": True,
        "frequency_hz": 2000,
        "balance_strengths": [0.75, 1.00],
        "parameters": {
            "solver": "lsqr",
            "shrinkage": "auto",
            "tol": 0.0001,
        },
        "final_model_bundle": None,
        "final_model_bundle_directory": None,
    },
}

ANALYSIS_SUFFIX = "voiced_only_fine_tuning"
EXPERIMENT_NAME = "voiced_only_fine_tuning"
EXPECTED_PLACE_CLASSES = {"alveolo-palatal", "dental", "retroflex"}
MODEL_NAMES = {
    "lr": "logistic_regression",
    "svm": "support_vector_machine",
    "qda": "quadratic_discriminant_analysis",
    "lda": "linear_discriminant_analysis",
    "rf": "random_forest",
}


def validate_operation_mode() -> str:
    for name, value in (("TRAIN", TRAIN), ("VALIDATE", VALIDATE), ("TEST", TEST)):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be True or False.")
    if TEST:
        if TRAIN or VALIDATE:
            raise ValueError(
                "TEST=True is permitted only when TRAIN=False and VALIDATE=False."
            )
        return "test"
    if TRAIN and VALIDATE:
        return "training_validation"
    if not TRAIN and VALIDATE:
        return "validation"
    if not TRAIN and not VALIDATE:
        raise ValueError("No operation selected. Set TRAIN/VALIDATE or TEST to True.")
    raise ValueError(
        "TRAIN=True requires VALIDATE=True because fine-tuning is evaluated on "
        "the validation set. For validation with stored models, use TRAIN=False "
        "and VALIDATE=True."
    )


def validate_configuration() -> list[tuple[str, dict]]:
    enabled_models = []
    for model_key, model_config in FINE_TUNING_CONFIG.items():
        if model_key not in MODEL_NAMES:
            raise ValueError(f"Unsupported model in FINE_TUNING_CONFIG: {model_key}")
        if not model_config.get("enabled", False):
            continue

        frequency_hz = model_config.get("frequency_hz")
        if not isinstance(frequency_hz, int) or frequency_hz <= 0:
            raise ValueError(f"{model_key}: frequency_hz must be a positive integer.")

        strengths = model_config.get("balance_strengths", [])
        if not strengths:
            raise ValueError(f"{model_key}: balance_strengths must not be empty.")
        normalized_strengths = []
        for value in strengths:
            strength = float(value)
            if not 0.0 <= strength <= 1.0:
                raise ValueError(
                    f"{model_key}: balance strength {strength} is outside [0, 1]."
                )
            normalized_strengths.append(strength)
        if len(set(normalized_strengths)) != len(normalized_strengths):
            raise ValueError(f"{model_key}: balance_strengths contains duplicates.")

        model_config["balance_strengths"] = normalized_strengths
        expand_parameter_combinations(model_key, model_config)
        enabled_models.append((model_key, model_config))

    if not enabled_models:
        raise ValueError("No model is enabled in FINE_TUNING_CONFIG.")
    return enabled_models


def expand_parameter_combinations(model_key: str, model_config: dict) -> list[dict]:
    """Expand scalar or sequence-valued parameters into all combinations."""
    parameters = model_config.get("parameters", {})
    parameter_grid = model_config.get("parameter_grid", {})
    if not isinstance(parameters, dict) or not isinstance(parameter_grid, dict):
        raise ValueError(
            f"{model_key}: parameters and parameter_grid must be dictionaries."
        )

    duplicate_keys = sorted(set(parameters) & set(parameter_grid))
    if duplicate_keys:
        raise ValueError(
            f"{model_key}: parameters are defined twice: " + ", ".join(duplicate_keys)
        )

    combined = {**parameters, **parameter_grid}
    if not combined:
        return [{}]

    names = list(combined)
    value_lists = []
    for name in names:
        configured_value = combined[name]
        if isinstance(configured_value, (list, tuple)):
            values = list(configured_value)
            if not values:
                raise ValueError(f"{model_key}: parameter '{name}' has no values.")
        else:
            values = [configured_value]
        value_lists.append(values)

    combinations = [
        dict(zip(names, values)) for values in product(*value_lists)
    ]
    if len({repr(item) for item in combinations}) != len(combinations):
        raise ValueError(f"{model_key}: parameter grid produces duplicate combinations.")
    return combinations


def experiment_folder_name(dataset: str, frequency_hz: int) -> str:
    return f"{dataset}_{frequency_hz}"


def resolve_input(dataset: str, frequency_hz: int) -> Path:
    feature_dir = (
        config.FEATURE_EXTRACTION_OUTPUT_DIR
        / experiment_folder_name(dataset, frequency_hz)
    )
    return find_feature_table(feature_dir)


def fine_tuning_root(dataset: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "C___classification"
        / CONFIG_STAGE
        / dataset
    )


def stored_model_path(run_name: str, model_filename: str) -> Path:
    """Return the mirrored bundle path beneath the central models directory."""
    return MODEL_STORAGE_ROOT / run_name / model_filename


def ensure_sufficient_disk_space(path: Path) -> None:
    """Stop safely before Windows reaches critically low free disk space."""
    ensure_directory(path)
    free_gb = shutil.disk_usage(path).free / (1024**3)
    if free_gb < MINIMUM_FREE_DISK_GB:
        raise RuntimeError(
            f"Only {free_gb:.2f} GB are free on the output drive. "
            f"At least {MINIMUM_FREE_DISK_GB:.2f} GB are required to continue."
        )


def save_compressed_bundle(bundle: dict, destination: Path) -> None:
    """Write one compressed model bundle atomically to the models directory."""
    ensure_sufficient_disk_space(destination.parent)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(bundle, temporary_path, compress=MODEL_BUNDLE_COMPRESSION)
    temporary_path.replace(destination)


def retain_voiced_rows(data):
    mask = data.labels["voicing_label"].eq("voiced")
    retained_rows = int(mask.sum())
    if retained_rows == 0:
        raise ValueError("No voiced rows remain after filtering.")

    filtered = type(data)(
        place_features=data.place_features.loc[mask].reset_index(drop=True),
        voicing_features=data.voicing_features.loc[mask].reset_index(drop=True),
        labels=data.labels.loc[mask].reset_index(drop=True),
        identifiers=data.identifiers.loc[mask].reset_index(drop=True),
        original_rows=data.original_rows,
        retained_rows=retained_rows,
        removed_rows=data.original_rows - retained_rows,
    )
    present_classes = set(filtered.labels["place_label"].astype(str).unique())
    missing_classes = sorted(EXPECTED_PLACE_CLASSES - present_classes)
    if missing_classes:
        raise ValueError(
            "Voiced-only data are missing place classes: " + ", ".join(missing_classes)
        )
    return filtered


def build_place_predictions(identifiers, labels, predicted, probabilities=None):
    output = identifiers.reset_index(drop=True).copy()
    output["true_place_label"] = labels["place_label"].reset_index(drop=True)
    output["predicted_place_label"] = pd.Series(predicted, dtype="string")
    output["place_correct"] = (
        output["true_place_label"] == output["predicted_place_label"]
    )
    if probabilities is not None:
        output = pd.concat(
            [output, probabilities.add_prefix("p_place_").reset_index(drop=True)],
            axis=1,
        )
    return output


def empirical_priors(labels: pd.Series, classes: list[str]) -> dict[str, float]:
    proportions = labels.value_counts(normalize=True)
    return {class_name: float(proportions[class_name]) for class_name in classes}


def interpolated_priors(
    empirical: dict[str, float], classes: list[str], strength: float
) -> list[float]:
    uniform_prior = 1.0 / len(classes)
    priors = [
        (1.0 - strength) * empirical[class_name] + strength * uniform_prior
        for class_name in classes
    ]
    total = sum(priors)
    return [prior / total for prior in priors]


def interpolated_class_weights(
    labels: pd.Series, classes: list[str], strength: float
) -> tuple[dict[str, float], dict[str, float]]:
    counts = labels.value_counts()
    balanced_map = {
        class_name: len(labels) / (len(classes) * int(counts[class_name]))
        for class_name in classes
    }
    interpolated = {
        class_name: (1.0 - strength) + strength * balanced_map[class_name]
        for class_name in classes
    }
    return interpolated, balanced_map


def create_estimator(
    model_key: str,
    parameters: dict,
    labels: pd.Series,
    classes: list[str],
    strength: float,
):
    parameters = dict(parameters)
    empirical = empirical_priors(labels, classes)

    if model_key in {"lr", "rf", "svm"}:
        class_weights, fully_balanced = interpolated_class_weights(
            labels, classes, strength
        )
        if model_key == "lr":
            estimator = LogisticRegression(class_weight=class_weights, **parameters)
        elif model_key == "rf":
            estimator = RandomForestClassifier(class_weight=class_weights, **parameters)
        else:
            estimator = SVC(class_weight=class_weights, **parameters)
        balance_information = {
            "balance_method": "interpolated_class_weights",
            "empirical_class_priors": empirical,
            "fully_balanced_class_weights": fully_balanced,
            "applied_class_weights": class_weights,
        }
        return estimator, balance_information

    priors = interpolated_priors(empirical, classes, strength)
    if model_key == "qda":
        estimator = QuadraticDiscriminantAnalysis(priors=priors, **parameters)
    elif model_key == "lda":
        solver = parameters.get("solver", "svd")
        shrinkage = parameters.get("shrinkage")
        if solver == "svd" and shrinkage is not None:
            raise ValueError(
                "LDA shrinkage cannot be used with solver='svd'. "
                "Use solver='lsqr' or solver='eigen'."
            )
        estimator = LinearDiscriminantAnalysis(priors=priors, **parameters)
    else:
        raise ValueError(f"Unsupported model: {model_key}")

    balance_information = {
        "balance_method": "interpolated_class_priors",
        "empirical_class_priors": empirical,
        "uniform_class_priors": {
            class_name: 1.0 / len(classes) for class_name in classes
        },
        "applied_class_priors": dict(zip(classes, priors)),
    }
    return estimator, balance_information


def strength_code(strength: float) -> str:
    return f"{round(strength * 100):03d}"


def append_class_metrics(
    summary_row: dict, true_labels: pd.Series, predicted_labels
) -> None:
    classes = sorted(EXPECTED_PLACE_CLASSES)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=classes,
        zero_division=0,
    )
    for index, class_name in enumerate(classes):
        key = class_name.replace("-", "_")
        summary_row[f"{key}_precision"] = float(precision[index])
        summary_row[f"{key}_recall"] = float(recall[index])
        summary_row[f"{key}_f1"] = float(f1[index])
        summary_row[f"{key}_support"] = int(support[index])


def summary_paths(output_root: Path) -> tuple[Path, Path]:
    return (
        output_root / "voiced_only_fine_tuning_summary.xlsx",
        output_root / "voiced_only_fine_tuning_summary.csv",
    )


def write_summary_checkpoint(summary_rows: list[dict], output_root: Path) -> None:
    """Atomically preserve every completed configuration in the stage summary."""
    if not summary_rows:
        return
    summary = pd.DataFrame(summary_rows).sort_values(
        ["balanced_accuracy", "f1_macro"], ascending=[False, False]
    )
    summary_path, summary_csv_path = summary_paths(output_root)
    temporary_xlsx = summary_path.with_name(summary_path.stem + ".tmp.xlsx")
    temporary_csv = summary_csv_path.with_name(summary_csv_path.stem + ".tmp.csv")
    summary.to_excel(temporary_xlsx, index=False)
    temporary_xlsx.replace(summary_path)
    summary.to_csv(temporary_csv, index=False, encoding="utf-8-sig")
    temporary_csv.replace(summary_csv_path)


def load_summary_checkpoint(output_root: Path) -> list[dict]:
    if not RESUME_COMPLETED_RUNS:
        return []
    _, summary_csv_path = summary_paths(output_root)
    if not summary_csv_path.is_file():
        return []
    return pd.read_csv(summary_csv_path).to_dict(orient="records")


def completed_run_names(summary_rows: list[dict]) -> set[str]:
    completed = set()
    for row in summary_rows:
        run_name = row.get("run_name")
        bundle_path = row.get("stored_model_bundle")
        if isinstance(run_name, str) and isinstance(bundle_path, str) and Path(bundle_path).is_file():
            completed.add(run_name)
    return completed


def run_training_validation() -> int:
    dataset = "validation"
    enabled_models = validate_configuration()
    config.create_core_output_directories()
    output_root = ensure_directory(fine_tuning_root(dataset))
    ensure_sufficient_disk_space(output_root)
    data_cache = {}
    summary_rows = load_summary_checkpoint(output_root)
    completed_runs = completed_run_names(summary_rows)
    experiment_run_id = timestamp()

    safe_print("\n==================================================")
    safe_print("Voiced-Only Fine-Tuning")
    safe_print("==================================================")
    safe_print(f"Evaluation set : {dataset}")
    safe_print(f"Output root    : {output_root}")
    safe_print("Row filter     : voicing_label == voiced")
    safe_print(f"Place features : {', '.join(PLACE_FEATURE_COLUMNS)}")
    expected_runs = sum(
        len(model_config["balance_strengths"])
        * len(expand_parameter_combinations(model_key, model_config))
        for model_key, model_config in enabled_models
    )
    safe_print(f"Configured runs: {expected_runs}")
    if completed_runs:
        safe_print(f"Resuming runs  : {len(completed_runs)} already completed")

    for model_key, model_config in enabled_models:
        frequency_hz = model_config["frequency_hz"]
        if frequency_hz not in data_cache:
            training_path = resolve_input("training", frequency_hz)
            evaluation_path = resolve_input(dataset, frequency_hz)
            training_all = prepare_dataset(load_table(training_path))
            evaluation_all = prepare_dataset(load_table(evaluation_path))
            training = retain_voiced_rows(training_all)
            evaluation = retain_voiced_rows(evaluation_all)
            scaler = fit_scaler(training.place_features)
            data_cache[frequency_hz] = {
                "training_path": training_path,
                "evaluation_path": evaluation_path,
                "training_all": training_all,
                "evaluation_all": evaluation_all,
                "training": training,
                "evaluation": evaluation,
                "scaler": scaler,
                "x_train": transform_features(scaler, training.place_features),
                "x_eval": transform_features(scaler, evaluation.place_features),
            }

        data = data_cache[frequency_hz]
        training = data["training"]
        evaluation = data["evaluation"]
        labels_train = training.labels["place_label"]
        classes = sorted(labels_train.astype(str).unique())
        parameter_combinations = expand_parameter_combinations(
            model_key, model_config
        )

        for strength in model_config["balance_strengths"]:
            code = strength_code(strength)
            for config_index, model_parameters in enumerate(
                parameter_combinations, start=1
            ):
                config_code = f"cfg{config_index:03d}"
                run_name = (
                    f"{model_key}_{frequency_hz}Hz_balance{code}_{config_code}"
                )
                if run_name in completed_runs:
                    safe_print(f"Skipping completed run: {run_name}")
                    continue
                ensure_sufficient_disk_space(output_root)
                run_dir = ensure_directory(output_root / run_name)
                place_model, balance_information = create_estimator(
                    model_key,
                    model_parameters,
                    labels_train,
                    classes,
                    strength,
                )

                parameter_text = ", ".join(
                    f"{name}={value!r}" for name, value in model_parameters.items()
                )
                safe_print(
                    f"\nTraining {model_key.upper()} at {frequency_hz} Hz "
                    f"with balance strength {strength:.2f}, {config_code} "
                    f"({parameter_text or 'default parameters'})..."
                )
                place_model.fit(data["x_train"], labels_train)
                predicted_place = place_model.predict(data["x_eval"])
                result = evaluate_predictions(
                    evaluation.labels["place_label"],
                    predicted_place,
                    target_name="place",
                    label_order=list(place_model.classes_),
                )
                predictions = build_place_predictions(
                    evaluation.identifiers,
                    evaluation.labels,
                    predicted_place,
                    probabilities=probability_frame(place_model, data["x_eval"]),
                )

                run_id = timestamp()
                model_name = MODEL_NAMES[model_key]
                prefix = (
                    f"{dataset}_{model_name}_{frequency_hz}Hz_balance{code}_"
                    f"{config_code}_{ANALYSIS_SUFFIX}_{run_id}"
                )
                predictions_path = run_dir / f"{prefix}_predictions.csv"
                metrics_path = run_dir / f"{prefix}_metrics.xlsx"
                info_path = run_dir / f"{prefix}_run_information.json"
                predictions.to_csv(
                    predictions_path, index=False, encoding="utf-8-sig"
                )

                run_information = {
                    "model": model_name,
                    "model_key": model_key,
                    "experiment": EXPERIMENT_NAME,
                    "analysis_scope": ANALYSIS_SUFFIX,
                    "row_filter": "voicing_label == voiced",
                    "prediction_target": "place_label",
                    "place_bandpass_low_hz": frequency_hz,
                    "balance_strength": strength,
                    **balance_information,
                    "parameter_configuration_id": config_code,
                    "configured_model_parameters": model_parameters,
                    "configured_parameter_space": {
                        "parameters": model_config.get("parameters", {}),
                        "parameter_grid": model_config.get("parameter_grid", {}),
                    },
                    "fitted_model_parameters": place_model.get_params(deep=False),
                    "training_experiment_folder": experiment_folder_name(
                        "training", frequency_hz
                    ),
                    "evaluation_experiment_folder": experiment_folder_name(
                        dataset, frequency_hz
                    ),
                    "output_experiment_folder": str(run_dir),
                    "training_subset": "training",
                    "evaluation_subset": dataset,
                    "training_input": str(data["training_path"]),
                    "evaluation_input": str(data["evaluation_path"]),
                    "place_feature_columns": list(PLACE_FEATURE_COLUMNS),
                    "feature_scaling": (
                        "StandardScaler fitted on voiced training place features"
                    ),
                    "training_original_rows": data["training_all"].original_rows,
                    "training_valid_rows_before_filter": data[
                        "training_all"
                    ].retained_rows,
                    "training_voiced_rows": training.retained_rows,
                    "training_removed_invalid_or_voiceless_rows": training.removed_rows,
                    "evaluation_original_rows": data["evaluation_all"].original_rows,
                    "evaluation_valid_rows_before_filter": data[
                        "evaluation_all"
                    ].retained_rows,
                    "evaluation_voiced_rows": evaluation.retained_rows,
                    "evaluation_removed_invalid_or_voiceless_rows": evaluation.removed_rows,
                    "python_version": platform.python_version(),
                    "pandas_version": pd.__version__,
                    "scikit_learn_version": sklearn.__version__,
                    "experiment_run_id": experiment_run_id,
                    "run_id": run_id,
                }

                save_evaluation_workbook(
                    [result], metrics_path, run_information=run_information
                )
                save_json(run_information, info_path)
                save_confusion_matrix(
                    result, run_dir / f"{prefix}_place_confusion.png"
                )
                save_confusion_matrix(
                    result,
                    run_dir / f"{prefix}_place_confusion_normalized.png",
                    normalize=True,
                )
                model_filename = f"{model_name}_{run_name}_model.joblib"
                central_model_path = stored_model_path(run_name, model_filename)
                ensure_directory(central_model_path.parent)
                save_compressed_bundle(
                    {
                        "model_name": model_name,
                        "model_key": model_key,
                        "experiment": EXPERIMENT_NAME,
                        "analysis_scope": ANALYSIS_SUFFIX,
                        "row_filter": "voicing_label == voiced",
                        "prediction_target": "place_label",
                        "place_bandpass_low_hz": frequency_hz,
                        "balance_strength": strength,
                        "balance_information": balance_information,
                        "place_feature_columns": list(PLACE_FEATURE_COLUMNS),
                        "place_scaler": data["scaler"],
                        "place_model": place_model,
                        "run_information": run_information,
                    },
                    central_model_path,
                )

                summary_row = {
                    "model": model_name,
                    "model_key": model_key,
                    "frequency_hz": frequency_hz,
                    "balance_strength": strength,
                    "parameter_configuration_id": config_code,
                    "run_name": run_name,
                    **result.summary,
                    "run_directory": str(run_dir),
                    "stored_model_bundle": str(central_model_path),
                    "run_id": run_id,
                }
                for parameter_name, parameter_value in model_parameters.items():
                    summary_row[f"parameter_{parameter_name}"] = parameter_value
                append_class_metrics(
                    summary_row,
                    evaluation.labels["place_label"],
                    predicted_place,
                )
                summary_rows.append(summary_row)
                completed_runs.add(run_name)
                write_summary_checkpoint(summary_rows, output_root)
                safe_print(
                    f"accuracy={result.summary['accuracy']:.4f} | "
                    f"balanced_accuracy={result.summary['balanced_accuracy']:.4f} | "
                    f"macro_F1={result.summary['f1_macro']:.4f}"
                )
                model_size_mb = central_model_path.stat().st_size / (1024**2)
                safe_print(f"Compressed model: {central_model_path} ({model_size_mb:.1f} MB)")

    write_summary_checkpoint(summary_rows, output_root)
    summary_path, summary_csv_path = summary_paths(output_root)

    safe_print("\n==================================================")
    safe_print("Fine-tuning completed")
    safe_print("==================================================")
    safe_print(f"Completed runs : {len(summary_rows)}")
    safe_print(f"Summary workbook: {summary_path}")
    safe_print(f"Summary CSV     : {summary_csv_path}")
    return 0


def resolve_model_bundle(model_key: str, model_config: dict) -> Path:
    configured_path = model_config.get("final_model_bundle")
    if configured_path not in (None, ""):
        bundle_path = Path(configured_path).expanduser()
        if not bundle_path.is_absolute():
            bundle_path = (PROJECT_ROOT / bundle_path).resolve()
        if not bundle_path.is_file():
            raise FileNotFoundError(
                f"{model_key}: final model bundle not found: {bundle_path}"
            )
        return bundle_path

    configured_directory = model_config.get("final_model_bundle_directory")
    if configured_directory in (None, ""):
        raise ValueError(
            f"{model_key}: stored-model evaluation requires either "
            "'final_model_bundle' or "
            "'final_model_bundle_directory'."
        )
    bundle_directory = Path(configured_directory).expanduser()
    if not bundle_directory.is_absolute():
        bundle_directory = (PROJECT_ROOT / bundle_directory).resolve()
    if not bundle_directory.is_dir():
        raise FileNotFoundError(
            f"{model_key}: final model directory not found: {bundle_directory}"
        )

    candidates = sorted(bundle_directory.glob("*_model.joblib"))
    if not candidates:
        raise FileNotFoundError(
            f"{model_key}: no *_model.joblib file found in {bundle_directory}"
        )
    if len(candidates) > 1:
        candidate_names = "\n".join(f"  - {path.name}" for path in candidates)
        raise ValueError(
            f"{model_key}: multiple model bundles found in {bundle_directory}. "
            "Set 'final_model_bundle' to the intended file:\n" + candidate_names
        )
    return candidates[0]


def load_checked_bundle(model_key: str, model_config: dict):
    bundle_path = resolve_model_bundle(model_key, model_config)
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict):
        raise TypeError(f"{model_key}: model bundle must contain a dictionary.")

    required_keys = {
        "place_model",
        "place_scaler",
        "place_feature_columns",
        "place_bandpass_low_hz",
    }
    missing_keys = sorted(required_keys - set(bundle))
    if missing_keys:
        raise ValueError(
            f"{model_key}: model bundle is missing: " + ", ".join(missing_keys)
        )

    expected_model_name = MODEL_NAMES[model_key]
    bundled_model_name = bundle.get("model_name")
    if bundled_model_name and bundled_model_name != expected_model_name:
        raise ValueError(
            f"{model_key}: bundle contains model '{bundled_model_name}', "
            f"expected '{expected_model_name}'."
        )

    bundled_features = list(bundle["place_feature_columns"])
    if bundled_features != list(PLACE_FEATURE_COLUMNS):
        raise ValueError(
            f"{model_key}: bundle feature columns do not match the current pipeline."
        )

    bundled_frequency = int(bundle["place_bandpass_low_hz"])
    configured_frequency = int(model_config["frequency_hz"])
    if bundled_frequency != configured_frequency:
        raise ValueError(
            f"{model_key}: bundle uses {bundled_frequency} Hz, but the "
            f"configuration specifies {configured_frequency} Hz."
        )
    return bundle_path, bundle


def run_saved_model_evaluation(dataset: str) -> int:
    if dataset not in {"validation", "test"}:
        raise ValueError(f"Unsupported evaluation dataset: {dataset}")
    enabled_models = validate_configuration()
    config.create_core_output_directories()
    output_root = ensure_directory(fine_tuning_root(dataset))
    summary_rows = []
    experiment_run_id = timestamp()

    safe_print("\n==================================================")
    title = (
        "Voiced-Only Final Test Evaluation"
        if dataset == "test"
        else "Voiced-Only Stored-Model Validation"
    )
    safe_print(title)
    safe_print("==================================================")
    safe_print("Training models : disabled")
    safe_print(
        f"Validation      : {'enabled' if dataset == 'validation' else 'disabled'}"
    )
    safe_print(f"Test evaluation : {'enabled' if dataset == 'test' else 'disabled'}")
    safe_print(f"Output root     : {output_root}")

    for model_key, model_config in enabled_models:
        bundle_path, bundle = load_checked_bundle(model_key, model_config)
        model_name = MODEL_NAMES[model_key]
        frequency_hz = int(bundle["place_bandpass_low_hz"])
        place_model = bundle["place_model"]
        place_scaler = bundle["place_scaler"]
        balance_strength = float(bundle.get("balance_strength", 1.0))
        code = strength_code(balance_strength)

        evaluation_path = resolve_input(dataset, frequency_hz)
        evaluation_all = prepare_dataset(load_table(evaluation_path))
        evaluation = retain_voiced_rows(evaluation_all)
        x_place_eval = transform_features(
            place_scaler, evaluation.place_features
        )

        safe_print(
            f"\nEvaluating {model_key.upper()} at {frequency_hz} Hz on {dataset} using "
            f"the locked bundle:\n{bundle_path}"
        )
        predicted_place = place_model.predict(x_place_eval)
        result = evaluate_predictions(
            evaluation.labels["place_label"],
            predicted_place,
            target_name="place",
            label_order=list(place_model.classes_),
        )
        predictions = build_place_predictions(
            evaluation.identifiers,
            evaluation.labels,
            predicted_place,
            probabilities=probability_frame(place_model, x_place_eval),
        )

        run_id = timestamp()
        run_name = f"{model_key}_{frequency_hz}Hz_balance{code}_final"
        run_dir = ensure_directory(output_root / run_name)
        prefix = (
            f"{dataset}_{model_name}_{frequency_hz}Hz_balance{code}_final_"
            f"{ANALYSIS_SUFFIX}_{run_id}"
        )
        predictions_path = run_dir / f"{prefix}_predictions.csv"
        metrics_path = run_dir / f"{prefix}_metrics.xlsx"
        info_path = run_dir / f"{prefix}_run_information.json"
        predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

        source_run_information = bundle.get("run_information", {})
        run_information = {
            "model": model_name,
            "model_key": model_key,
            "experiment": EXPERIMENT_NAME,
            "analysis_scope": ANALYSIS_SUFFIX,
            "operation_mode": f"stored_model_{dataset}_evaluation",
            "train_enabled": TRAIN,
            "validation_enabled": VALIDATE,
            "test_enabled": TEST,
            "source_model_bundle": str(bundle_path),
            "source_model_run_id": source_run_information.get("run_id"),
            "row_filter": "voicing_label == voiced",
            "prediction_target": "place_label",
            "place_bandpass_low_hz": frequency_hz,
            "balance_strength": balance_strength,
            "fitted_model_parameters": place_model.get_params(deep=False),
            "evaluation_experiment_folder": experiment_folder_name(
                dataset, frequency_hz
            ),
            "output_experiment_folder": str(run_dir),
            "evaluation_subset": dataset,
            "evaluation_input": str(evaluation_path),
            "place_feature_columns": list(PLACE_FEATURE_COLUMNS),
            "feature_scaling": "Loaded unchanged from the locked model bundle",
            "evaluation_original_rows": evaluation_all.original_rows,
            "evaluation_valid_rows_before_filter": evaluation_all.retained_rows,
            "evaluation_voiced_rows": evaluation.retained_rows,
            "evaluation_removed_invalid_or_voiceless_rows": evaluation.removed_rows,
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "experiment_run_id": experiment_run_id,
            "run_id": run_id,
        }
        save_evaluation_workbook(
            [result], metrics_path, run_information=run_information
        )
        save_json(run_information, info_path)
        save_confusion_matrix(result, run_dir / f"{prefix}_place_confusion.png")
        save_confusion_matrix(
            result,
            run_dir / f"{prefix}_place_confusion_normalized.png",
            normalize=True,
        )

        summary_row = {
            "model": model_name,
            "model_key": model_key,
            "frequency_hz": frequency_hz,
            "balance_strength": balance_strength,
            **result.summary,
            "source_model_bundle": str(bundle_path),
            "run_directory": str(run_dir),
            "run_id": run_id,
        }
        for parameter_name, parameter_value in place_model.get_params(
            deep=False
        ).items():
            if isinstance(parameter_value, (str, int, float, bool, type(None))):
                summary_row[f"parameter_{parameter_name}"] = parameter_value
        append_class_metrics(
            summary_row,
            evaluation.labels["place_label"],
            predicted_place,
        )
        summary_rows.append(summary_row)
        safe_print(
            f"accuracy={result.summary['accuracy']:.4f} | "
            f"balanced_accuracy={result.summary['balanced_accuracy']:.4f} | "
            f"macro_F1={result.summary['f1_macro']:.4f}"
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["balanced_accuracy", "f1_macro"], ascending=[False, False]
    )
    summary_path = output_root / f"voiced_only_{dataset}_summary.xlsx"
    summary.to_excel(summary_path, index=False)
    summary_csv_path = output_root / f"voiced_only_{dataset}_summary.csv"
    summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    safe_print(f"\n{dataset.title()} summary workbook: {summary_path}")
    safe_print(f"{dataset.title()} summary CSV     : {summary_csv_path}")
    return 0


def main() -> int:
    operation_mode = validate_operation_mode()
    if operation_mode == "training_validation":
        return run_training_validation()
    return run_saved_model_evaluation(operation_mode)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        safe_print("\nInterrupted by user.")
        raise SystemExit(130)
    except Exception as error:
        safe_print(f"\nERROR: {type(error).__name__}: {error}")
        raise
