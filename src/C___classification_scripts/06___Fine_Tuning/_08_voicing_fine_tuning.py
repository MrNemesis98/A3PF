"""_08_voicing_fine_tuning.py
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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC

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
    VOICING_FEATURE_COLUMNS,
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
# TRAIN=True,  VALIDATE=True,  TEST=False -> train, validate, and store bundles
# TRAIN=False, VALIDATE=True,  TEST=False -> load bundles and validate
# TRAIN=False, VALIDATE=False, TEST=True  -> load the locked bundle and test once
TRAIN = False
VALIDATE = False
TEST = True

CONFIG_STAGE = "voicing_fine_tuning_stage_02"
SOURCE_FREQUENCY_HZ = 2000
MODEL_BUNDLE_COMPRESSION = 3
RESUME_COMPLETED_RUNS = True
MINIMUM_FREE_DISK_GB = 5.0
MODEL_STORAGE_ROOT = (
    PROJECT_ROOT / "models" / "08___Fine_Tuning_voicing" / CONFIG_STAGE
)

FINE_TUNING_CONFIG = {
    "lr": {
        "enabled": False,
        "balance_strengths": [0.00, 0.50, 0.75, 1.00],
        "parameters": {
            "C": [0.1, 1.0, 10.0],
            "solver": "lbfgs",
            "max_iter": 2000,
            "random_state": 42,
        },
        "final_model_bundle": None,
        "final_model_bundle_directory": None,
    },
    "lda": {
        "enabled": False,
        "balance_strengths": [0.00, 0.50, 0.75, 1.00],
        # A list permits conditional grids and excludes invalid combinations.
        "parameter_grid": [
            {"solver": "svd", "shrinkage": None},
            {"solver": "lsqr", "shrinkage": [None, "auto"]},
        ],
        "final_model_bundle": None,
        "final_model_bundle_directory": None,
    },
    "qda": {
        "enabled": False,
        "balance_strengths": [0.00, 0.50, 0.75, 1.00],
        "parameter_grid": {
            "reg_param": [0.0, 0.05, 0.1, 0.2],
        },
        "final_model_bundle": None,
        "final_model_bundle_directory": None,
    },
    "rf": {
        "enabled": True,
        "balance_strengths": [1.00],
        "parameters": {
            "n_estimators": [500],
            "max_depth": [15],
            "min_samples_split": 2,
            "min_samples_leaf": [10],
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        "final_model_bundle": None,
        "final_model_bundle_directory": MODEL_STORAGE_ROOT / "rf_balance100_cfg012",
    },
    "svm": {
        "enabled": False,
        "balance_strengths": [1.00],
        "parameters": {
            "C": [0.1, 1.0, 10.0],
            "kernel": "rbf",
            "gamma": 1.0,
            "cache_size": 2000,
        },
        "final_model_bundle": None,
        "final_model_bundle_directory": None,
    },
}

ANALYSIS_SUFFIX = "voicing_fine_tuning"
EXPERIMENT_NAME = "voicing_fine_tuning"
EXPECTED_VOICING_CLASSES = {"voiced", "voiceless"}
MODEL_NAMES = {
    "lr": "logistic_regression",
    "lda": "linear_discriminant_analysis",
    "qda": "quadratic_discriminant_analysis",
    "rf": "random_forest",
    "svm": "support_vector_machine",
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
    """Expand scalar, Cartesian, or conditional parameter configurations."""
    parameters = model_config.get("parameters", {})
    parameter_grid = model_config.get("parameter_grid", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"{model_key}: parameters must be a dictionary.")
    if isinstance(parameter_grid, dict):
        grids = [parameter_grid]
    elif isinstance(parameter_grid, list) and all(
        isinstance(item, dict) for item in parameter_grid
    ):
        grids = parameter_grid
    else:
        raise ValueError(
            f"{model_key}: parameter_grid must be a dictionary or list of dictionaries."
        )

    combinations = []
    for grid in grids:
        duplicate_keys = sorted(set(parameters) & set(grid))
        if duplicate_keys:
            raise ValueError(
                f"{model_key}: parameters are defined twice: "
                + ", ".join(duplicate_keys)
            )
        combined = {**parameters, **grid}
        names = list(combined)
        value_lists = []
        for name in names:
            configured_value = combined[name]
            if isinstance(configured_value, (list, tuple)):
                values = list(configured_value)
                if not values:
                    raise ValueError(
                        f"{model_key}: parameter '{name}' has no values."
                    )
            else:
                values = [configured_value]
            value_lists.append(values)
        combinations.extend(
            dict(zip(names, values)) for values in product(*value_lists)
        )

    if not combinations:
        combinations = [{}]
    if len({repr(item) for item in combinations}) != len(combinations):
        raise ValueError(f"{model_key}: parameter grid produces duplicate combinations.")
    return combinations


def experiment_folder_name(dataset: str) -> str:
    """Voicing uses the canonical 2000-Hz extraction output in every run."""
    return f"{dataset}_{SOURCE_FREQUENCY_HZ}"


def resolve_input(dataset: str) -> Path:
    feature_dir = (
        config.FEATURE_EXTRACTION_OUTPUT_DIR
        / experiment_folder_name(dataset)
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
    """Return the sole bundle path beneath the central models directory."""
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


def validate_voicing_classes(data):
    present_classes = set(data.labels["voicing_label"].astype(str).unique())
    missing_classes = sorted(EXPECTED_VOICING_CLASSES - present_classes)
    if missing_classes:
        raise ValueError(
            "Data are missing voicing classes: "
            + ", ".join(missing_classes)
        )
    return data


def build_voicing_predictions(identifiers, labels, predicted, probabilities=None):
    output = identifiers.reset_index(drop=True).copy()
    output["true_voicing_label"] = labels["voicing_label"].reset_index(drop=True)
    output["predicted_voicing_label"] = pd.Series(predicted, dtype="string")
    output["voicing_correct"] = (
        output["true_voicing_label"] == output["predicted_voicing_label"]
    )
    if probabilities is not None:
        output = pd.concat(
            [output, probabilities.add_prefix("p_voicing_").reset_index(drop=True)],
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
            estimator = LogisticRegression(
                class_weight=class_weights, **parameters
            )
        elif model_key == "rf":
            estimator = RandomForestClassifier(
                class_weight=class_weights, **parameters
            )
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
    classes = ["voiced", "voiceless"]
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
        output_root / "voicing_fine_tuning_summary.xlsx",
        output_root / "voicing_fine_tuning_summary.csv",
    )


def write_summary_checkpoint(summary_rows: list[dict], output_root: Path) -> None:
    """Persist all completed results after every configuration."""
    if not summary_rows:
        return
    summary = pd.DataFrame(summary_rows).sort_values(
        ["balanced_accuracy", "f1_macro"], ascending=[False, False]
    )
    summary_path, summary_csv_path = summary_paths(output_root)
    temporary_xlsx = summary_path.with_name(
        summary_path.stem + ".tmp" + summary_path.suffix
    )
    temporary_csv = summary_csv_path.with_name(
        summary_csv_path.stem + ".tmp" + summary_csv_path.suffix
    )
    summary.to_excel(temporary_xlsx, index=False)
    temporary_xlsx.replace(summary_path)
    summary.to_csv(temporary_csv, index=False, encoding="utf-8-sig")
    temporary_csv.replace(summary_csv_path)


def load_summary_checkpoint(output_root: Path) -> list[dict]:
    """Restore completed rows when a prior run was interrupted."""
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
        if (
            isinstance(run_name, str)
            and isinstance(bundle_path, str)
            and Path(bundle_path).is_file()
        ):
            completed.add(run_name)
    return completed


def prepare_feature_matrices(model_key: str, training, evaluation):
    """Use the preprocessing of the original model-specific classifiers."""
    if model_key == "rf":
        return None, training.voicing_features, evaluation.voicing_features
    scaler = fit_scaler(training.voicing_features)
    return (
        scaler,
        transform_features(scaler, training.voicing_features),
        transform_features(scaler, evaluation.voicing_features),
    )


def run_training_validation() -> int:
    dataset = "validation"
    enabled_models = validate_configuration()
    config.create_core_output_directories()
    output_root = ensure_directory(fine_tuning_root(dataset))
    ensure_sufficient_disk_space(output_root)
    summary_rows = load_summary_checkpoint(output_root)
    completed_runs = completed_run_names(summary_rows)
    experiment_run_id = timestamp()

    safe_print("\n==================================================")
    safe_print("Voicing Fine-Tuning")
    safe_print("==================================================")
    safe_print(f"Evaluation set : {dataset}")
    safe_print(f"Output root    : {output_root}")
    safe_print("Row filter     : none")
    safe_print(f"Voicing features: {', '.join(VOICING_FEATURE_COLUMNS)}")
    safe_print(
        f"Canonical input: {SOURCE_FREQUENCY_HZ} Hz place-extraction folder "
        "(not a voicing parameter)"
    )
    expected_runs = sum(
        len(model_config["balance_strengths"])
        * len(expand_parameter_combinations(model_key, model_config))
        for model_key, model_config in enabled_models
    )
    safe_print(f"Configured runs: {expected_runs}")
    if completed_runs:
        safe_print(f"Resuming runs  : {len(completed_runs)} already completed")

    training_path = resolve_input("training")
    evaluation_path = resolve_input(dataset)
    training = validate_voicing_classes(prepare_dataset(load_table(training_path)))
    evaluation = validate_voicing_classes(prepare_dataset(load_table(evaluation_path)))

    for model_key, model_config in enabled_models:
        scaler, x_train, x_eval = prepare_feature_matrices(
            model_key, training, evaluation
        )
        labels_train = training.labels["voicing_label"]
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
                run_name = f"{model_key}_balance{code}_{config_code}"
                if run_name in completed_runs:
                    safe_print(f"Skipping completed run: {run_name}")
                    continue

                ensure_sufficient_disk_space(output_root)
                run_dir = ensure_directory(output_root / run_name)
                voicing_model, balance_information = create_estimator(
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
                    f"\nTraining {model_key.upper()} with balance strength "
                    f"{strength:.2f}, {config_code} "
                    f"({parameter_text or 'default parameters'})..."
                )
                voicing_model.fit(x_train, labels_train)
                predicted_voicing = voicing_model.predict(x_eval)
                result = evaluate_predictions(
                    evaluation.labels["voicing_label"],
                    predicted_voicing,
                    target_name="voicing",
                    label_order=list(voicing_model.classes_),
                )
                predictions = build_voicing_predictions(
                    evaluation.identifiers,
                    evaluation.labels,
                    predicted_voicing,
                    probabilities=probability_frame(voicing_model, x_eval),
                )

                run_id = timestamp()
                model_name = MODEL_NAMES[model_key]
                prefix = (
                    f"{dataset}_{model_name}_balance{code}_{config_code}_"
                    f"{ANALYSIS_SUFFIX}_{run_id}"
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
                    "row_filter": "none",
                    "prediction_target": "voicing_label",
                    "source_place_bandpass_low_hz": SOURCE_FREQUENCY_HZ,
                    "source_frequency_role": (
                        "canonical input only; not a voicing parameter"
                    ),
                    "balance_strength": strength,
                    **balance_information,
                    "parameter_configuration_id": config_code,
                    "configured_model_parameters": model_parameters,
                    "configured_parameter_space": {
                        "parameters": model_config.get("parameters", {}),
                        "parameter_grid": model_config.get("parameter_grid", {}),
                    },
                    "fitted_model_parameters": voicing_model.get_params(deep=False),
                    "training_experiment_folder": experiment_folder_name("training"),
                    "evaluation_experiment_folder": experiment_folder_name(dataset),
                    "output_experiment_folder": str(run_dir),
                    "training_subset": "training",
                    "evaluation_subset": dataset,
                    "training_input": str(training_path),
                    "evaluation_input": str(evaluation_path),
                    "voicing_feature_columns": list(VOICING_FEATURE_COLUMNS),
                    "feature_scaling": (
                        "none"
                        if scaler is None
                        else "StandardScaler fitted on training voicing features"
                    ),
                    "training_original_rows": training.original_rows,
                    "training_retained_rows": training.retained_rows,
                    "training_removed_rows": training.removed_rows,
                    "evaluation_original_rows": evaluation.original_rows,
                    "evaluation_retained_rows": evaluation.retained_rows,
                    "evaluation_removed_rows": evaluation.removed_rows,
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
                    result, run_dir / f"{prefix}_voicing_confusion.png"
                )
                save_confusion_matrix(
                    result,
                    run_dir / f"{prefix}_voicing_confusion_normalized.png",
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
                        "row_filter": "none",
                        "prediction_target": "voicing_label",
                        "source_place_bandpass_low_hz": SOURCE_FREQUENCY_HZ,
                        "balance_strength": strength,
                        "balance_information": balance_information,
                        "voicing_feature_columns": list(VOICING_FEATURE_COLUMNS),
                        "voicing_scaler": scaler,
                        "voicing_model": voicing_model,
                        "run_information": run_information,
                    },
                    central_model_path,
                )

                summary_row = {
                    "model": model_name,
                    "model_key": model_key,
                    "source_frequency_hz": SOURCE_FREQUENCY_HZ,
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
                    evaluation.labels["voicing_label"],
                    predicted_voicing,
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
                safe_print(
                    f"Compressed model: {central_model_path} "
                    f"({model_size_mb:.1f} MB)"
                )

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
        "voicing_model",
        "voicing_scaler",
        "voicing_feature_columns",
        "source_place_bandpass_low_hz",
        "row_filter",
        "prediction_target",
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

    bundled_features = list(bundle["voicing_feature_columns"])
    if bundled_features != list(VOICING_FEATURE_COLUMNS):
        raise ValueError(
            f"{model_key}: bundle feature columns do not match the current pipeline."
        )

    bundled_frequency = int(bundle["source_place_bandpass_low_hz"])
    if bundled_frequency != SOURCE_FREQUENCY_HZ:
        raise ValueError(
            f"{model_key}: bundle uses {bundled_frequency} Hz, but the "
            f"canonical input is configured as {SOURCE_FREQUENCY_HZ} Hz."
        )

    if bundle["row_filter"] != "none":
        raise ValueError(
            f"{model_key}: bundle row filter is {bundle['row_filter']!r}; "
            "expected an unfiltered voicing model bundle."
        )
    if bundle["prediction_target"] != "voicing_label":
        raise ValueError(
            f"{model_key}: bundle predicts {bundle['prediction_target']!r}; "
            "expected 'voicing_label'."
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
        "Voicing Final Test Evaluation"
        if dataset == "test"
        else "Voicing Stored-Model Validation"
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
        voicing_model = bundle["voicing_model"]
        voicing_scaler = bundle["voicing_scaler"]
        balance_strength = float(bundle.get("balance_strength", 1.0))
        code = strength_code(balance_strength)

        evaluation_path = resolve_input(dataset)
        evaluation = validate_voicing_classes(
            prepare_dataset(load_table(evaluation_path))
        )
        x_voicing_eval = (
            evaluation.voicing_features
            if voicing_scaler is None
            else transform_features(voicing_scaler, evaluation.voicing_features)
        )

        safe_print(
            f"\nEvaluating {model_key.upper()} on {dataset} using "
            f"the locked bundle:\n{bundle_path}"
        )
        predicted_voicing = voicing_model.predict(x_voicing_eval)
        result = evaluate_predictions(
            evaluation.labels["voicing_label"],
            predicted_voicing,
            target_name="voicing",
            label_order=list(voicing_model.classes_),
        )
        predictions = build_voicing_predictions(
            evaluation.identifiers,
            evaluation.labels,
            predicted_voicing,
            probabilities=probability_frame(voicing_model, x_voicing_eval),
        )

        run_id = timestamp()
        run_name = f"{model_key}_balance{code}_final"
        run_dir = ensure_directory(output_root / run_name)
        prefix = (
            f"{dataset}_{model_name}_balance{code}_final_"
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
            "row_filter": "none",
            "prediction_target": "voicing_label",
            "source_place_bandpass_low_hz": SOURCE_FREQUENCY_HZ,
            "source_frequency_role": "canonical input only; not a voicing parameter",
            "balance_strength": balance_strength,
            "fitted_model_parameters": voicing_model.get_params(deep=False),
            "evaluation_experiment_folder": experiment_folder_name(dataset),
            "output_experiment_folder": str(run_dir),
            "evaluation_subset": dataset,
            "evaluation_input": str(evaluation_path),
            "voicing_feature_columns": list(VOICING_FEATURE_COLUMNS),
            "feature_scaling": "Loaded unchanged from the locked model bundle",
            "evaluation_original_rows": evaluation.original_rows,
            "evaluation_retained_rows": evaluation.retained_rows,
            "evaluation_removed_rows": evaluation.removed_rows,
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
        save_confusion_matrix(result, run_dir / f"{prefix}_voicing_confusion.png")
        save_confusion_matrix(
            result,
            run_dir / f"{prefix}_voicing_confusion_normalized.png",
            normalize=True,
        )

        summary_row = {
            "model": model_name,
            "model_key": model_key,
            "source_frequency_hz": SOURCE_FREQUENCY_HZ,
            "balance_strength": balance_strength,
            **result.summary,
            "source_model_bundle": str(bundle_path),
            "run_directory": str(run_dir),
            "run_id": run_id,
        }
        for parameter_name, parameter_value in voicing_model.get_params(
            deep=False
        ).items():
            if isinstance(parameter_value, (str, int, float, bool, type(None))):
                summary_row[f"parameter_{parameter_name}"] = parameter_value
        append_class_metrics(
            summary_row,
            evaluation.labels["voicing_label"],
            predicted_voicing,
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
    summary_path = output_root / f"voicing_{dataset}_summary.xlsx"
    summary.to_excel(summary_path, index=False)
    summary_csv_path = output_root / f"voicing_{dataset}_summary.csv"
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
