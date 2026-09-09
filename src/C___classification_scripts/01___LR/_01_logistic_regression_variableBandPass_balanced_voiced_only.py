"""_01_logistic_regression_variableBandPass_balanced_voiced_only.py
===============================

Copyright © 2026 Till Preidt (GitHub: MrNemesis98).
This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License
(CC BY-NC 4.0). Reuse and adaptation are permitted for non-commercial purposes,
provided appropriate credit is given."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    save_model_bundle,
    timestamp,
    transform_features,
)

MODEL_NAME = "logistic_regression"
PLACE_BANDPASS_LOW_HZ = 3000  # match feature-extraction folder
ANALYSIS_SUFFIX = "balanced_voiced_only"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a class-balanced voiced-only Logistic Regression place classifier."
    )
    parser.add_argument(
        "--dataset", choices=("validation", "test"), default="validation"
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required for test-set evaluation.",
    )
    parser.add_argument("--training-input", type=Path, default=None)
    parser.add_argument("--evaluation-input", type=Path, default=None)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--c", type=float, default=1.0)
    return parser.parse_args()


def experiment_folder_name(dataset: str) -> str:
    """Return the existing feature-extraction folder, e.g. training_500."""
    return f"{dataset}_{PLACE_BANDPASS_LOW_HZ}"


def output_folder_name(dataset: str) -> str:
    """Return the protected result folder, e.g. validation_500_balanced_voiced_only."""
    return f"{experiment_folder_name(dataset)}_{ANALYSIS_SUFFIX}"


def configured_feature_directory(dataset: str) -> Path:
    return config.FEATURE_EXTRACTION_OUTPUT_DIR / experiment_folder_name(dataset)


def configured_output_directory(dataset: str) -> Path:
    base_dir = (
        config.LR_VALIDATION_DIR.parent
        if dataset == "validation"
        else config.LR_TEST_DIR.parent
    )
    return base_dir / output_folder_name(dataset)


def resolve_input(explicit_path: Path | None, dataset: str) -> Path:
    return (
        explicit_path.resolve()
        if explicit_path
        else find_feature_table(configured_feature_directory(dataset))
    )


def retain_voiced_rows(data):
    """Return PreparedData containing only rows labelled voiced."""
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
    expected_classes = {"dental", "retroflex", "alveolo-palatal"}
    present_classes = set(filtered.labels["place_label"].astype(str).unique())
    missing_classes = sorted(expected_classes - present_classes)
    if missing_classes:
        raise ValueError(
            "Voiced-only data are missing place classes: " + ", ".join(missing_classes)
        )
    return filtered


def build_place_predictions(identifiers, labels, predicted, probabilities=None):
    """Build a prediction table without redundant voicing or combined targets."""
    output = identifiers.reset_index(drop=True).copy()
    output["true_place_label"] = labels["place_label"].reset_index(drop=True)
    output["predicted_place_label"] = pd.Series(predicted, dtype="string")
    output["place_correct"] = (
        output["true_place_label"] == output["predicted_place_label"]
    )
    if probabilities is not None:
        output = pd.concat(
            [
                output,
                probabilities.add_prefix("p_place_").reset_index(drop=True),
            ],
            axis=1,
        )
    return output


def create_estimator(args: argparse.Namespace) -> LogisticRegression:
    return LogisticRegression(
        C=args.c,
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )


def main() -> int:
    args = parse_args()
    if args.dataset == "test" and not args.allow_test:
        safe_print(
            "ERROR: Test-set evaluation is protected. "
            "Use --dataset test --allow-test only after model selection."
        )
        return 2

    config.create_core_output_directories()
    training_path = resolve_input(args.training_input, "training")
    evaluation_path = resolve_input(args.evaluation_input, args.dataset)
    output_dir = ensure_directory(configured_output_directory(args.dataset))
    model_dir = ensure_directory(config.LR_MODELS_DIR / ANALYSIS_SUFFIX)

    safe_print("\n==================================================")
    safe_print("Logistic Regression Place Classification - Balanced Voiced Only")
    safe_print("==================================================")
    safe_print(f"Training input   : {training_path}")
    safe_print(f"Evaluation input : {evaluation_path}")
    safe_print(f"Evaluation set   : {args.dataset}")
    safe_print(f"BandPassLowHz    : {PLACE_BANDPASS_LOW_HZ} Hz")
    safe_print(f"Training folder : {experiment_folder_name('training')}")
    safe_print(f"Evaluation folder: {experiment_folder_name(args.dataset)}")
    safe_print(f"Output folder    : {output_folder_name(args.dataset)}")
    safe_print("Row filter       : voicing_label == voiced")
    safe_print(f"Place features   : {', '.join(PLACE_FEATURE_COLUMNS)}")

    training_all = prepare_dataset(load_table(training_path))
    evaluation_all = prepare_dataset(load_table(evaluation_path))
    training = retain_voiced_rows(training_all)
    evaluation = retain_voiced_rows(evaluation_all)

    safe_print(
        f"Training rows    : {training.retained_rows} voiced of "
        f"{training_all.retained_rows} valid rows"
    )
    safe_print(
        f"Evaluation rows  : {evaluation.retained_rows} voiced of "
        f"{evaluation_all.retained_rows} valid rows"
    )

    place_scaler = fit_scaler(training.place_features)
    x_place_train = transform_features(place_scaler, training.place_features)
    x_place_eval = transform_features(place_scaler, evaluation.place_features)

    place_model = create_estimator(args)
    safe_print("\nTraining place classifier...")
    place_model.fit(x_place_train, training.labels["place_label"])
    predicted_place = place_model.predict(x_place_eval)

    result = evaluate_predictions(
        evaluation.labels["place_label"],
        predicted_place,
        target_name="place",
        label_order=list(place_model.classes_),
    )
    results = [result]
    predictions = build_place_predictions(
        evaluation.identifiers,
        evaluation.labels,
        predicted_place,
        probabilities=probability_frame(place_model, x_place_eval),
    )

    run_id = timestamp()
    prefix = (
        f"{args.dataset}_{MODEL_NAME}_{PLACE_BANDPASS_LOW_HZ}Hz_"
        f"{ANALYSIS_SUFFIX}_{run_id}"
    )
    predictions_path = output_dir / f"{prefix}_predictions.csv"
    metrics_path = output_dir / f"{prefix}_metrics.xlsx"
    info_path = output_dir / f"{prefix}_run_information.json"
    model_path = (
        model_dir
        / f"{MODEL_NAME}_{PLACE_BANDPASS_LOW_HZ}Hz_{ANALYSIS_SUFFIX}_{run_id}.joblib"
    )
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    run_information = {
        "model": MODEL_NAME,
        "experiment": "variableBandPass_balanced_voiced_only",
        "analysis_scope": ANALYSIS_SUFFIX,
        "row_filter": "voicing_label == voiced",
        "prediction_target": "place_label",
        "place_bandpass_low_hz": PLACE_BANDPASS_LOW_HZ,
        "training_experiment_folder": experiment_folder_name("training"),
        "evaluation_experiment_folder": experiment_folder_name(args.dataset),
        "output_experiment_folder": output_folder_name(args.dataset),
        "training_subset": "training",
        "evaluation_subset": args.dataset,
        "training_input": str(training_path),
        "evaluation_input": str(evaluation_path),
        "place_feature_columns": list(PLACE_FEATURE_COLUMNS),
        "feature_scaling": "StandardScaler fitted on voiced training place features",
        "C": args.c,
        "class_weight": "balanced",
        "max_iter": args.max_iter,
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
        "training_original_rows": training_all.original_rows,
        "training_valid_rows_before_filter": training_all.retained_rows,
        "training_voiced_rows": training.retained_rows,
        "training_removed_invalid_or_voiceless_rows": training.removed_rows,
        "evaluation_original_rows": evaluation_all.original_rows,
        "evaluation_valid_rows_before_filter": evaluation_all.retained_rows,
        "evaluation_voiced_rows": evaluation.retained_rows,
        "evaluation_removed_invalid_or_voiceless_rows": evaluation.removed_rows,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "run_id": run_id,
    }

    save_evaluation_workbook(results, metrics_path, run_information=run_information)
    save_json(run_information, info_path)
    save_confusion_matrix(result, output_dir / f"{prefix}_place_confusion.png")
    save_confusion_matrix(
        result,
        output_dir / f"{prefix}_place_confusion_normalized.png",
        normalize=True,
    )
    save_model_bundle(
        {
            "model_name": MODEL_NAME,
            "experiment": "variableBandPass_balanced_voiced_only",
            "analysis_scope": ANALYSIS_SUFFIX,
            "row_filter": "voicing_label == voiced",
            "prediction_target": "place_label",
            "place_bandpass_low_hz": PLACE_BANDPASS_LOW_HZ,
            "place_feature_columns": list(PLACE_FEATURE_COLUMNS),
            "place_scaler": place_scaler,
            "place_model": place_model,
            "run_information": run_information,
        },
        model_path,
    )

    safe_print("\n==================================================")
    safe_print("Evaluation summary")
    safe_print("==================================================")
    safe_print(
        f"place | accuracy={result.summary['accuracy']:.4f} | "
        f"balanced_accuracy={result.summary['balanced_accuracy']:.4f} | "
        f"macro_F1={result.summary['f1_macro']:.4f}"
    )
    safe_print(f"\nPredictions : {predictions_path}")
    safe_print(f"Metrics     : {metrics_path}")
    safe_print(f"Run info    : {info_path}")
    safe_print(f"Model bundle: {model_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        safe_print("\nInterrupted by user.")
        raise SystemExit(130)
    except Exception as error:
        safe_print(f"\nERROR: {type(error).__name__}: {error}")
        raise
