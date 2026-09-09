"""_02_linear_discriminant_analysis_variableBandPass.py
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SCRIPT_DIR = Path(__file__).resolve().parent
CLASSIFICATION_SCRIPTS_DIR = SCRIPT_DIR.parent
SRC_DIR = CLASSIFICATION_SCRIPTS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CLASSIFICATION_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_SCRIPTS_DIR))

import _00_config as config
from _00_classification_utils import (
    PLACE_FEATURE_COLUMNS,
    VOICING_FEATURE_COLUMNS,
    build_predictions_table,
    ensure_directory,
    evaluate_predictions,
    find_feature_table,
    fit_scaler,
    load_table,
    prepare_dataset,
    print_dataset_summary,
    probability_frame,
    safe_print,
    save_confusion_matrix,
    save_evaluation_workbook,
    save_json,
    save_model_bundle,
    timestamp,
    transform_features,
)

MODEL_NAME = "linear_discriminant_analysis"
PLACE_BANDPASS_LOW_HZ = 500  # must match the variable-band-pass feature extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on training data and evaluate Linear Discriminant Analysis."
    )
    parser.add_argument(
        "--dataset",
        choices=("validation", "test"),
        default="validation",
        help="Dataset used for evaluation. Default: validation.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required for test-set evaluation.",
    )
    parser.add_argument("--training-input", type=Path, default=None)
    parser.add_argument("--evaluation-input", type=Path, default=None)
    parser.add_argument(
        "--solver",
        choices=("svd", "lsqr", "eigen"),
        default="svd",
        help="LDA solver. Default: svd.",
    )
    parser.add_argument(
        "--shrinkage",
        default="none",
        help=(
            "Covariance shrinkage: 'none', 'auto', or a float from 0 to 1. "
            "Shrinkage is only available with solver lsqr or eigen."
        ),
    )
    return parser.parse_args()


def experiment_folder_name(dataset: str) -> str:
    """Return experiment folder name, e.g. training_4000."""
    return f"{dataset}_{PLACE_BANDPASS_LOW_HZ}"


def configured_feature_directory(dataset: str) -> Path:
    return config.FEATURE_EXTRACTION_OUTPUT_DIR / experiment_folder_name(dataset)


def configured_output_directory(dataset: str) -> Path:
    base_dir = (
        config.LDA_VALIDATION_DIR.parent
        if dataset == "validation"
        else config.LDA_TEST_DIR.parent
    )
    return base_dir / experiment_folder_name(dataset)


def resolve_input(explicit_path: Path | None, dataset: str) -> Path:
    return (
        explicit_path.resolve()
        if explicit_path
        else find_feature_table(configured_feature_directory(dataset))
    )


def parse_shrinkage(value: str) -> str | float | None:
    normalized = value.strip().lower()
    if normalized == "none":
        return None
    if normalized == "auto":
        return "auto"

    try:
        numeric = float(normalized)
    except ValueError as error:
        raise ValueError(
            "--shrinkage must be 'none', 'auto', or a float from 0 to 1."
        ) from error

    if not 0.0 <= numeric <= 1.0:
        raise ValueError("Numeric shrinkage must be between 0 and 1.")
    return numeric


def create_estimator(args: argparse.Namespace) -> LinearDiscriminantAnalysis:
    shrinkage = parse_shrinkage(args.shrinkage)
    if args.solver == "svd" and shrinkage is not None:
        raise ValueError(
            "Shrinkage cannot be used with solver='svd'. "
            "Use --solver lsqr or --solver eigen."
        )
    return LinearDiscriminantAnalysis(solver=args.solver, shrinkage=shrinkage)


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
    model_dir = ensure_directory(config.LDA_MODELS_DIR)

    safe_print("\n==================================================")
    safe_print("Linear Discriminant Analysis Classification")
    safe_print("==================================================")
    safe_print(f"Training input  : {training_path}")
    safe_print(f"Evaluation input: {evaluation_path}")
    safe_print(f"Evaluation set  : {args.dataset}")
    safe_print(f"BandPassLowHz    : {PLACE_BANDPASS_LOW_HZ} Hz")
    safe_print(f"Training folder : {experiment_folder_name('training')}")
    safe_print(f"Evaluation folder: {experiment_folder_name(args.dataset)}")
    safe_print(f"Place features  : {', '.join(PLACE_FEATURE_COLUMNS)}")
    safe_print(f"Voicing features: {', '.join(VOICING_FEATURE_COLUMNS)}")
    safe_print(f"Solver          : {args.solver}")
    safe_print(f"Shrinkage       : {args.shrinkage}")

    training = prepare_dataset(load_table(training_path))
    evaluation = prepare_dataset(load_table(evaluation_path))
    print_dataset_summary("Training data", training)
    print_dataset_summary(f"{args.dataset.capitalize()} data", evaluation)

    # Fit two independent scalers on training data only. This prevents the
    # voicing predictors from influencing the place transformation and vice versa.
    place_scaler = fit_scaler(training.place_features)
    voicing_scaler = fit_scaler(training.voicing_features)
    x_place_train = transform_features(place_scaler, training.place_features)
    x_place_eval = transform_features(place_scaler, evaluation.place_features)
    x_voicing_train = transform_features(voicing_scaler, training.voicing_features)
    x_voicing_eval = transform_features(voicing_scaler, evaluation.voicing_features)

    place_model = create_estimator(args)
    voicing_model = create_estimator(args)

    safe_print("\nTraining place classifier...")
    place_model.fit(x_place_train, training.labels["place_label"])
    safe_print("Training voicing classifier...")
    voicing_model.fit(x_voicing_train, training.labels["voicing_label"])

    predicted_place = place_model.predict(x_place_eval)
    predicted_voicing = voicing_model.predict(x_voicing_eval)
    predicted_combined = (
        pd.Series(predicted_place, dtype="string")
        + "_"
        + pd.Series(predicted_voicing, dtype="string")
    )

    results = [
        evaluate_predictions(
            evaluation.labels["place_label"],
            predicted_place,
            target_name="place",
            label_order=list(place_model.classes_),
        ),
        evaluate_predictions(
            evaluation.labels["voicing_label"],
            predicted_voicing,
            target_name="voicing",
            label_order=list(voicing_model.classes_),
        ),
        evaluate_predictions(
            evaluation.labels["combined_label"],
            predicted_combined,
            target_name="combined",
        ),
    ]

    predictions = build_predictions_table(
        evaluation.identifiers,
        evaluation.labels,
        predicted_place,
        predicted_voicing,
        place_probabilities=probability_frame(place_model, x_place_eval),
        voicing_probabilities=probability_frame(voicing_model, x_voicing_eval),
    )

    run_id = timestamp()
    prefix = f"{args.dataset}_{MODEL_NAME}_{PLACE_BANDPASS_LOW_HZ}Hz_{run_id}"
    predictions_path = output_dir / f"{prefix}_predictions.csv"
    metrics_path = output_dir / f"{prefix}_metrics.xlsx"
    info_path = output_dir / f"{prefix}_run_information.json"
    model_path = model_dir / f"{MODEL_NAME}_{PLACE_BANDPASS_LOW_HZ}Hz_{run_id}.joblib"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    shrinkage = parse_shrinkage(args.shrinkage)
    run_information = {
        "model": MODEL_NAME,
        "experiment": "variableBandPass",
        "place_bandpass_low_hz": PLACE_BANDPASS_LOW_HZ,
        "training_experiment_folder": experiment_folder_name("training"),
        "evaluation_experiment_folder": experiment_folder_name(args.dataset),
        "training_subset": "training",
        "evaluation_subset": args.dataset,
        "training_input": str(training_path),
        "evaluation_input": str(evaluation_path),
        "place_feature_columns": list(PLACE_FEATURE_COLUMNS),
        "voicing_feature_columns": list(VOICING_FEATURE_COLUMNS),
        "solver": args.solver,
        "shrinkage": shrinkage,
        "training_original_rows": training.original_rows,
        "training_retained_rows": training.retained_rows,
        "training_removed_rows": training.removed_rows,
        "evaluation_original_rows": evaluation.original_rows,
        "evaluation_retained_rows": evaluation.retained_rows,
        "evaluation_removed_rows": evaluation.removed_rows,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "run_id": run_id,
    }

    save_evaluation_workbook(results, metrics_path, run_information=run_information)
    save_json(run_information, info_path)
    for result in results:
        save_confusion_matrix(
            result, output_dir / f"{prefix}_{result.target_name}_confusion.png"
        )
        save_confusion_matrix(
            result,
            output_dir / f"{prefix}_{result.target_name}_confusion_normalized.png",
            normalize=True,
        )

    save_model_bundle(
        {
            "model_name": MODEL_NAME,
            "experiment": "variableBandPass",
            "place_bandpass_low_hz": PLACE_BANDPASS_LOW_HZ,
            "place_feature_columns": list(PLACE_FEATURE_COLUMNS),
            "voicing_feature_columns": list(VOICING_FEATURE_COLUMNS),
            "place_scaler": place_scaler,
            "voicing_scaler": voicing_scaler,
            "place_model": place_model,
            "voicing_model": voicing_model,
            "run_information": run_information,
        },
        model_path,
    )

    safe_print("\n==================================================")
    safe_print("Evaluation summary")
    safe_print("==================================================")
    for result in results:
        safe_print(
            f"{result.target_name:8s} | "
            f"accuracy={result.summary['accuracy']:.4f} | "
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
