"""_03_quadratic_discriminant_analysis_balanced_voicing.py
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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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
    save_model_bundle,
    timestamp,
    transform_features,
)

MODEL_NAME = "quadratic_discriminant_analysis"
MODEL_FOLDER = "03___QDA"
SOURCE_PLACE_BANDPASS_LOW_HZ = 2000
ANALYSIS_SUFFIX = "balanced_voicing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate fully balanced QDA for voicing."
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
    return parser.parse_args()


def source_experiment_folder(dataset: str) -> str:
    return f"{dataset}_{SOURCE_PLACE_BANDPASS_LOW_HZ}"


def configured_feature_directory(dataset: str) -> Path:
    return config.FEATURE_EXTRACTION_OUTPUT_DIR / source_experiment_folder(dataset)


def configured_output_directory(dataset: str) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "C___classification"
        / MODEL_FOLDER
        / f"{dataset}_{ANALYSIS_SUFFIX}"
    )


def configured_model_directory() -> Path:
    return PROJECT_ROOT / "models" / MODEL_FOLDER / ANALYSIS_SUFFIX


def resolve_input(explicit_path: Path | None, dataset: str) -> Path:
    return (
        explicit_path.resolve()
        if explicit_path
        else find_feature_table(configured_feature_directory(dataset))
    )


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


def create_estimator(args: argparse.Namespace) -> QuadraticDiscriminantAnalysis:
    return QuadraticDiscriminantAnalysis(priors=[0.5, 0.5])


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
    model_dir = ensure_directory(configured_model_directory())

    safe_print("\n==================================================")
    safe_print("QDA Voicing Classification - Full Balance")
    safe_print("==================================================")
    safe_print(f"Training input   : {training_path}")
    safe_print(f"Evaluation input : {evaluation_path}")
    safe_print(f"Evaluation set   : {args.dataset}")
    safe_print(f"Source folder    : {source_experiment_folder('training')}")
    safe_print(f"Analysis suffix  : {ANALYSIS_SUFFIX}")
    safe_print("Class balancing  : uniform priors [0.5, 0.5]")
    safe_print(f"Voicing features : {', '.join(VOICING_FEATURE_COLUMNS)}")

    training = prepare_dataset(load_table(training_path))
    evaluation = prepare_dataset(load_table(evaluation_path))
    present_classes = set(training.labels["voicing_label"].astype(str).unique())
    if present_classes != {"voiced", "voiceless"}:
        raise ValueError(
            "Training data must contain both voicing classes; found: "
            + ", ".join(sorted(present_classes))
        )

    voicing_scaler = fit_scaler(training.voicing_features)
    x_train = transform_features(voicing_scaler, training.voicing_features)
    x_eval = transform_features(voicing_scaler, evaluation.voicing_features)
    voicing_model = create_estimator(args)
    safe_print("\nTraining fully balanced voicing classifier...")
    voicing_model.fit(x_train, training.labels["voicing_label"])
    predicted = voicing_model.predict(x_eval)

    result = evaluate_predictions(
        evaluation.labels["voicing_label"],
        predicted,
        target_name="voicing",
        label_order=list(voicing_model.classes_),
    )
    predictions = build_voicing_predictions(
        evaluation.identifiers,
        evaluation.labels,
        predicted,
        probabilities=probability_frame(voicing_model, x_eval),
    )

    run_id = timestamp()
    prefix = f"{args.dataset}_{MODEL_NAME}_{ANALYSIS_SUFFIX}_{run_id}"
    predictions_path = output_dir / f"{prefix}_predictions.csv"
    metrics_path = output_dir / f"{prefix}_metrics.xlsx"
    info_path = output_dir / f"{prefix}_run_information.json"
    model_path = model_dir / f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_{run_id}.joblib"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    run_information = {
        "model": MODEL_NAME,
        "experiment": ANALYSIS_SUFFIX,
        "analysis_scope": ANALYSIS_SUFFIX,
        "row_filter": "none",
        "prediction_target": "voicing_label",
        "source_place_bandpass_low_hz": SOURCE_PLACE_BANDPASS_LOW_HZ,
        "source_frequency_role": "canonical input only; not a voicing parameter",
        "training_experiment_folder": source_experiment_folder("training"),
        "evaluation_experiment_folder": source_experiment_folder(args.dataset),
        "output_experiment_folder": str(output_dir),
        "training_subset": "training",
        "evaluation_subset": args.dataset,
        "training_input": str(training_path),
        "evaluation_input": str(evaluation_path),
        "voicing_feature_columns": list(VOICING_FEATURE_COLUMNS),
        "feature_scaling": "StandardScaler fitted on training voicing features",
        "balancing": "full",
        "class_priors": {"voiced": 0.5, "voiceless": 0.5},
        "model_parameters": voicing_model.get_params(deep=False),
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

    save_evaluation_workbook([result], metrics_path, run_information=run_information)
    save_json(run_information, info_path)
    save_confusion_matrix(result, output_dir / f"{prefix}_voicing_confusion.png")
    save_confusion_matrix(
        result,
        output_dir / f"{prefix}_voicing_confusion_normalized.png",
        normalize=True,
    )
    save_model_bundle(
        {
            "model_name": MODEL_NAME,
            "experiment": ANALYSIS_SUFFIX,
            "analysis_scope": ANALYSIS_SUFFIX,
            "row_filter": "none",
            "prediction_target": "voicing_label",
            "source_place_bandpass_low_hz": SOURCE_PLACE_BANDPASS_LOW_HZ,
            "voicing_feature_columns": list(VOICING_FEATURE_COLUMNS),
            "voicing_scaler": voicing_scaler,
            "voicing_model": voicing_model,
            "run_information": run_information,
        },
        model_path,
    )

    safe_print("\n==================================================")
    safe_print("Evaluation summary")
    safe_print("==================================================")
    safe_print(
        f"voicing | accuracy={result.summary['accuracy']:.4f} | "
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
