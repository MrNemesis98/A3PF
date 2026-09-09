"""_00_classification_utils.py
===============================

Copyright © 2026 Till Preidt (GitHub: MrNemesis98).
This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License
(CC BY-NC 4.0). Reuse and adaptation are permitted for non-commercial purposes,
provided appropriate credit is given."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

# Place and voicing are acoustically distinct classification tasks. The place
# model uses spectral moments from the high-frequency frication-noise band;
# the voicing model uses duration, detected periodicity, low-frequency energy,
# and harmonicity.
PLACE_FEATURE_COLUMNS = (
    "cog_hz",
    "spread_hz",
    "skewness",
    "kurtosis")

VOICING_FEATURE_COLUMNS = (
    "duration_s",
    "voiced_fraction",
    "low_total_intensity_ratio_db",
    "mean_hnr_db",
)
ALL_FEATURE_COLUMNS = PLACE_FEATURE_COLUMNS + VOICING_FEATURE_COLUMNS

# Compatibility alias for classification scripts that have not yet been
# migrated. It intentionally preserves the former four-feature place matrix.
FEATURE_COLUMNS = PLACE_FEATURE_COLUMNS

IDENTIFIER_COLUMNS = (
    "dataset",
    "file_name",
    "interval_index",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "mfa_ground_truth",
)


@dataclass(frozen=True)
class PreparedData:
    place_features: pd.DataFrame
    voicing_features: pd.DataFrame
    labels: pd.DataFrame
    identifiers: pd.DataFrame
    original_rows: int
    retained_rows: int
    removed_rows: int

    @property
    def features(self) -> pd.DataFrame:
        """Compatibility view returning the original place-feature matrix."""
        return self.place_features


@dataclass(frozen=True)
class EvaluationResult:
    target_name: str
    labels: list[str]
    summary: dict[str, float | int | str]
    class_report: pd.DataFrame
    confusion: pd.DataFrame


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def safe_print(message: object = "") -> None:
    text = str(message)
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_text_encoding(path: Path) -> str:
    with path.open("rb") as handle:
        prefix = handle.read(4)
    if prefix.startswith(b"\xff\xfe") or prefix.startswith(b"\xfe\xff"):
        return "utf-16"
    if prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def detect_delimiter(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
        sample = handle.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        counts = {delimiter: first.count(delimiter) for delimiter in (",", ";", "\t")}
        return max(counts, key=counts.get) if any(counts.values()) else ","


def find_feature_table(directory: Path) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(f"Feature directory does not exist: {directory}")
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
    )
    if not candidates:
        raise FileNotFoundError(f"No CSV or Excel feature table found in: {directory}")

    def priority(path: Path) -> tuple[int, str]:
        name = path.name.casefold()
        return (0 if "token" in name else 1 if "feature" in name else 2, name)

    return min(candidates, key=priority)


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        encoding = detect_text_encoding(path)
        return pd.read_csv(
            path,
            encoding=encoding,
            sep=detect_delimiter(path, encoding),
            low_memory=False,
        )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def normalize_label_text(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    for old, new in {
        "alveopalatal": "alveolo-palatal",
        "alveolo_palatal": "alveolo-palatal",
        "alveolo palatal": "alveolo-palatal",
    }.items():
        result = result.str.replace(old, new, regex=False)
    return result


def derive_combined_label(place: pd.Series, voicing: pd.Series) -> pd.Series:
    return place.astype("string").str.strip() + "_" + voicing.astype("string").str.strip()


def _finite_row_mask(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    values = frame[list(columns)]
    mask = values.notna().all(axis=1)
    mask &= np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    return mask


def prepare_dataset(frame: pd.DataFrame) -> PreparedData:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]

    required = set(ALL_FEATURE_COLUMNS) | {"place_label", "voicing_label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Required columns are missing: " + ", ".join(missing))

    frame["place_label"] = normalize_label_text(frame["place_label"])
    frame["voicing_label"] = normalize_label_text(frame["voicing_label"])
    if "combined_label" in frame.columns:
        frame["combined_label"] = normalize_label_text(frame["combined_label"])
    else:
        frame["combined_label"] = derive_combined_label(
            frame["place_label"], frame["voicing_label"]
        )

    for column in ALL_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # A common row mask keeps place, voicing, and combined predictions aligned.
    # The Praat extractor emits numeric fallback values for undefined HNR, so
    # valid voiceless tokens are not removed merely because F0 is absent.
    valid = _finite_row_mask(frame, ALL_FEATURE_COLUMNS)
    valid &= frame["voiced_fraction"].between(0.0, 1.0, inclusive="both")
    for target in ("place_label", "voicing_label"):
        valid &= frame[target].notna() & frame[target].str.len().fillna(0).gt(0)

    cleaned = frame.loc[valid].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No valid rows remain after validation.")

    identifiers = pd.DataFrame(index=cleaned.index)
    for column in IDENTIFIER_COLUMNS:
        if column in cleaned.columns:
            identifiers[column] = cleaned[column]

    return PreparedData(
        place_features=cleaned[list(PLACE_FEATURE_COLUMNS)].astype(float),
        voicing_features=cleaned[list(VOICING_FEATURE_COLUMNS)].astype(float),
        labels=cleaned[["place_label", "voicing_label", "combined_label"]].copy(),
        identifiers=identifiers,
        original_rows=len(frame),
        retained_rows=len(cleaned),
        removed_rows=len(frame) - len(cleaned),
    )


def fit_scaler(training_features: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(training_features)


def transform_features(
    scaler: StandardScaler, features: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(
        scaler.transform(features),
        columns=features.columns,
        index=features.index,
    )


def evaluate_predictions(
    y_true,
    y_pred,
    *,
    target_name: str,
    label_order=None,
) -> EvaluationResult:
    true = pd.Series(y_true, dtype="string")
    pred = pd.Series(y_pred, dtype="string")
    labels = (
        list(label_order)
        if label_order is not None
        else sorted(set(true.dropna()) | set(pred.dropna()))
    )
    summary = {
        "target": target_name,
        "n_tokens": len(true),
        "n_classes": len(labels),
        "accuracy": accuracy_score(true, pred),
        "balanced_accuracy": balanced_accuracy_score(true, pred),
        "precision_macro": precision_score(
            true, pred, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            true, pred, labels=labels, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            true, pred, labels=labels, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            true, pred, labels=labels, average="weighted", zero_division=0
        ),
    }
    report = classification_report(
        true,
        pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    class_report = pd.DataFrame(
        [
            {
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1_score": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in labels
        ]
    )
    matrix = pd.DataFrame(
        confusion_matrix(true, pred, labels=labels),
        index=labels,
        columns=labels,
    )
    matrix.index.name, matrix.columns.name = "true", "predicted"
    return EvaluationResult(target_name, labels, summary, class_report, matrix)


def save_confusion_matrix(
    result: EvaluationResult,
    output_path: Path,
    *,
    normalize: bool = False,
    dpi: int = 200,
) -> None:
    matrix = result.confusion.to_numpy(dtype=float)
    if normalize:
        sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            sums,
            out=np.zeros_like(matrix),
            where=sums != 0,
        )

    size = max(6.5, 0.9 * len(result.labels) + 3)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(result.labels)),
        yticks=np.arange(len(result.labels)),
        xticklabels=result.labels,
        yticklabels=result.labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{result.target_name} confusion matrix"
        + (" (row-normalized)" if normalize else ""),
    )
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}" if normalize else str(int(value)),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def probability_frame(model: Any, features: pd.DataFrame) -> pd.DataFrame | None:
    if not hasattr(model, "predict_proba"):
        return None
    return pd.DataFrame(
        model.predict_proba(features),
        columns=[str(class_name) for class_name in model.classes_],
    )


def build_predictions_table(
    identifiers,
    true_labels,
    predicted_place,
    predicted_voicing,
    *,
    place_probabilities=None,
    voicing_probabilities=None,
) -> pd.DataFrame:
    output = identifiers.reset_index(drop=True).copy()
    output["true_place_label"] = true_labels["place_label"].reset_index(drop=True)
    output["predicted_place_label"] = pd.Series(predicted_place, dtype="string")
    output["place_correct"] = (
        output["true_place_label"] == output["predicted_place_label"]
    )
    output["true_voicing_label"] = true_labels["voicing_label"].reset_index(drop=True)
    output["predicted_voicing_label"] = pd.Series(predicted_voicing, dtype="string")
    output["voicing_correct"] = (
        output["true_voicing_label"] == output["predicted_voicing_label"]
    )
    output["true_combined_label"] = true_labels["combined_label"].reset_index(drop=True)
    output["predicted_combined_label"] = derive_combined_label(
        output["predicted_place_label"], output["predicted_voicing_label"]
    )
    output["combined_correct"] = (
        output["true_combined_label"] == output["predicted_combined_label"]
    )
    if place_probabilities is not None:
        output = pd.concat(
            [
                output,
                place_probabilities.add_prefix("p_place_").reset_index(drop=True),
            ],
            axis=1,
        )
    if voicing_probabilities is not None:
        output = pd.concat(
            [
                output,
                voicing_probabilities.add_prefix("p_voicing_").reset_index(drop=True),
            ],
            axis=1,
        )
    return output


def save_evaluation_workbook(
    results,
    output_path: Path,
    *,
    run_information: Mapping[str, Any] | None = None,
) -> None:
    ensure_directory(output_path.parent)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([result.summary for result in results]).to_excel(
            writer, sheet_name="summary", index=False
        )
        if run_information:
            pd.DataFrame(
                [
                    {"parameter": key, "value": value}
                    for key, value in run_information.items()
                ]
            ).to_excel(writer, sheet_name="run_information", index=False)
        for result in results:
            result.class_report.to_excel(
                writer,
                sheet_name=f"{result.target_name}_classes"[:31],
                index=False,
            )
            result.confusion.to_excel(
                writer,
                sheet_name=f"{result.target_name}_confusion"[:31],
            )


def save_json(data: Mapping[str, Any], path: Path) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def save_model_bundle(bundle: Mapping[str, Any], output_path: Path) -> None:
    ensure_directory(output_path.parent)
    joblib.dump(dict(bundle), output_path)


def print_dataset_summary(name: str, data: PreparedData) -> None:
    safe_print(f"\n{name}\n{'-' * len(name)}")
    safe_print(f"Original rows : {data.original_rows}")
    safe_print(f"Retained rows : {data.retained_rows}")
    safe_print(f"Removed rows  : {data.removed_rows}")
    safe_print(
        "Place classes : "
        + ", ".join(sorted(data.labels["place_label"].astype(str).unique()))
    )
    safe_print(
        "Voicing classes: "
        + ", ".join(sorted(data.labels["voicing_label"].astype(str).unique()))
    )
