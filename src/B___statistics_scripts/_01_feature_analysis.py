"""
_01_feature_analysis.py
===============================

Copyright © 2026 Till Preidt (GitHub: MrNemesis98).
This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License
(CC BY-NC 4.0). Reuse and adaptation are permitted for non-commercial purposes,
provided appropriate credit is given.

Statistical summary of token-level fricative measurements.

Input
-----
data/A___preprocessing/05___feature_extraction_output/<subset>/<subset>_features.csv

Output
------
data/B___statistics/01___feature_analysis_output/<subset>/
    <subset>_feature_summary.csv
    <subset>_feature_summary.xlsx
    <subset>_feature_analysis_log.txt

The script reads the token-level output produced by _04_feature_extraction.praat
and calculates count, percentage, P10, median, and P90 for each fricative type
and each place- or voicing-related acoustic feature.

No acoustic measurements are performed here.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# =============================================================================
# Load central project configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import _00_config as config  # noqa: E402


# =============================================================================
# Script-specific settings
# =============================================================================

DATASET_SUFFIX = "training"  # allowed: training, validation, test

PLACE_FEATURE_COLUMNS = (
    "cog_hz",
    "spread_hz",
    "skewness",
    "kurtosis",
)

VOICING_FEATURE_COLUMNS = (
    "duration_s",
    "voiced_fraction",
    "low_total_intensity_ratio_db",
    "mean_hnr_db",
)

FEATURE_COLUMNS = PLACE_FEATURE_COLUMNS + VOICING_FEATURE_COLUMNS

CLASS_COLUMN = "combined_label"

CLASS_ORDER = (
    "dental_voiceless",
    "dental_voiced",
    "retroflex_voiceless",
    "retroflex_voiced",
    "alveolo-palatal_voiceless",
    "alveolo-palatal_voiced",
)

CLASS_ALIASES = {
    "alveopalatal_voiceless": "alveolo-palatal_voiceless",
    "alveopalatal_voiced": "alveolo-palatal_voiced",
    "alveolo_palatal_voiceless": "alveolo-palatal_voiceless",
    "alveolo_palatal_voiced": "alveolo-palatal_voiced",
}

ROUND_DIGITS = 3
CREATE_XLSX = True


# =============================================================================
# Path handling
# =============================================================================

def validate_dataset_suffix(dataset_suffix: str) -> str:
    """Validate the subset through the central project configuration."""
    try:
        return config.validate_dataset_suffix(dataset_suffix)
    except ValueError as exc:
        raise ValueError(
            f"Invalid DATASET_SUFFIX: {dataset_suffix!r}. "
            "Use 'training', 'validation', or 'test'."
        ) from exc


def get_paths(dataset_suffix: str) -> tuple[Path, Path, Path, Path]:
    """Build all paths from _00_config.py."""
    input_file = (
        config.dataset_directory(
            config.FEATURE_EXTRACTION_OUTPUT_DIR,
            dataset_suffix,
        )
        / f"{dataset_suffix}_features.csv"
    )

    output_dir = config.dataset_directory(
        config.FEATURE_ANALYSIS_OUTPUT_DIR,
        dataset_suffix,
    )

    csv_output = output_dir / f"{dataset_suffix}_feature_summary.csv"
    xlsx_output = output_dir / f"{dataset_suffix}_feature_summary.xlsx"

    return input_file, output_dir, csv_output, xlsx_output


# =============================================================================
# Input validation and preparation
# =============================================================================

def detect_text_encoding(input_file: Path) -> str:
    """Detect the BOM used by Praat/Windows text output."""
    with input_file.open("rb") as handle:
        prefix = handle.read(4)

    if prefix.startswith(b"\xff\xfe") or prefix.startswith(b"\xfe\xff"):
        return "utf-16"
    if prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def load_feature_table(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(
            "Feature extraction table not found:\n"
            f"{input_file}\n\n"
            "Run _04_feature_extraction.praat for this subset first."
        )

    encoding = detect_text_encoding(input_file)

    # sep=None lets pandas detect comma, semicolon, or tab delimiters. Praat may
    # write UTF-16 text on Windows even when the file extension is .csv.
    table = pd.read_csv(
        input_file,
        encoding=encoding,
        sep=None,
        engine="python",
    )

    # Remove a possible BOM or surrounding whitespace from column names.
    table.columns = [
        str(column).lstrip("\\ufeff").strip()
        for column in table.columns
    ]

    required_columns = {CLASS_COLUMN, *FEATURE_COLUMNS}
    missing_columns = sorted(required_columns - set(table.columns))

    if missing_columns:
        raise ValueError(
            "The feature table is missing required columns: "
            + ", ".join(missing_columns)
        )

    table[CLASS_COLUMN] = (
        table[CLASS_COLUMN]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace(CLASS_ALIASES)
    )

    for feature in FEATURE_COLUMNS:
        table[feature] = pd.to_numeric(table[feature], errors="coerce")

    table = table[
        table[CLASS_COLUMN].notna()
        & (table[CLASS_COLUMN] != "")
    ].copy()

    return table


# =============================================================================
# Statistical analysis
# =============================================================================

def calculate_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Calculate count, percentage, P10, median, and P90 per class."""
    total_rows = len(table)
    rows: list[dict[str, object]] = []

    unexpected_classes = sorted(
        value
        for value in table[CLASS_COLUMN].dropna().astype(str).unique()
        if value not in CLASS_ORDER
    )
    class_names = [*CLASS_ORDER, *unexpected_classes]

    for class_name in class_names:
        class_rows = table[table[CLASS_COLUMN] == class_name]
        count = len(class_rows)

        row: dict[str, object] = {
            "fricative_type": class_name,
            "token_count": count,
            "percentage": (count / total_rows * 100.0) if total_rows else 0.0,
        }

        for feature in FEATURE_COLUMNS:
            values = class_rows[feature].dropna()
            row[f"{feature}_p10"] = (
                values.quantile(0.10, interpolation="linear")
                if not values.empty else pd.NA
            )
            row[f"{feature}_median"] = (
                values.median() if not values.empty else pd.NA
            )
            row[f"{feature}_p90"] = (
                values.quantile(0.90, interpolation="linear")
                if not values.empty else pd.NA
            )

        rows.append(row)

    total_row: dict[str, object] = {
        "fricative_type": "TOTAL",
        "token_count": total_rows,
        "percentage": 100.0 if total_rows else 0.0,
    }
    for feature in FEATURE_COLUMNS:
        total_row[f"{feature}_p10"] = pd.NA
        total_row[f"{feature}_median"] = pd.NA
        total_row[f"{feature}_p90"] = pd.NA
    rows.append(total_row)

    summary = pd.DataFrame(rows)
    numeric_columns = [
        column
        for column in summary.columns
        if column not in {"fricative_type", "token_count"}
    ]
    summary[numeric_columns] = summary[numeric_columns].round(ROUND_DIGITS)
    return summary


# =============================================================================
# Output
# =============================================================================

def write_xlsx(summary: pd.DataFrame, xlsx_output: Path) -> None:
    try:
        from openpyxl.styles import Alignment, Font
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError(
            "XLSX output requires openpyxl. Install it with:\n"
            "python -m pip install openpyxl"
        ) from exc

    with pd.ExcelWriter(xlsx_output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Feature Summary", index=False)

        worksheet = writer.book["Feature Summary"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")

        for column_cells in worksheet.columns:
            max_length = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column_cells
            )
            column_letter = get_column_letter(column_cells[0].column)
            worksheet.column_dimensions[column_letter].width = min(
                max(max_length + 2, 12),
                28,
            )


def write_log(
    log_file: Path,
    dataset_suffix: str,
    input_file: Path,
    output_dir: Path,
    table: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    lines = [
        "Feature analysis",
        "================",
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset subset: {dataset_suffix}",
        f"Input file: {input_file}",
        f"Output directory: {output_dir}",
        "",
        f"Input rows: {len(table)}",
        f"Fricative types in summary: {len(summary[summary['fricative_type'] != 'TOTAL'])}",
        "",
        "Class counts:",
    ]

    for _, row in summary.iterrows():
        if row["fricative_type"] == "TOTAL":
            continue
        lines.append(
            f"  {row['fricative_type']}: {int(row['token_count'])}"
        )

    lines.extend(
        [
            "",
            "Calculated statistics:",
            "  P10",
            "  Median",
            "  P90",
            "",
            "Place features:",
            *[f"  {feature}" for feature in PLACE_FEATURE_COLUMNS],
            "",
            "Voicing features:",
            *[f"  {feature}" for feature in VOICING_FEATURE_COLUMNS],
        ]
    )

    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    dataset_suffix = validate_dataset_suffix(DATASET_SUFFIX)
    input_file, output_dir, csv_output, xlsx_output = get_paths(
        dataset_suffix
    )
    log_file = output_dir / f"{dataset_suffix}_feature_analysis_log.txt"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Feature analysis")
    print("================")
    print(f"Dataset: {dataset_suffix}")
    print(f"Input:   {input_file}")
    print(f"Output:  {output_dir}")

    detected_encoding = detect_text_encoding(input_file)
    print(f"Encoding: {detected_encoding}")

    table = load_feature_table(input_file)
    summary = calculate_summary(table)

    summary.to_csv(
        csv_output,
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )

    if CREATE_XLSX:
        write_xlsx(summary, xlsx_output)

    write_log(
        log_file=log_file,
        dataset_suffix=dataset_suffix,
        input_file=input_file,
        output_dir=output_dir,
        table=table,
        summary=summary,
    )

    print(f"Input rows processed: {len(table)}")
    print(f"Summary CSV: {csv_output}")
    if CREATE_XLSX:
        print(f"Summary XLSX: {xlsx_output}")
    print(f"Log: {log_file}")
    print("Done.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
