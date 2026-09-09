"""
_02_feature_pattern_visualization_variableBandPass.py
===============================

Copyright © 2026 Till Preidt (GitHub: MrNemesis98).
This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License
(CC BY-NC 4.0). Reuse and adaptation are permitted for non-commercial purposes,
provided appropriate credit is given.


Input
-----
data/A___preprocessing/05___feature_extraction_output/<subset>/
    <subset>_features.csv

Output
------
data/B___statistics/02___feature_pattern_visualization_output/<subset>/
    place/
        <subset>_place_feature_patterns_<stat>_overview.png/.svg
        <subset>_place_feature_pattern_<stat>_<place>.png/.svg
    voicing/
        <subset>_voicing_distribution_<feature>.png/.svg
        <subset>_voicing_distribution_<feature>_by_place.png/.svg

Place is visualized with radar plots based on CoG, spread, skewness, and
kurtosis. Voicing is visualized with token-level distributions for duration,
voiced fraction, low/total intensity ratio, and mean HNR.

The voicing plots use the raw token table because distributions cannot be
reconstructed from P10/median/P90 summary values. Plot ranges for unbounded
features are limited to P1-P99 for readability; values outside that range are
not removed from the data and their number is reported in the console.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Load central project configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent

for config_search_dir in (SCRIPT_DIR, SRC_DIR):
    if str(config_search_dir) not in sys.path:
        sys.path.insert(0, str(config_search_dir))

import _00_config as config  # noqa: E402


# =============================================================================
# Script-specific settings
# =============================================================================

DATASET_SUFFIX = "validation"  # allowed: training, validation, test
PLACE_BANDPASS_LOW_HZ = 4000  # must match the feature-extraction experiment

PLACE_FEATURES = (
    ("cog_hz", "CoG", "Hz"),
    ("spread_hz", "Spread", "Hz"),
    ("skewness", "Skewness", ""),
    ("kurtosis", "Kurtosis", ""),
)

VOICING_FEATURES = (
    ("duration_s", "Duration", "s"),
    ("voiced_fraction", "Voiced fraction", ""),
    ("low_total_intensity_ratio_db", "Low/total intensity ratio", "dB"),
    ("mean_hnr_db", "Mean HNR", "dB"),
)

PLACE_ORDER = ("dental", "retroflex", "alveolo-palatal")
VOICING_ORDER = ("voiceless", "voiced")

PLACE_LABELS = {
    "dental": "Dental",
    "retroflex": "Retroflex",
    "alveolo-palatal": "Alveolo-palatal",
}

PLACE_FILE_STEMS = {
    "dental": "dental",
    "retroflex": "retroflex",
    "alveolo-palatal": "alveolo_palatal",
}

DEFAULT_VALUE_STAT = "median"
LOWER_REFERENCE_STAT = "p10"
UPPER_REFERENCE_STAT = "p90"
DISTRIBUTION_LOWER_QUANTILE = 0.01
DISTRIBUTION_UPPER_QUANTILE = 0.99
HISTOGRAM_BINS = 60
DEFAULT_DPI = 300


# =============================================================================
# Paths and input
# =============================================================================

def validate_dataset_suffix(dataset_suffix: str) -> str:
    try:
        return config.validate_dataset_suffix(dataset_suffix)
    except ValueError as exc:
        raise ValueError(
            f"Invalid DATASET_SUFFIX: {dataset_suffix!r}. "
            "Use 'training', 'validation', or 'test'."
        ) from exc


def experiment_folder_name(dataset_suffix: str) -> str:
    """Return the temporary experimental folder name, e.g. validation_4000."""
    return f"{dataset_suffix}_{PLACE_BANDPASS_LOW_HZ}"


def get_default_paths(dataset_suffix: str) -> tuple[Path, Path]:
    experiment_folder = experiment_folder_name(dataset_suffix)

    input_file = (
        config.FEATURE_EXTRACTION_OUTPUT_DIR
        / experiment_folder
        / f"{dataset_suffix}_features.csv"
    )

    output_dir = (
        config.FEATURE_PATTERN_VISUALIZATION_OUTPUT_DIR
        / experiment_folder
    )

    return input_file, output_dir


def detect_text_encoding(input_file: Path) -> str:
    with input_file.open("rb") as handle:
        prefix = handle.read(4)
    if prefix.startswith(b"\xff\xfe") or prefix.startswith(b"\xfe\xff"):
        return "utf-16"
    if prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8"


def load_token_table(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(
            "Feature extraction table not found:\n"
            f"{input_file}\n\n"
            "Run _04_feature_extraction_variableBandPass.praat with the same "
            "PLACE_BANDPASS_LOW_HZ first."
        )

    encoding = detect_text_encoding(input_file)
    table = pd.read_csv(
        input_file,
        encoding=encoding,
        sep=None,
        engine="python",
    )
    table.columns = [
        str(column).lstrip("\ufeff").strip()
        for column in table.columns
    ]

    feature_columns = [key for key, _, _ in (*PLACE_FEATURES, *VOICING_FEATURES)]
    required_columns = {"place_label", "voicing_label", *feature_columns}
    missing = sorted(required_columns - set(table.columns))
    if missing:
        raise ValueError(
            "The token table is missing required columns: "
            + ", ".join(missing)
        )

    table["place_label"] = (
        table["place_label"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace(
            {
                "alveopalatal": "alveolo-palatal",
                "alveolo_palatal": "alveolo-palatal",
                "alveolo palatal": "alveolo-palatal",
            }
        )
    )
    table["voicing_label"] = (
        table["voicing_label"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    for feature in feature_columns:
        table[feature] = pd.to_numeric(table[feature], errors="coerce")

    table = table[
        table["place_label"].isin(PLACE_ORDER)
        & table["voicing_label"].isin(VOICING_ORDER)
    ].copy()

    if table.empty:
        raise ValueError("No valid rows remain after label validation.")

    return table


# =============================================================================
# Place summary and normalization
# =============================================================================

def calculate_place_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for place in PLACE_ORDER:
        subset = table[table["place_label"] == place]
        row = {
            "place_label": place,
            "PlotLabel": PLACE_LABELS[place],
            "FileStem": PLACE_FILE_STEMS[place],
            "token_count": len(subset),
        }
        for key, _, _ in PLACE_FEATURES:
            values = pd.to_numeric(subset[key], errors="coerce")
            values = values[np.isfinite(values)]
            row[f"{key}_p10"] = values.quantile(0.10) if not values.empty else np.nan
            row[f"{key}_median"] = values.median() if not values.empty else np.nan
            row[f"{key}_p90"] = values.quantile(0.90) if not values.empty else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    needed = [
        f"{key}_{stat}"
        for key, _, _ in PLACE_FEATURES
        for stat in ("p10", "median", "p90")
    ]
    summary = summary.dropna(subset=needed).copy()
    if summary.empty:
        raise ValueError("No complete place summaries could be calculated.")
    return summary


def add_place_normalization(
    summary: pd.DataFrame,
    value_stat: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = summary.copy()
    references = []

    for key, label, unit in PLACE_FEATURES:
        lower_col = f"{key}_{LOWER_REFERENCE_STAT}"
        value_col = f"{key}_{value_stat}"
        upper_col = f"{key}_{UPPER_REFERENCE_STAT}"

        lower_idx = out[lower_col].astype(float).idxmin()
        upper_idx = out[upper_col].astype(float).idxmax()
        lower = float(out.loc[lower_idx, lower_col])
        upper = float(out.loc[upper_idx, upper_col])
        width = upper - lower

        if math.isclose(width, 0.0):
            out[f"norm_{key}"] = 50.0
        else:
            out[f"norm_{key}"] = (
                (out[value_col].astype(float) - lower) / width * 100.0
            ).clip(0.0, 100.0)

        references.append(
            {
                "Feature": label,
                "Unit": unit or "dimensionless",
                "0 % value": lower,
                "0 % place": out.loc[lower_idx, "PlotLabel"],
                "100 % value": upper,
                "100 % place": out.loc[upper_idx, "PlotLabel"],
            }
        )

    return out, pd.DataFrame(references)


def display_stat_name(value_stat: str) -> str:
    return {"p10": "P10", "median": "Median", "p90": "P90"}[value_stat]


def radar_angles() -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, len(PLACE_FEATURES), endpoint=False)
    return np.concatenate([angles, angles[:1]])


def setup_radar_axis(ax: plt.Axes, title: str) -> None:
    angles = radar_angles()[:-1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([label for _, label, _ in PLACE_FEATURES], fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7)
    ax.set_rlabel_position(22.5)
    ax.tick_params(axis="x", pad=12)
    ax.grid(alpha=0.35)
    ax.set_title(title, fontsize=13, pad=20)


def draw_radar(ax: plt.Axes, row: pd.Series) -> None:
    values = [float(row[f"norm_{key}"]) for key, _, _ in PLACE_FEATURES]
    closed = np.array(values + values[:1])
    angles = radar_angles()
    ax.plot(angles, closed, linewidth=2)
    ax.fill(angles, closed, alpha=0.18)
    ax.scatter(angles[:-1], values, s=35, zorder=3)


def raw_place_values_table(ax: plt.Axes, row: pd.Series, value_stat: str) -> None:
    ax.axis("off")
    rows = []
    for key, label, unit in PLACE_FEATURES:
        value = float(row[f"{key}_{value_stat}"])
        if unit == "Hz":
            raw = f"{value:.0f} Hz"
        else:
            raw = f"{value:.3f}"
        rows.append([label, raw, f"{float(row[f'norm_{key}']):.0f} %"])

    table = ax.table(
        cellText=rows,
        colLabels=["Feature", display_stat_name(value_stat), "Relative"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        cell.set_linewidth(0.4)


def reference_table(ax: plt.Axes, references: pd.DataFrame) -> None:
    ax.axis("off")
    rows = []
    for _, row in references.iterrows():
        if row["Unit"] == "Hz":
            lo = f"{float(row['0 % value']):.0f}"
            hi = f"{float(row['100 % value']):.0f}"
        else:
            lo = f"{float(row['0 % value']):.3f}"
            hi = f"{float(row['100 % value']):.3f}"
        rows.append(
            [
                row["Feature"],
                row["Unit"],
                f"P10 = {lo}",
                row["0 % place"],
                f"P90 = {hi}",
                row["100 % place"],
            ]
        )

    ax.text(
        0.5,
        1.04,
        "Place normalization: 0 % = lowest place P10; 100 % = highest place P90",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    table = ax.table(
        cellText=rows,
        colLabels=["Feature", "Unit", "0 % reference", "Place", "100 % reference", "Place"],
        bbox=[0.0, 0.0, 1.0, 0.82],
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.1)
    for (r, _), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
        cell.set_linewidth(0.4)


def make_place_overview(
    summary: pd.DataFrame,
    references: pd.DataFrame,
    output_base: Path,
    dataset_suffix: str,
    value_stat: str,
    dpi: int,
) -> None:
    columns = len(summary)
    fig = plt.figure(figsize=(5.4 * columns, 8.0))
    grid = fig.add_gridspec(3, columns, height_ratios=[4.0, 1.25, 1.55], hspace=0.85, wspace=0.32)

    for column, (_, row) in enumerate(summary.iterrows()):
        ax_radar = fig.add_subplot(grid[0, column], projection="polar")
        setup_radar_axis(ax_radar, str(row["PlotLabel"]))
        draw_radar(ax_radar, row)
        ax_values = fig.add_subplot(grid[1, column])
        raw_place_values_table(ax_values, row, value_stat)

    ax_refs = fig.add_subplot(grid[2, :])
    reference_table(ax_refs, references)
    fig.suptitle(
        f"Place Feature Patterns ({display_stat_name(value_stat)})\n"
        f"{dataset_suffix.capitalize()} set | BandPassLowHz = {PLACE_BANDPASS_LOW_HZ} Hz\n"
        "Voiced and voiceless tokens pooled within each place",
        fontsize=15,
        y=0.995,
        linespacing=1.15,
    )
    fig.subplots_adjust(top=0.80)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def make_individual_place_pattern(
    row: pd.Series,
    references: pd.DataFrame,
    output_dir: Path,
    dataset_suffix: str,
    value_stat: str,
    dpi: int,
) -> None:
    output_base = output_dir / (
        f"{dataset_suffix}_place_feature_pattern_{value_stat}_{row['FileStem']}"
    )
    fig = plt.figure(figsize=(7.5, 9.5))
    grid = fig.add_gridspec(3, 1, height_ratios=[4.8, 1.45, 1.70], hspace=0.48)
    ax_radar = fig.add_subplot(grid[0, 0], projection="polar")
    setup_radar_axis(
        ax_radar,
        f"Place Feature Pattern: {row['PlotLabel']}\n"
        f"{dataset_suffix.capitalize()} set | BandPassLowHz = {PLACE_BANDPASS_LOW_HZ} Hz",
    )
    draw_radar(ax_radar, row)
    ax_values = fig.add_subplot(grid[1, 0])
    raw_place_values_table(ax_values, row, value_stat)
    ax_refs = fig.add_subplot(grid[2, 0])
    reference_table(ax_refs, references)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Voicing distributions
# =============================================================================

def finite_values(table: pd.DataFrame, feature_key: str) -> pd.DataFrame:
    result = table[["place_label", "voicing_label", feature_key]].copy()
    mask = np.isfinite(result[feature_key].to_numpy(dtype=float))
    return result.loc[mask].copy()


def display_limits(values: pd.Series, feature_key: str) -> tuple[float, float, int]:
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        raise ValueError(f"No finite values available for {feature_key}.")

    if feature_key == "voiced_fraction":
        outside = int(((values < 0.0) | (values > 1.0)).sum())
        return 0.0, 1.0, outside

    lower = float(values.quantile(DISTRIBUTION_LOWER_QUANTILE))
    upper = float(values.quantile(DISTRIBUTION_UPPER_QUANTILE))
    if math.isclose(lower, upper):
        pad = max(abs(lower) * 0.05, 0.01)
        lower -= pad
        upper += pad
    outside = int(((values < lower) | (values > upper)).sum())
    return lower, upper, outside


def make_voicing_distribution(
    table: pd.DataFrame,
    feature: tuple[str, str, str],
    output_dir: Path,
    dataset_suffix: str,
    dpi: int,
) -> None:
    key, label, unit = feature
    finite = finite_values(table, key)
    lower, upper, outside = display_limits(finite[key], key)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for voicing in VOICING_ORDER:
        values = finite.loc[finite["voicing_label"] == voicing, key].astype(float)
        if values.empty:
            continue
        weights = np.ones(len(values), dtype=float) / len(values)
        ax.hist(
            values,
            bins=HISTOGRAM_BINS,
            range=(lower, upper),
            weights=weights,
            histtype="step",
            linewidth=2,
            label=f"{voicing} (n={len(values):,})",
        )

    ax.set_xlabel(label if not unit else f"{label} ({unit})")
    ax.set_ylabel("Proportion of tokens per bin")
    ax.set_title(
        f"Voicing Distribution: {label}\n"
        f"{dataset_suffix.capitalize()} set | Place BandPassLowHz = {PLACE_BANDPASS_LOW_HZ} Hz"
    )
    ax.set_xlim(lower, upper)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    if key == "voiced_fraction":
        note = "Natural display range: 0-1"
    else:
        note = (
            f"Display range: P{int(DISTRIBUTION_LOWER_QUANTILE*100)}-"
            f"P{int(DISTRIBUTION_UPPER_QUANTILE*100)} of all valid tokens"
        )
    if outside:
        note += f"; {outside:,} values outside display range (not deleted)"
    fig.text(0.5, 0.01, note, ha="center", fontsize=8.5)

    output_base = output_dir / f"{dataset_suffix}_voicing_distribution_{key}"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def make_voicing_distribution_by_place(
    table: pd.DataFrame,
    feature: tuple[str, str, str],
    output_dir: Path,
    dataset_suffix: str,
    dpi: int,
) -> None:
    key, label, unit = feature
    finite = finite_values(table, key)
    lower, upper, outside = display_limits(finite[key], key)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    line_styles = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))
    style_index = 0

    for place in PLACE_ORDER:
        for voicing in VOICING_ORDER:
            values = finite.loc[
                (finite["place_label"] == place)
                & (finite["voicing_label"] == voicing),
                key,
            ].astype(float)
            if values.empty:
                style_index += 1
                continue

            hist, edges = np.histogram(values, bins=HISTOGRAM_BINS, range=(lower, upper))
            if hist.sum() == 0:
                style_index += 1
                continue
            proportions = hist.astype(float) / hist.sum()
            centers = (edges[:-1] + edges[1:]) / 2.0
            ax.plot(
                centers,
                proportions,
                linewidth=1.8,
                linestyle=line_styles[style_index],
                label=f"{PLACE_LABELS[place]} / {voicing} (n={len(values):,})",
            )
            style_index += 1

    ax.set_xlabel(label if not unit else f"{label} ({unit})")
    ax.set_ylabel("Proportion of tokens per bin")
    ax.set_title(
        f"Voicing Distribution by Place: {label}\n"
        f"{dataset_suffix.capitalize()} set | Place BandPassLowHz = {PLACE_BANDPASS_LOW_HZ} Hz"
    )
    ax.set_xlim(lower, upper)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5)

    if key == "voiced_fraction":
        note = "Natural display range: 0-1"
    else:
        note = (
            f"Display range: P{int(DISTRIBUTION_LOWER_QUANTILE*100)}-"
            f"P{int(DISTRIBUTION_UPPER_QUANTILE*100)} of all valid tokens"
        )
    if outside:
        note += f"; {outside:,} values outside display range (not deleted)"
    fig.text(0.5, 0.01, note, ha="center", fontsize=8.5)

    output_base = output_dir / f"{dataset_suffix}_voicing_distribution_{key}_by_place"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Command-line interface and execution
# =============================================================================

def parse_args(default_input: Path, default_output_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create separate place radar plots and voicing distribution plots "
            "from the token-level feature table."
        )
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument(
        "--value-stat",
        type=str.lower,
        choices=["p10", "median", "p90"],
        default=DEFAULT_VALUE_STAT,
        help="Statistic plotted in place radar plots. Default: median.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--no-individual-place",
        action="store_true",
        help="Create only the place overview, without individual place figures.",
    )
    parser.add_argument(
        "--no-by-place-voicing",
        action="store_true",
        help="Skip supplementary voicing distributions split by place.",
    )
    return parser.parse_args()


def main() -> None:
    dataset_suffix = validate_dataset_suffix(DATASET_SUFFIX)
    default_input, default_output_dir = get_default_paths(dataset_suffix)
    args = parse_args(default_input, default_output_dir)

    table = load_token_table(args.input)
    place_output_dir = args.output_dir / "place"
    voicing_output_dir = args.output_dir / "voicing"

    print("Feature-pattern visualization")
    print("=============================")
    print(f"Dataset subset: {dataset_suffix}")
    print(f"BandPassLowHz:  {PLACE_BANDPASS_LOW_HZ} Hz")
    print(f"Experiment:     {experiment_folder_name(dataset_suffix)}")
    print(f"Input:          {args.input}")
    print(f"Rows:           {len(table):,}")
    print(f"Place output:   {place_output_dir}")
    print(f"Voicing output: {voicing_output_dir}")

    # Place
    place_summary = calculate_place_summary(table)
    place_summary, references = add_place_normalization(place_summary, args.value_stat)
    place_overview = place_output_dir / (
        f"{dataset_suffix}_place_feature_patterns_{args.value_stat}_overview"
    )
    make_place_overview(
        place_summary,
        references,
        place_overview,
        dataset_suffix,
        args.value_stat,
        args.dpi,
    )

    if not args.no_individual_place:
        for _, row in place_summary.iterrows():
            make_individual_place_pattern(
                row,
                references,
                place_output_dir,
                dataset_suffix,
                args.value_stat,
                args.dpi,
            )

    # Voicing
    print("\nVoicing feature diagnostics:")
    for feature in VOICING_FEATURES:
        key = feature[0]
        finite = finite_values(table, key)
        missing = len(table) - len(finite)
        lower, upper, outside = display_limits(finite[key], key)
        print(
            f"  {key}: finite={len(finite):,}, missing/non-finite={missing:,}, "
            f"display=[{lower:.6g}, {upper:.6g}], outside display={outside:,}"
        )
        make_voicing_distribution(
            table,
            feature,
            voicing_output_dir,
            dataset_suffix,
            args.dpi,
        )
        if not args.no_by_place_voicing:
            make_voicing_distribution_by_place(
                table,
                feature,
                voicing_output_dir,
                dataset_suffix,
                args.dpi,
            )

    print("\nVisualization created successfully.")
    print(f"Place output:   {place_output_dir}")
    print(f"Voicing output: {voicing_output_dir}")


if __name__ == "__main__":
    main()
