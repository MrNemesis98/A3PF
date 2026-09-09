"""
_00_config.py
===============================

Copyright © 2026 Till Preidt (GitHub: MrNemesis98).
This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License
(CC BY-NC 4.0). Reuse and adaptation are permitted for non-commercial purposes,
provided appropriate credit is given.

Central path configuration.

This module contains shared project, data, output, and script paths only.
Processing parameters remain in the individual pipeline scripts.

Location:
    <project_root>/src/_00_config.py
"""

from __future__ import annotations

from pathlib import Path


# =============================================================================
# Project roots
# =============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
ENV_DIR = PROJECT_ROOT / "env"
MODELS_DIR = PROJECT_ROOT / "models"
ANIMATION_DIR = PROJECT_ROOT / "anim"
TEXTUAL_DRAFTS_DIR = PROJECT_ROOT / "textual_drafts"


# =============================================================================
# Runtime environments and reusable models
# =============================================================================

MFA_ENV_DIR = PROJECT_ROOT / "env_mfa"
MFA_ENV_PYTHON_EXE = MFA_ENV_DIR / "python.exe"
MFA_ENV_SCRIPTS_DIR = MFA_ENV_DIR / "Scripts"
MFA_ENV_LIBRARY_BIN_DIR = MFA_ENV_DIR / "Library" / "bin"
MFA_ENV_MFA_EXE = MFA_ENV_SCRIPTS_DIR / "mfa.exe"

PREPROCESSING_MODELS_DIR = MODELS_DIR / "00___preprocessing"
FASTER_WHISPER_MODELS_DIR = PREPROCESSING_MODELS_DIR / "faster_whisper"

# MFA stores downloaded pretrained models, configuration files, logs, and
# temporary working data below MFA_ROOT_DIR. The symbolic model names are kept
# centrally as well so all MFA callers use the same resources.
MFA_MODELS_DIR = PREPROCESSING_MODELS_DIR / "montreal_forced_aligner"
MFA_ROOT_DIR = MFA_MODELS_DIR
MFA_PRETRAINED_MODELS_DIR = MFA_ROOT_DIR / "pretrained_models"
MFA_ACOUSTIC_MODELS_DIR = MFA_PRETRAINED_MODELS_DIR / "acoustic"
MFA_DICTIONARY_MODELS_DIR = MFA_PRETRAINED_MODELS_DIR / "dictionary"
MFA_ACOUSTIC_MODEL_NAME = "polish_mfa"
MFA_DICTIONARY_NAME = "polish_mfa"

LR_MODELS_DIR = MODELS_DIR / "01___LR"
LDA_MODELS_DIR = MODELS_DIR / "02___LDA"
QDA_MODELS_DIR = MODELS_DIR / "03___QDA"
RF_MODELS_DIR = MODELS_DIR / "04___RF"
SVM_MODELS_DIR = MODELS_DIR / "05___SVM"


# =============================================================================
# Main data areas
# =============================================================================

PREPROCESSING_DATA_DIR = DATA_DIR / "A___preprocessing"
STATISTICS_DATA_DIR = DATA_DIR / "B___statistics"
CLASSIFICATION_DATA_DIR = DATA_DIR / "C___classification"


# =============================================================================
# Corpus subsets
# =============================================================================

DATASET_SUFFIXES = ("training", "validation", "test")


# =============================================================================
# A — Preprocessing directories
# =============================================================================

DATA_COLLECTIONS_DIR = PREPROCESSING_DATA_DIR / "00___data_collections"

RAW_INPUT_DIR = PREPROCESSING_DATA_DIR / "01___raw_input"
RAW_INPUT_TRAINING_DIR = RAW_INPUT_DIR / "training"
RAW_INPUT_VALIDATION_DIR = RAW_INPUT_DIR / "validation"
RAW_INPUT_TEST_DIR = RAW_INPUT_DIR / "test"

PREPARED_WHISPER_INPUT_DIR = (
    PREPROCESSING_DATA_DIR / "02___prepared_whisper_input"
)
PREPARED_WHISPER_TRAINING_DIR = PREPARED_WHISPER_INPUT_DIR / "training"
PREPARED_WHISPER_VALIDATION_DIR = PREPARED_WHISPER_INPUT_DIR / "validation"
PREPARED_WHISPER_TEST_DIR = PREPARED_WHISPER_INPUT_DIR / "test"

WHISPER_OUTPUT_DIR = PREPROCESSING_DATA_DIR / "03___whisper_output"

WHISPER_DEBUG_DIR = WHISPER_OUTPUT_DIR / "debug"
WHISPER_DEBUG_TRAINING_DIR = WHISPER_DEBUG_DIR / "training"
WHISPER_DEBUG_VALIDATION_DIR = WHISPER_DEBUG_DIR / "validation"
WHISPER_DEBUG_TEST_DIR = WHISPER_DEBUG_DIR / "test"

MFA_INPUT_DIR = WHISPER_OUTPUT_DIR / "input_for_mfa"
MFA_INPUT_TRAINING_DIR = MFA_INPUT_DIR / "training"
MFA_INPUT_VALIDATION_DIR = MFA_INPUT_DIR / "validation"
MFA_INPUT_TEST_DIR = MFA_INPUT_DIR / "test"

MFA_OUTPUT_DIR = PREPROCESSING_DATA_DIR / "04___mfa_output"

MFA_DEBUG_DIR = MFA_OUTPUT_DIR / "debug"
MFA_DEBUG_TRAINING_DIR = MFA_DEBUG_DIR / "training"
MFA_DEBUG_VALIDATION_DIR = MFA_DEBUG_DIR / "validation"
MFA_DEBUG_TEST_DIR = MFA_DEBUG_DIR / "test"

FEATURE_EXTRACTION_INPUT_DIR = (
    MFA_OUTPUT_DIR / "input_for_feature_extraction"
)
FEATURE_EXTRACTION_INPUT_TRAINING_DIR = (
    FEATURE_EXTRACTION_INPUT_DIR / "training"
)
FEATURE_EXTRACTION_INPUT_VALIDATION_DIR = (
    FEATURE_EXTRACTION_INPUT_DIR / "validation"
)
FEATURE_EXTRACTION_INPUT_TEST_DIR = (
    FEATURE_EXTRACTION_INPUT_DIR / "test"
)

FEATURE_EXTRACTION_OUTPUT_DIR = (
    PREPROCESSING_DATA_DIR / "05___feature_extraction_output"
)
FEATURE_EXTRACTION_OUTPUT_TRAINING_DIR = (
    FEATURE_EXTRACTION_OUTPUT_DIR / "training"
)
FEATURE_EXTRACTION_OUTPUT_VALIDATION_DIR = (
    FEATURE_EXTRACTION_OUTPUT_DIR / "validation"
)
FEATURE_EXTRACTION_OUTPUT_TEST_DIR = (
    FEATURE_EXTRACTION_OUTPUT_DIR / "test"
)


# =============================================================================
# B — Statistics directories
# =============================================================================

CORPUS_OVERVIEW_OUTPUT_DIR = (
    STATISTICS_DATA_DIR / "00___corpus_overview_output"
)
CORPUS_OVERVIEW_TRAINING_DIR = CORPUS_OVERVIEW_OUTPUT_DIR / "training"
CORPUS_OVERVIEW_VALIDATION_DIR = CORPUS_OVERVIEW_OUTPUT_DIR / "validation"
CORPUS_OVERVIEW_TEST_DIR = CORPUS_OVERVIEW_OUTPUT_DIR / "test"

FEATURE_ANALYSIS_OUTPUT_DIR = (
    STATISTICS_DATA_DIR / "01___feature_analysis_output"
)
FEATURE_ANALYSIS_TRAINING_DIR = FEATURE_ANALYSIS_OUTPUT_DIR / "training"
FEATURE_ANALYSIS_VALIDATION_DIR = FEATURE_ANALYSIS_OUTPUT_DIR / "validation"
FEATURE_ANALYSIS_TEST_DIR = FEATURE_ANALYSIS_OUTPUT_DIR / "test"

FEATURE_PATTERN_VISUALIZATION_OUTPUT_DIR = (
    STATISTICS_DATA_DIR / "02___feature_pattern_visualization_output"
)
FEATURE_PATTERN_VISUALIZATION_TRAINING_DIR = (
    FEATURE_PATTERN_VISUALIZATION_OUTPUT_DIR / "training"
)
FEATURE_PATTERN_VISUALIZATION_VALIDATION_DIR = (
    FEATURE_PATTERN_VISUALIZATION_OUTPUT_DIR / "validation"
)
FEATURE_PATTERN_VISUALIZATION_TEST_DIR = (
    FEATURE_PATTERN_VISUALIZATION_OUTPUT_DIR / "test"
)

# British-spelling compatibility aliases.
FEATURE_PATTERN_VISUALISATION_OUTPUT_DIR = (
    FEATURE_PATTERN_VISUALIZATION_OUTPUT_DIR
)
FEATURE_PATTERN_VISUALISATION_TRAINING_DIR = (
    FEATURE_PATTERN_VISUALIZATION_TRAINING_DIR
)
FEATURE_PATTERN_VISUALISATION_VALIDATION_DIR = (
    FEATURE_PATTERN_VISUALIZATION_VALIDATION_DIR
)
FEATURE_PATTERN_VISUALISATION_TEST_DIR = (
    FEATURE_PATTERN_VISUALIZATION_TEST_DIR
)


# =============================================================================
# C — Classification directories
# =============================================================================

LR_CLASSIFICATION_DIR = CLASSIFICATION_DATA_DIR / "01___LR"
LR_VALIDATION_DIR = LR_CLASSIFICATION_DIR / "validation"
LR_TEST_DIR = LR_CLASSIFICATION_DIR / "test"

LDA_CLASSIFICATION_DIR = CLASSIFICATION_DATA_DIR / "02___LDA"
LDA_VALIDATION_DIR = LDA_CLASSIFICATION_DIR / "validation"
LDA_TEST_DIR = LDA_CLASSIFICATION_DIR / "test"

QDA_CLASSIFICATION_DIR = CLASSIFICATION_DATA_DIR / "03___QDA"
QDA_VALIDATION_DIR = QDA_CLASSIFICATION_DIR / "validation"
QDA_TEST_DIR = QDA_CLASSIFICATION_DIR / "test"

RF_CLASSIFICATION_DIR = CLASSIFICATION_DATA_DIR / "04___RF"
RF_VALIDATION_DIR = RF_CLASSIFICATION_DIR / "validation"
RF_TEST_DIR = RF_CLASSIFICATION_DIR / "test"

SVM_CLASSIFICATION_DIR = CLASSIFICATION_DATA_DIR / "05___SVM"
SVM_VALIDATION_DIR = SVM_CLASSIFICATION_DIR / "validation"
SVM_TEST_DIR = SVM_CLASSIFICATION_DIR / "test"

CLASSIFICATION_OUTPUT_DIR = CLASSIFICATION_DATA_DIR


# =============================================================================
# Script directories
# =============================================================================

PREPROCESSING_SCRIPTS_DIR = SRC_DIR / "A___preprocessing_scripts"
STATISTICS_SCRIPTS_DIR = SRC_DIR / "B___statistics_scripts"
CLASSIFICATION_SCRIPTS_DIR = SRC_DIR / "C___classification_scripts"
TEMPORARY_BAT_SCRIPTS_DIR = SRC_DIR / "X___temporal_bat_scripts"


# =============================================================================
# Script files
# =============================================================================

PYTHON_CONFIG_FILE = SRC_DIR / "_00_config.py"
PRAAT_CONFIG_FILE = SRC_DIR / "_00_config.praat"
TEST_SCRIPT = SRC_DIR / "_99_test.py"

INPUT_PREPARATION_SCRIPT = (
    PREPROCESSING_SCRIPTS_DIR / "_01_input_preparation.py"
)
WHISPER_SEGMENTER_SCRIPT = (
    PREPROCESSING_SCRIPTS_DIR / "_02_whisper_segmenter.py"
)
MFA_BATCH_MANAGER_SCRIPT = (
    PREPROCESSING_SCRIPTS_DIR / "_03_mfa_batch_manager.py"
)
FEATURE_EXTRACTION_SCRIPT = (
    PREPROCESSING_SCRIPTS_DIR / "_04_feature_extraction.praat"
)

CORPUS_OVERVIEW_SCRIPT = (
    STATISTICS_SCRIPTS_DIR / "_00_corpus_overview.py"
)
FEATURE_ANALYSIS_SCRIPT = (
    STATISTICS_SCRIPTS_DIR / "_01_feature_analysis.py"
)
FEATURE_PATTERN_VISUALIZATION_SCRIPT = (
    STATISTICS_SCRIPTS_DIR / "_02_feature_pattern_visualization.py"
)

# British-spelling compatibility alias.
FEATURE_PATTERN_VISUALISATION_SCRIPT = (
    FEATURE_PATTERN_VISUALIZATION_SCRIPT
)


# =============================================================================
# Dataset helpers
# =============================================================================

def validate_dataset_suffix(dataset_suffix: str) -> str:
    """Return a normalized corpus subset or raise ValueError."""
    normalized = dataset_suffix.strip().lower()
    if normalized not in DATASET_SUFFIXES:
        allowed = ", ".join(DATASET_SUFFIXES)
        raise ValueError(
            f"Invalid dataset suffix: {dataset_suffix!r}. "
            f"Expected one of: {allowed}."
        )
    return normalized


def dataset_directory(base_directory: Path, dataset_suffix: str) -> Path:
    """Return a validated subset directory below a configured base path."""
    return base_directory / validate_dataset_suffix(dataset_suffix)


# =============================================================================
# Directory creation
# =============================================================================

CORE_OUTPUT_DIRECTORIES = (
    FASTER_WHISPER_MODELS_DIR,
    MFA_MODELS_DIR,
    MFA_PRETRAINED_MODELS_DIR,
    MFA_ACOUSTIC_MODELS_DIR,
    MFA_DICTIONARY_MODELS_DIR,
    LR_MODELS_DIR,
    LDA_MODELS_DIR,
    QDA_MODELS_DIR,
    RF_MODELS_DIR,
    SVM_MODELS_DIR,
    RAW_INPUT_TRAINING_DIR,
    RAW_INPUT_VALIDATION_DIR,
    RAW_INPUT_TEST_DIR,
    PREPARED_WHISPER_TRAINING_DIR,
    PREPARED_WHISPER_VALIDATION_DIR,
    PREPARED_WHISPER_TEST_DIR,
    WHISPER_DEBUG_TRAINING_DIR,
    WHISPER_DEBUG_VALIDATION_DIR,
    WHISPER_DEBUG_TEST_DIR,
    MFA_INPUT_TRAINING_DIR,
    MFA_INPUT_VALIDATION_DIR,
    MFA_INPUT_TEST_DIR,
    MFA_DEBUG_TRAINING_DIR,
    MFA_DEBUG_VALIDATION_DIR,
    MFA_DEBUG_TEST_DIR,
    FEATURE_EXTRACTION_INPUT_TRAINING_DIR,
    FEATURE_EXTRACTION_INPUT_VALIDATION_DIR,
    FEATURE_EXTRACTION_INPUT_TEST_DIR,
    FEATURE_EXTRACTION_OUTPUT_TRAINING_DIR,
    FEATURE_EXTRACTION_OUTPUT_VALIDATION_DIR,
    FEATURE_EXTRACTION_OUTPUT_TEST_DIR,
    CORPUS_OVERVIEW_TRAINING_DIR,
    CORPUS_OVERVIEW_VALIDATION_DIR,
    CORPUS_OVERVIEW_TEST_DIR,
    FEATURE_ANALYSIS_TRAINING_DIR,
    FEATURE_ANALYSIS_VALIDATION_DIR,
    FEATURE_ANALYSIS_TEST_DIR,
    FEATURE_PATTERN_VISUALIZATION_TRAINING_DIR,
    FEATURE_PATTERN_VISUALIZATION_VALIDATION_DIR,
    FEATURE_PATTERN_VISUALIZATION_TEST_DIR,
    LR_VALIDATION_DIR,
    LR_TEST_DIR,
    LDA_VALIDATION_DIR,
    LDA_TEST_DIR,
    QDA_VALIDATION_DIR,
    QDA_TEST_DIR,
    RF_VALIDATION_DIR,
    RF_TEST_DIR,
    SVM_VALIDATION_DIR,
    SVM_TEST_DIR,
    TEMPORARY_BAT_SCRIPTS_DIR,
)


def create_core_output_directories() -> None:
    """Create all standard pipeline leaf directories when necessary."""
    for directory in CORE_OUTPUT_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
