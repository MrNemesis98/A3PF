# _00_config.praat
# =====================================================================================================================
# Copyright © 2026 Till Preidt (GitHub: MrNemesis98).
# This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License
# (CC BY-NC 4.0). Reuse and adaptation are permitted for non-commercial purposes,
# provided appropriate credit is given.
# ======================================================================================================================
#
# =============================================================================
# Central path configuration
#
# This file mirrors _00_config.py as closely as Praat permits.
#
# Location:
#     <project_root>/src/_00_config.praat
#
# IMPORTANT:
# Praat cannot reliably derive the project root from the included config file.
# Adjust ONLY projectRoot$ when the repository is moved to another location.
# Use forward slashes in the path.
# =============================================================================


# -----------------------------------------------------------------------------
# Project roots
# -----------------------------------------------------------------------------

projectRoot$ = "<project_root>"   ; enter the your root to the project here!

srcDir$ = projectRoot$ + "/src"
dataDir$ = projectRoot$ + "/data"
envDir$ = projectRoot$ + "/env"
modelsDir$ = projectRoot$ + "/models"
animationDir$ = projectRoot$ + "/anim"
textualDraftsDir$ = projectRoot$ + "/textual_drafts"


# -----------------------------------------------------------------------------
# Runtime environments and reusable models
# -----------------------------------------------------------------------------

mfaEnvDir$ = projectRoot$ + "/env_mfa"
mfaEnvPythonExe$ = mfaEnvDir$ + "/python.exe"
mfaEnvScriptsDir$ = mfaEnvDir$ + "/Scripts"
mfaEnvLibraryBinDir$ = mfaEnvDir$ + "/Library/bin"
mfaEnvMfaExe$ = mfaEnvScriptsDir$ + "/mfa.exe"

preprocessingModelsDir$ = modelsDir$ + "/00___preprocessing"
fasterWhisperModelsDir$ = preprocessingModelsDir$ + "/faster_whisper"

# MFA keeps pretrained models, configuration files, logs, and temporary working
# data below this project-local root.
mfaModelsDir$ = preprocessingModelsDir$ + "/montreal_forced_aligner"
mfaRootDir$ = mfaModelsDir$
mfaPretrainedModelsDir$ = mfaRootDir$ + "/pretrained_models"
mfaAcousticModelsDir$ = mfaPretrainedModelsDir$ + "/acoustic"
mfaDictionaryModelsDir$ = mfaPretrainedModelsDir$ + "/dictionary"
mfaAcousticModelName$ = "polish_mfa"
mfaDictionaryName$ = "polish_mfa"

lrModelsDir$ = modelsDir$ + "/01___LR"
ldaModelsDir$ = modelsDir$ + "/02___LDA"
qdaModelsDir$ = modelsDir$ + "/03___QDA"
rfModelsDir$ = modelsDir$ + "/04___RF"
svmModelsDir$ = modelsDir$ + "/05___SVM"


# -----------------------------------------------------------------------------
# Main data areas
# -----------------------------------------------------------------------------

preprocessingDataDir$ = dataDir$ + "/A___preprocessing"
statisticsDataDir$ = dataDir$ + "/B___statistics"
classificationDataDir$ = dataDir$ + "/C___classification"


# -----------------------------------------------------------------------------
# A — Preprocessing data paths
# -----------------------------------------------------------------------------

dataCollectionsDir$ = preprocessingDataDir$ + "/00___data_collections"

rawInputDir$ = preprocessingDataDir$ + "/01___raw_input"
rawInputTrainingDir$ = rawInputDir$ + "/training"
rawInputValidationDir$ = rawInputDir$ + "/validation"
rawInputTestDir$ = rawInputDir$ + "/test"

preparedWhisperInputDir$ = preprocessingDataDir$ + "/02___prepared_whisper_input"
preparedWhisperTrainingDir$ = preparedWhisperInputDir$ + "/training"
preparedWhisperValidationDir$ = preparedWhisperInputDir$ + "/validation"
preparedWhisperTestDir$ = preparedWhisperInputDir$ + "/test"

whisperOutputDir$ = preprocessingDataDir$ + "/03___whisper_output"

whisperDebugDir$ = whisperOutputDir$ + "/debug"
whisperDebugTrainingDir$ = whisperDebugDir$ + "/training"
whisperDebugValidationDir$ = whisperDebugDir$ + "/validation"
whisperDebugTestDir$ = whisperDebugDir$ + "/test"

mfaInputDir$ = whisperOutputDir$ + "/input_for_mfa"
mfaInputTrainingDir$ = mfaInputDir$ + "/training"
mfaInputValidationDir$ = mfaInputDir$ + "/validation"
mfaInputTestDir$ = mfaInputDir$ + "/test"

mfaOutputDir$ = preprocessingDataDir$ + "/04___mfa_output"

mfaDebugDir$ = mfaOutputDir$ + "/debug"
mfaDebugTrainingDir$ = mfaDebugDir$ + "/training"
mfaDebugValidationDir$ = mfaDebugDir$ + "/validation"
mfaDebugTestDir$ = mfaDebugDir$ + "/test"

featureExtractionInputDir$ = mfaOutputDir$ + "/input_for_feature_extraction"
featureExtractionInputTrainingDir$ = featureExtractionInputDir$ + "/training"
featureExtractionInputValidationDir$ = featureExtractionInputDir$ + "/validation"
featureExtractionInputTestDir$ = featureExtractionInputDir$ + "/test"

featureExtractionOutputDir$ = preprocessingDataDir$ + "/05___feature_extraction_output"
featureExtractionOutputTrainingDir$ = featureExtractionOutputDir$ + "/training"
featureExtractionOutputValidationDir$ = featureExtractionOutputDir$ + "/validation"
featureExtractionOutputTestDir$ = featureExtractionOutputDir$ + "/test"


# -----------------------------------------------------------------------------
# B — Statistics data paths
# -----------------------------------------------------------------------------

corpusOverviewOutputDir$ = statisticsDataDir$ + "/00___corpus_overview_output"
corpusOverviewTrainingDir$ = corpusOverviewOutputDir$ + "/training"
corpusOverviewValidationDir$ = corpusOverviewOutputDir$ + "/validation"
corpusOverviewTestDir$ = corpusOverviewOutputDir$ + "/test"

featureAnalysisOutputDir$ = statisticsDataDir$ + "/01___feature_analysis_output"
featureAnalysisTrainingDir$ = featureAnalysisOutputDir$ + "/training"
featureAnalysisValidationDir$ = featureAnalysisOutputDir$ + "/validation"
featureAnalysisTestDir$ = featureAnalysisOutputDir$ + "/test"

featurePatternVisualizationOutputDir$ = statisticsDataDir$ + "/02___feature_pattern_visualization_output"
featurePatternVisualizationTrainingDir$ = featurePatternVisualizationOutputDir$ + "/training"
featurePatternVisualizationValidationDir$ = featurePatternVisualizationOutputDir$ + "/validation"
featurePatternVisualizationTestDir$ = featurePatternVisualizationOutputDir$ + "/test"

# British-spelling compatibility aliases.
featurePatternVisualisationOutputDir$ = featurePatternVisualizationOutputDir$
featurePatternVisualisationTrainingDir$ = featurePatternVisualizationTrainingDir$
featurePatternVisualisationValidationDir$ = featurePatternVisualizationValidationDir$
featurePatternVisualisationTestDir$ = featurePatternVisualizationTestDir$


# -----------------------------------------------------------------------------
# C — Classification data paths
# -----------------------------------------------------------------------------

lrClassificationDir$ = classificationDataDir$ + "/01___LR"
lrValidationDir$ = lrClassificationDir$ + "/validation"
lrTestDir$ = lrClassificationDir$ + "/test"

ldaClassificationDir$ = classificationDataDir$ + "/02___LDA"
ldaValidationDir$ = ldaClassificationDir$ + "/validation"
ldaTestDir$ = ldaClassificationDir$ + "/test"

qdaClassificationDir$ = classificationDataDir$ + "/03___QDA"
qdaValidationDir$ = qdaClassificationDir$ + "/validation"
qdaTestDir$ = qdaClassificationDir$ + "/test"

rfClassificationDir$ = classificationDataDir$ + "/04___RF"
rfValidationDir$ = rfClassificationDir$ + "/validation"
rfTestDir$ = rfClassificationDir$ + "/test"

svmClassificationDir$ = classificationDataDir$ + "/05___SVM"
svmValidationDir$ = svmClassificationDir$ + "/validation"
svmTestDir$ = svmClassificationDir$ + "/test"

classificationOutputDir$ = classificationDataDir$


# -----------------------------------------------------------------------------
# Script directories
# -----------------------------------------------------------------------------

preprocessingScriptsDir$ = srcDir$ + "/A___preprocessing_scripts"
statisticsScriptsDir$ = srcDir$ + "/B___statistics_scripts"
classificationScriptsDir$ = srcDir$ + "/C___classification_scripts"
temporaryBatScriptsDir$ = srcDir$ + "/X___temporal_bat_scripts"


# -----------------------------------------------------------------------------
# Script paths
# -----------------------------------------------------------------------------

pythonConfigScript$ = srcDir$ + "/_00_config.py"
praatConfigScript$ = srcDir$ + "/_00_config.praat"
testScript$ = srcDir$ + "/_99_test.py"

inputPreparationScript$ = preprocessingScriptsDir$ + "/_01_input_preparation.py"
whisperSegmenterScript$ = preprocessingScriptsDir$ + "/_02_whisper_segmenter.py"
mfaBatchManagerScript$ = preprocessingScriptsDir$ + "/_03_mfa_batch_manager.py"
featureExtractionScript$ = preprocessingScriptsDir$ + "/_04_feature_extraction.praat"

corpusOverviewScript$ = statisticsScriptsDir$ + "/_00_corpus_overview.py"
featureAnalysisScript$ = statisticsScriptsDir$ + "/_01_feature_analysis.py"
featurePatternVisualizationScript$ = statisticsScriptsDir$ + "/_02_feature_pattern_visualization.py"

# British-spelling compatibility alias.
featurePatternVisualisationScript$ = featurePatternVisualizationScript$
