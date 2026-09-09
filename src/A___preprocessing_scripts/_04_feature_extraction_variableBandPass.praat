clearinfo

# =====================================================================================================================
# _04_feature_extraction_variableBandPass.praat
# Token-level place and voicing feature extraction for classifier training and evaluation
# =====================================================================================================================
# Copyright © 2026 Till Preidt (GitHub: MrNemesis98).
# This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License
# (CC BY-NC 4.0). Reuse and adaptation are permitted for non-commercial purposes,
# provided appropriate credit is given.
# ======================================================================================================================
#
# The script reads every TextGrid in one corpus subset and measures the corresponding WAV file.
# It writes one CSV row per eligible fricative token. Place and voicing features are measured
# within the same token loop; no sampling or class balancing is applied.
#
# Before each run, set corpusSubset$ to exactly one of:
#     "training"
#     "validation"
#     "test"
#
# Input:
#     WAV:      data/A___preprocessing/03___whisper_output/input_for_mfa/<subset>/*.wav
#     TextGrid: data/A___preprocessing/04___mfa_output/input_for_feature_extraction/<subset>/*.TextGrid
#
# Output:
#     data/A___preprocessing/05___feature_extraction_output/<subset>/<subset>_features.csv
#     data/A___preprocessing/05___feature_extraction_output/<subset>/<subset>_feature_extraction_log.txt
#
# IMPORTANT:
# TextGrids and WAV files intentionally remain in separate directories to avoid permanent audio duplication.
# Files are matched by identical base names. During processing, temporary copies named
# current.TextGrid/current.wav are created to avoid the Windows/Praat path-length limit.
# The originals are never renamed or modified; the temporary copies are deleted immediately afterward.
# =====================================================================================================================


# Central path configuration -------------------------------------------------------------------------------------------

include ../_00_config.praat


# Parameters -----------------------------------------------------------------------------------------------------------

corpusSubset$ = "test"   ; allowed values: training, validation, test

phoneTierName$ = "phones"

# Place-feature settings: spectral moments are measured in the central
# portion of the fricative and restricted to the high-frequency noise band.
cogPower = 1.0
placeEdgeTrimProportion = 0.25
minimumSegmentDuration = 0.03
placeBandPassLowHz = 3000   ; here

# Experimental variable-band-pass version:
# Only the place-feature lower cutoff is varied between runs.
# Input subsets remain training/validation/test.
# Output is written to <subset>_<placeBandPassLowHz>, e.g. training_4000.

placeBandPassHighHz = 11000
placeBandPassSmoothingHz = 100

# Voicing-feature settings: periodicity and low-frequency energy are measured
# over the complete MFA interval. F0 itself is not exported as a predictor.
# Raw cross-correlation is used for periodicity detection because Praat
# recommends it for voice analysis and short time windows. At a 75-Hz floor,
# its analysis window is about 13.3 ms instead of the 40-ms window used by
# raw autocorrelation, so accepted fricatives of at least 30 ms can be analysed
# without raising the pitch floor token by token.
voicingEdgeTrimProportion = 0.0
voicingPitchTimeStep = 0.0
voicingPitchFloorHz = 75
voicingPitchMaximumCandidates = 15
voicingPitchVeryAccurate$ = "no"
voicingPitchSilenceThreshold = 0.03
voicingPitchThreshold = 0.55
voicingPitchOctaveCost = 0.01
voicingPitchOctaveJumpCost = 0.35
voicingPitchVoicedUnvoicedCost = 0.14
voicingPitchCeilingHz = 400

lowFrequencyUpperHz = 900
lowFrequencyBandSmoothingHz = 50

harmonicityTimeStep = 0.01
harmonicityPitchFloorHz = 75
harmonicitySilenceThreshold = 0.10
harmonicityPeriodsPerWindow = 1.0
harmonicityUndefinedFloorDb = -200


# Validate analysis parameters -----------------------------------------------------------------------------------------

if voicingPitchFloorHz < 75
    exitScript: "voicingPitchFloorHz must be at least 75 Hz."
endif

minimumPitchAnalysisDuration = 1 / voicingPitchFloorHz
if minimumSegmentDuration < minimumPitchAnalysisDuration
    exitScript: "minimumSegmentDuration is too short for raw cross-correlation at the configured pitch floor. Increase minimumSegmentDuration or voicingPitchFloorHz."
endif

if voicingPitchCeilingHz <= voicingPitchFloorHz
    exitScript: "voicingPitchCeilingHz must be greater than voicingPitchFloorHz."
endif


# Validate subset and construct fixed project paths --------------------------------------------------------------------

subsetIsValid = 0
if corpusSubset$ = "training"
    subsetIsValid = 1
endif
if corpusSubset$ = "validation"
    subsetIsValid = 1
endif
if corpusSubset$ = "test"
    subsetIsValid = 1
endif

if subsetIsValid = 0
    exitScript: "Invalid corpusSubset$: ", corpusSubset$, newline$, "Use training, validation, or test."
endif

textGridInputDir$ = featureExtractionInputDir$ + "/" + corpusSubset$
audioInputDir$ = mfaInputDir$ + "/" + corpusSubset$
experimentSuffix$ = corpusSubset$ + "_" + string$ (placeBandPassLowHz)
outputDir$ = featureExtractionOutputDir$ + "/" + experimentSuffix$

featureTableFile$ = outputDir$ + "/" + corpusSubset$ + "_features.csv"
logFile$ = outputDir$ + "/" + corpusSubset$ + "_feature_extraction_log.txt"

# Praat on Windows may fail to open very long paths. Each matched TextGrid/WAV
# pair is therefore copied temporarily to these short names and processed.
# The original files remain untouched.
currentTextGridPath$ = textGridInputDir$ + "/current.TextGrid"
currentSoundPath$ = audioInputDir$ + "/current.wav"

# Praat has no portable folderReadable() function. The TextGrid directory is
# validated below by creating a file list; missing WAV files are checked per item.
runSystem: "cmd /c if not exist """ + outputDir$ + """ mkdir """ + outputDir$ + """"


# Prepare output files -------------------------------------------------------------------------------------------------

writeFileLine: featureTableFile$, "file_name,interval_index,start_time_s,end_time_s,duration_s,mfa_ground_truth,place_label,voicing_label,combined_label,cog_hz,spread_hz,skewness,kurtosis,voiced_fraction,low_total_intensity_ratio_db,mean_hnr_db"

writeFileLine: logFile$, "====================================="
appendFileLine: logFile$, "Token-level feature extraction"
appendFileLine: logFile$, "====================================="
appendFileLine: logFile$, ""
appendFileLine: logFile$, "Corpus subset: " + corpusSubset$
appendFileLine: logFile$, "Variable-band-pass experiment folder: " + experimentSuffix$
appendFileLine: logFile$, "Place band-pass lower cutoff (Hz): " + string$ (placeBandPassLowHz)
appendFileLine: logFile$, "TextGrid input directory: " + textGridInputDir$
appendFileLine: logFile$, "Audio input directory: " + audioInputDir$
appendFileLine: logFile$, "Output table: " + featureTableFile$
appendFileLine: logFile$, ""
appendFileLine: logFile$, "Measurement settings:"
appendFileLine: logFile$, "phoneTierName=" + phoneTierName$
appendFileLine: logFile$, "Place features:"
appendFileLine: logFile$, "cogPower=" + string$ (cogPower)
appendFileLine: logFile$, "placeEdgeTrimProportion=" + string$ (placeEdgeTrimProportion)
appendFileLine: logFile$, "minimumSegmentDuration=" + string$ (minimumSegmentDuration)
appendFileLine: logFile$, "placeBandPassLowHz=" + string$ (placeBandPassLowHz)
appendFileLine: logFile$, "placeBandPassHighHz=" + string$ (placeBandPassHighHz) + " (automatically lowered when required by Nyquist)"
appendFileLine: logFile$, "placeBandPassSmoothingHz=" + string$ (placeBandPassSmoothingHz)
appendFileLine: logFile$, ""
appendFileLine: logFile$, "Voicing features:"
appendFileLine: logFile$, "voicingEdgeTrimProportion=" + string$ (voicingEdgeTrimProportion)
appendFileLine: logFile$, "voicingPitchMethod=raw cross-correlation (legacy-compatible To Pitch (cc))"
appendFileLine: logFile$, "voicingPitchTimeStep=" + string$ (voicingPitchTimeStep)
appendFileLine: logFile$, "voicingPitchFloorHz=" + string$ (voicingPitchFloorHz)
appendFileLine: logFile$, "voicingPitchCeilingHz=" + string$ (voicingPitchCeilingHz)
appendFileLine: logFile$, "voicingPitchThreshold=" + string$ (voicingPitchThreshold)
appendFileLine: logFile$, "lowFrequencyUpperHz=" + string$ (lowFrequencyUpperHz)
appendFileLine: logFile$, "lowFrequencyBandSmoothingHz=" + string$ (lowFrequencyBandSmoothingHz)
appendFileLine: logFile$, "harmonicityMethod=cross-correlation"
appendFileLine: logFile$, "harmonicityTimeStep=" + string$ (harmonicityTimeStep)
appendFileLine: logFile$, "harmonicityPitchFloorHz=" + string$ (harmonicityPitchFloorHz)
appendFileLine: logFile$, "harmonicitySilenceThreshold=" + string$ (harmonicitySilenceThreshold)
appendFileLine: logFile$, "harmonicityPeriodsPerWindow=" + string$ (harmonicityPeriodsPerWindow)
appendFileLine: logFile$, "harmonicityUndefinedFloorDb=" + string$ (harmonicityUndefinedFloorDb)
appendFileLine: logFile$, ""


# Counters -------------------------------------------------------------------------------------------------------------

totalTextGrids = 0
totalMeasuredTokens = 0
totalShortSegments = 0
totalMissingWavs = 0
totalMissingPhoneTiers = 0
totalInvalidPlaceBandFiles = 0

totalDentalVoiceless = 0
totalDentalVoiced = 0
totalRetroflexVoiceless = 0
totalRetroflexVoiced = 0
totalAlveopalatalVoiceless = 0
totalAlveopalatalVoiced = 0


# Create TextGrid file list --------------------------------------------------------------------------------------------

Create Strings as file list: "classification_textgrid_files", textGridInputDir$ + "/*.TextGrid"
numberOfTextGrids = Get number of strings
totalTextGrids = numberOfTextGrids

appendFileLine: logFile$, "Number of TextGrids found: " + string$ (numberOfTextGrids)
appendFileLine: logFile$, ""

if numberOfTextGrids = 0
    selectObject: "Strings classification_textgrid_files"
    Remove
    exitScript: "No TextGrid files found in: ", textGridInputDir$
endif


# Remove harmless temporary copies left by an interrupted previous run -----------------------------------------------

if fileReadable (currentTextGridPath$)
    deleteFile: currentTextGridPath$
endif

if fileReadable (currentSoundPath$)
    deleteFile: currentSoundPath$
endif


# Process files --------------------------------------------------------------------------------------------------------

for fileIndex from 1 to numberOfTextGrids

    selectObject: "Strings classification_textgrid_files"
    textGridFileName$ = Get string: fileIndex

    baseName$ = replace$ (textGridFileName$, ".TextGrid", "", 0)
    originalTextGridPath$ = textGridInputDir$ + "/" + textGridFileName$
    originalSoundFileName$ = baseName$ + ".wav"
    originalSoundPath$ = audioInputDir$ + "/" + originalSoundFileName$

    appendFileLine: logFile$, "Processing [" + string$ (fileIndex) + "/" + string$ (numberOfTextGrids) + "]: " + textGridFileName$

    # Create disposable short-name copies. runSystem_nocheck prevents one failed
    # copy operation from aborting the complete corpus run; success is verified
    # immediately with fileReadable().
    sourceSoundWindows$ = replace$ (originalSoundPath$, "/", "\", 0)
    currentSoundWindows$ = replace$ (currentSoundPath$, "/", "\", 0)
    sourceTextGridWindows$ = replace$ (originalTextGridPath$, "/", "\", 0)
    currentTextGridWindows$ = replace$ (currentTextGridPath$, "/", "\", 0)

    sourceSoundExtended$ = "\\?\" + sourceSoundWindows$
    currentSoundExtended$ = "\\?\" + currentSoundWindows$
    sourceTextGridExtended$ = "\\?\" + sourceTextGridWindows$
    currentTextGridExtended$ = "\\?\" + currentTextGridWindows$

    sourceSoundPS$ = replace$ (sourceSoundExtended$, "'", "''", 0)
    currentSoundPS$ = replace$ (currentSoundExtended$, "'", "''", 0)
    sourceTextGridPS$ = replace$ (sourceTextGridExtended$, "'", "''", 0)
    currentTextGridPS$ = replace$ (currentTextGridExtended$, "'", "''", 0)

    runSystem_nocheck: "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ""[System.IO.File]::Copy('" + sourceSoundPS$ + "','" + currentSoundPS$ + "', $true)"""

    if fileReadable (currentSoundPath$)

        runSystem_nocheck: "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ""[System.IO.File]::Copy('" + sourceTextGridPS$ + "','" + currentTextGridPS$ + "', $true)"""

        if not fileReadable (currentTextGridPath$)
            deleteFile: currentSoundPath$
            appendFileLine: logFile$, "  ERROR: TextGrid could not be copied to current.TextGrid."
            selectObject: "Strings classification_textgrid_files"
            Remove
            exitScript: "Could not prepare temporary TextGrid copy: ", originalTextGridPath$
        endif

        Read from file: currentSoundPath$
        soundObjectName$ = selected$ ("Sound")

        selectObject: "Sound " + soundObjectName$
        samplingFrequency = Get sampling frequency
        nyquistFrequency = samplingFrequency / 2
        placeBandPassHighForFile = placeBandPassHighHz

        if placeBandPassHighForFile >= nyquistFrequency
            placeBandPassHighForFile = nyquistFrequency - placeBandPassSmoothingHz
            appendFileLine: logFile$, "  Upper cutoff adjusted to " + fixed$ (placeBandPassHighForFile, 3) + " Hz."
        endif

        if placeBandPassHighForFile > placeBandPassLowHz

            Read from file: currentTextGridPath$
            textGridObjectName$ = selected$ ("TextGrid")

            tierNumber = 0
            selectObject: "TextGrid " + textGridObjectName$
            numberOfTiers = Get number of tiers

            for tierIndex from 1 to numberOfTiers
                selectObject: "TextGrid " + textGridObjectName$
                currentTierName$ = Get tier name: tierIndex
                if currentTierName$ = phoneTierName$
                    tierNumber = tierIndex
                endif
            endfor

            if tierNumber > 0

                selectObject: "TextGrid " + textGridObjectName$
                numberOfIntervals = Get number of intervals: tierNumber

                for intervalIndex from 1 to numberOfIntervals

                    selectObject: "TextGrid " + textGridObjectName$
                    mfaGroundTruth$ = Get label of interval: tierNumber, intervalIndex

                    isTarget = 0
                    placeLabel$ = ""
                    voicingLabel$ = ""
                    combinedLabel$ = ""

                    if mfaGroundTruth$ = "s" or mfaGroundTruth$ = "s̪"
                        isTarget = 1
                        placeLabel$ = "dental"
                        voicingLabel$ = "voiceless"
                        combinedLabel$ = "dental_voiceless"
                    endif

                    if mfaGroundTruth$ = "z" or mfaGroundTruth$ = "z̪"
                        isTarget = 1
                        placeLabel$ = "dental"
                        voicingLabel$ = "voiced"
                        combinedLabel$ = "dental_voiced"
                    endif

                    if mfaGroundTruth$ = "ʂ"
                        isTarget = 1
                        placeLabel$ = "retroflex"
                        voicingLabel$ = "voiceless"
                        combinedLabel$ = "retroflex_voiceless"
                    endif

                    if mfaGroundTruth$ = "ʐ"
                        isTarget = 1
                        placeLabel$ = "retroflex"
                        voicingLabel$ = "voiced"
                        combinedLabel$ = "retroflex_voiced"
                    endif

                    if mfaGroundTruth$ = "ɕ"
                        isTarget = 1
                        placeLabel$ = "alveopalatal"
                        voicingLabel$ = "voiceless"
                        combinedLabel$ = "alveopalatal_voiceless"
                    endif

                    if mfaGroundTruth$ = "ʑ"
                        isTarget = 1
                        placeLabel$ = "alveopalatal"
                        voicingLabel$ = "voiced"
                        combinedLabel$ = "alveopalatal_voiced"
                    endif

                    if isTarget = 1

                        selectObject: "TextGrid " + textGridObjectName$
                        startTime = Get start time of interval: tierNumber, intervalIndex
                        endTime = Get end time of interval: tierNumber, intervalIndex
                        duration = endTime - startTime

                        if duration >= minimumSegmentDuration


                            # PLACE FEATURES --------------------------------------------------------------
                            # Measure spectral moments in the central 50% after isolating
                            # the high-frequency frication-noise band.
                            placeAnalysisStart = startTime + duration * placeEdgeTrimProportion
                            placeAnalysisEnd = endTime - duration * placeEdgeTrimProportion

                            selectObject: "Sound " + soundObjectName$
                            Extract part: placeAnalysisStart, placeAnalysisEnd, "Hamming", 1.0, "no"
                            placeSoundName$ = selected$ ("Sound")

                            Filter (pass Hann band): placeBandPassLowHz, placeBandPassHighForFile, placeBandPassSmoothingHz
                            placeFilteredSoundName$ = selected$ ("Sound")

                            To Spectrum: "yes"
                            placeSpectrumName$ = selected$ ("Spectrum")

                            cog = Get centre of gravity: cogPower
                            spectralSpread = Get standard deviation: cogPower
                            spectralSkewness = Get skewness: cogPower
                            spectralKurtosis = Get kurtosis: cogPower

                            selectObject: "Spectrum " + placeSpectrumName$
                            Remove
                            selectObject: "Sound " + placeFilteredSoundName$
                            Remove
                            selectObject: "Sound " + placeSoundName$
                            Remove

                            # VOICING FEATURES ------------------------------------------------------------
                            # Use the complete MFA interval by default. The configurable trim
                            # remains available if boundary contamination is found during validation.
                            voicingAnalysisStart = startTime + duration * voicingEdgeTrimProportion
                            voicingAnalysisEnd = endTime - duration * voicingEdgeTrimProportion

                            selectObject: "Sound " + soundObjectName$
                            Extract part: voicingAnalysisStart, voicingAnalysisEnd, "rectangular", 1.0, "no"
                            voicingSoundName$ = selected$ ("Sound")

                            # Low-frequency-to-total intensity ratio in dB. RMS amplitudes are
                            # converted with 20*log10, which is equivalent to an energy ratio in dB.
                            totalRms = Get root-mean-square: 0, 0

                            Filter (pass Hann band): 0, lowFrequencyUpperHz, lowFrequencyBandSmoothingHz
                            lowFrequencySoundName$ = selected$ ("Sound")
                            lowFrequencyRms = Get root-mean-square: 0, 0

                            if totalRms > 0 and lowFrequencyRms > 0
                                lowTotalIntensityRatioDb = 20 * log10 (lowFrequencyRms / totalRms)
                            else
                                lowTotalIntensityRatioDb = harmonicityUndefinedFloorDb
                            endif

                            selectObject: "Sound " + lowFrequencySoundName$
                            Remove

                            # Proportion of Pitch frames for which periodic vibration is detected.
                            # Raw cross-correlation uses a one-period window and is therefore
                            # appropriate for short fricative intervals. The numeric F0 values
                            # themselves are deliberately not exported.
                            selectObject: "Sound " + voicingSoundName$
                            To Pitch (cc): voicingPitchTimeStep, voicingPitchFloorHz, voicingPitchMaximumCandidates, voicingPitchVeryAccurate$, voicingPitchSilenceThreshold, voicingPitchThreshold, voicingPitchOctaveCost, voicingPitchOctaveJumpCost, voicingPitchVoicedUnvoicedCost, voicingPitchCeilingHz
                            pitchObjectName$ = selected$ ("Pitch")

                            numberOfPitchFrames = Get number of frames
                            numberOfVoicedFrames = 0

                            for pitchFrameIndex from 1 to numberOfPitchFrames
                                pitchValue = Get value in frame: pitchFrameIndex, "Hertz"
                                if pitchValue <> undefined
                                    numberOfVoicedFrames = numberOfVoicedFrames + 1
                                endif
                            endfor

                            if numberOfPitchFrames > 0
                                voicedFraction = numberOfVoicedFrames / numberOfPitchFrames
                            else
                                voicedFraction = 0
                            endif

                            selectObject: "Pitch " + pitchObjectName$
                            Remove

                            # Mean harmonics-to-noise ratio. Undefined values are mapped to a
                            # documented floor so voiceless tokens are retained rather than dropped.
                            selectObject: "Sound " + voicingSoundName$
                            To Harmonicity (cc): harmonicityTimeStep, harmonicityPitchFloorHz, harmonicitySilenceThreshold, harmonicityPeriodsPerWindow
                            harmonicityObjectName$ = selected$ ("Harmonicity")
                            meanHnrDb = Get mean: 0, 0

                            if meanHnrDb = undefined
                                meanHnrDb = harmonicityUndefinedFloorDb
                            endif

                            selectObject: "Harmonicity " + harmonicityObjectName$
                            Remove
                            selectObject: "Sound " + voicingSoundName$
                            Remove

                            appendFileLine: featureTableFile$, textGridFileName$ + "," + string$ (intervalIndex) + "," + fixed$ (startTime, 6) + "," + fixed$ (endTime, 6) + "," + fixed$ (duration, 6) + "," + mfaGroundTruth$ + "," + placeLabel$ + "," + voicingLabel$ + "," + combinedLabel$ + "," + fixed$ (cog, 3) + "," + fixed$ (spectralSpread, 3) + "," + fixed$ (spectralSkewness, 6) + "," + fixed$ (spectralKurtosis, 6) + "," + fixed$ (voicedFraction, 6) + "," + fixed$ (lowTotalIntensityRatioDb, 6) + "," + fixed$ (meanHnrDb, 6)

                            totalMeasuredTokens = totalMeasuredTokens + 1

                            if combinedLabel$ = "dental_voiceless"
                                totalDentalVoiceless = totalDentalVoiceless + 1
                            endif
                            if combinedLabel$ = "dental_voiced"
                                totalDentalVoiced = totalDentalVoiced + 1
                            endif
                            if combinedLabel$ = "retroflex_voiceless"
                                totalRetroflexVoiceless = totalRetroflexVoiceless + 1
                            endif
                            if combinedLabel$ = "retroflex_voiced"
                                totalRetroflexVoiced = totalRetroflexVoiced + 1
                            endif
                            if combinedLabel$ = "alveopalatal_voiceless"
                                totalAlveopalatalVoiceless = totalAlveopalatalVoiceless + 1
                            endif
                            if combinedLabel$ = "alveopalatal_voiced"
                                totalAlveopalatalVoiced = totalAlveopalatalVoiced + 1
                            endif

                        else
                            totalShortSegments = totalShortSegments + 1
                            appendFileLine: logFile$, "  Skipped short target interval " + string$ (intervalIndex) + " | label=" + mfaGroundTruth$ + " | duration=" + fixed$ (duration, 6)
                        endif

                    endif

                endfor

            else
                totalMissingPhoneTiers = totalMissingPhoneTiers + 1
                appendFileLine: logFile$, "  WARNING: Tier '" + phoneTierName$ + "' not found."
            endif

            selectObject: "TextGrid " + textGridObjectName$
            Remove

        else
            totalInvalidPlaceBandFiles = totalInvalidPlaceBandFiles + 1
            appendFileLine: logFile$, "  WARNING: Sampling frequency too low for configured place-feature band-pass range."
        endif

        selectObject: "Sound " + soundObjectName$
        Remove

        # Delete only the disposable copies. The original files were never changed.
        if fileReadable (currentTextGridPath$)
            deleteFile: currentTextGridPath$
        endif
        if fileReadable (currentSoundPath$)
            deleteFile: currentSoundPath$
        endif

    else
        totalMissingWavs = totalMissingWavs + 1
        appendFileLine: logFile$, "  WARNING: Corresponding WAV file could not be copied or was not found: " + originalSoundPath$
    endif

endfor


# Clean up file list ---------------------------------------------------------------------------------------------------

selectObject: "Strings classification_textgrid_files"
Remove


# Final log summary ----------------------------------------------------------------------------------------------------

appendFileLine: logFile$, ""
appendFileLine: logFile$, "====================================="
appendFileLine: logFile$, "Extraction summary"
appendFileLine: logFile$, "====================================="
appendFileLine: logFile$, "Subset: " + corpusSubset$
appendFileLine: logFile$, "TextGrids found: " + string$ (totalTextGrids)
appendFileLine: logFile$, "Measured target tokens: " + string$ (totalMeasuredTokens)
appendFileLine: logFile$, "Skipped target intervals shorter than minimum: " + string$ (totalShortSegments)
appendFileLine: logFile$, "Missing WAV files: " + string$ (totalMissingWavs)
appendFileLine: logFile$, "Missing phone tiers: " + string$ (totalMissingPhoneTiers)
appendFileLine: logFile$, "Files with unusable place-feature band-pass range: " + string$ (totalInvalidPlaceBandFiles)
appendFileLine: logFile$, ""
appendFileLine: logFile$, "Class counts:"
appendFileLine: logFile$, "dental_voiceless=" + string$ (totalDentalVoiceless)
appendFileLine: logFile$, "dental_voiced=" + string$ (totalDentalVoiced)
appendFileLine: logFile$, "retroflex_voiceless=" + string$ (totalRetroflexVoiceless)
appendFileLine: logFile$, "retroflex_voiced=" + string$ (totalRetroflexVoiced)
appendFileLine: logFile$, "alveopalatal_voiceless=" + string$ (totalAlveopalatalVoiceless)
appendFileLine: logFile$, "alveopalatal_voiced=" + string$ (totalAlveopalatalVoiced)
appendFileLine: logFile$, ""
appendFileLine: logFile$, "Feature table written to: " + featureTableFile$

appendInfoLine: "Feature extraction completed."
appendInfoLine: "Subset: ", corpusSubset$
appendInfoLine: "Experiment folder: ", experimentSuffix$
appendInfoLine: "Place band-pass low cutoff: ", placeBandPassLowHz, " Hz"
appendInfoLine: "TextGrids found: ", numberOfTextGrids
appendInfoLine: "Measured tokens: ", totalMeasuredTokens
appendInfoLine: "Output: ", featureTableFile$
appendInfoLine: "Log: ", logFile$
