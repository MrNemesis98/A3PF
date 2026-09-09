# -*- coding: utf-8 -*-
"""
_02_whisper_segmenter.py
===============================

Copyright © 2026 Till Preidt (GitHub: MrNemesis98).
This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License
(CC BY-NC 4.0). Reuse and adaptation are permitted for non-commercial purposes,
provided appropriate credit is given.

Purpose
-------
This script prepares long audiobook recordings and their corresponding transcript
files for Montreal Forced Aligner (MFA). It is designed for a master's thesis
pipeline in which long Polish audiobook chapters are first segmented into
shorter, MFA-friendly audio/text pairs and then aligned with MFA for later
phonetic analysis in Praat.

The main problem addressed by this script is that MFA performs poorly on very
long recordings when the audio contains additional material that is not present
in the transcript, such as LibriVox intros, chapter announcements, outro text,
long background-noise passages, or extended silence. This script uses automatic
speech recognition (ASR) only as a preprocessing aid. The final text files
exported for MFA are based on the original input transcript, not on the ASR
transcript. This keeps the scientific analysis grounded in the intended
reference text while using ASR timestamps to make the audio and text easier for
MFA to process.

Core workflow
-------------
1. The user selects an input folder with a Windows folder dialog.
2. The script searches for matching audio/text pairs:
      chapter_01.wav + chapter_01.txt
      chapter_02.mp3 + chapter_02.txt
3. Each audio file is transcribed with faster-whisper using word-level timestamps.
4. The concatenated Whisper word sequence is aligned with the original transcript.
5. Larger unmatched Whisper-only regions are treated as possible extra audio
   material, for example intros or outros.
6. The script trims the recording to the time range that corresponds to the
   original transcript.
7. The original transcript is segmented on word boundaries. The algorithm
   prefers acoustic pauses between neighboring Whisper-timed words.
8. Each final segment is kept close to the target duration while never exceeding
   the maximum duration unless an emergency fallback is required.
9. Each final segment is exported as:
      <chapter>_seg_0001.wav
      <chapter>_seg_0001.txt
   and is ready to be used as MFA input.

Important methodological note
-----------------------------
Whisper/faster-whisper is not used as the final phonetic alignment tool. The ASR
output is used only to estimate approximate word times and to remove audio
material that is not part of the transcript. The exported MFA text snippets are
taken from the original text file. Phoneme-level alignment should still be
performed with MFA afterwards.

Installation requirements
-------------------------
Recommended installation inside the project's Python environment:

    python -m pip cache purge
    python -m pip install --no-cache-dir --timeout 120 faster-whisper pydub rapidfuzz unidecode

FFmpeg must also be available on the system because pydub uses FFmpeg to load
and export MP3/WAV files. If MP3 loading fails, install FFmpeg and make sure it
is available in the system PATH.

CUDA/GPU note
-------------
The current stable settings for this project are:

    WHISPER_MODEL_SIZE = "base"
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"

A compatible NVIDIA GPU may be tested later with:

    WHISPER_DEVICE = "cuda"
    WHISPER_COMPUTE_TYPE = "float16"

Expected input structure
------------------------
The selected input folder should contain one or more audio/text pairs with the
same base filename:

    chapter_01.wav
    chapter_01.txt
    chapter_02.mp3
    chapter_02.txt

Expected output structure
-------------------------
The script creates an output folder next to the selected input folder:

    <input_folder>_MFA_segmented_v10/
        trimmed/
            chapter_01_trimmed.wav
            chapter_01_trimmed.txt
        mfa_segments/
            chapter_01_seg_0001.wav
            chapter_01_seg_0001.txt
        cog_analysis_input/
            chapter_01_seg_0001.wav
        logs/
            chapter_01_whisper_words.csv
            chapter_01_alignment_matches.csv
            chapter_01_segment_log.csv
            run_summary.txt

Limitations
-----------
This script is intentionally conservative. It avoids cutting the audio because
of isolated word mismatches between Whisper and the original transcript. For the
final MFA snippets, it cuts only between original-text tokens and prefers longer
pauses between Whisper-timed words. This reduces the risk of cutting through
individual sound realizations. The automatic alignment is approximate and should
be checked on a small subset before processing a complete corpus.

Prepared as part of a Praat/MFA/Python preprocessing pipeline for fricative
articulation analysis.
"""

from __future__ import annotations

import os

import winsound

# Prevent OpenMP runtime conflicts observed on Windows when faster-whisper,
# NumPy, MFA-related libraries, or Intel runtimes are installed together.
# These variables must be set before importing faster_whisper/ctranslate2.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "12"

import csv
import difflib
import re
import traceback
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Load the central project configuration from the parent src directory.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import _00_config as config

from pydub import AudioSegment, silence
from rapidfuzz import fuzz

try:
    from faster_whisper import WhisperModel
except ImportError as exc:
    raise SystemExit(
        "ERROR: faster-whisper is not installed.\n"
        "Install it with:\n"
        "python -m pip install faster-whisper pydub rapidfuzz unidecode\n"
    ) from exc


# ---------------------------------------------------------------------------
# User-adjustable parameters
# ---------------------------------------------------------------------------

# Select the dataset split processed by this run:
# "training", "validation", or "test"
DATASET_SUFFIX = "test"

WHISPER_MODEL_SIZE = "base"    # Stable current setting for this project. Later: test "small" or "medium".
WHISPER_MODEL_DIRECTORY = config.FASTER_WHISPER_MODELS_DIR
WHISPER_DEVICE = "cpu"          # Current stable setting: "cpu". Later: "cuda" may be tested again.
WHISPER_COMPUTE_TYPE = "int8"   # int8 for cpu, float16 for gpu.

# Set this to True only after the model was downloaded successfully once.
# It prevents the script from silently trying to access Hugging Face again.
LOCAL_MODEL_FILES_ONLY = False
WHISPER_LANGUAGE = "pl"

MIN_SEGMENT_SEC = 30.0
TARGET_SEGMENT_SEC = 45.0
MAX_SEGMENT_SEC = 60.0

# Pause-aware word-level segmentation settings.
# The algorithm prefers cuts in clearly audible/recognized pauses between words.
MIN_PREFERRED_WORD_PAUSE_SEC = 0.30
PAUSE_SCORE_WEIGHT = 4.0
TARGET_SCORE_WEIGHT = 1.0

SEGMENT_PADDING_MS = 150
TRIM_PADDING_MS = 300

# Internal non-reference cleaning. These settings are deliberately conservative:
# they remove long Whisper-only passages such as LibriVox intros, repeated
# chapter announcements, comments, or longer speech-like background material,
# but they do not remove isolated ASR word errors.
MIN_INTERNAL_REMOVAL_WORDS = 6
MIN_INTERNAL_REMOVAL_SEC = 3.0
MIN_EDGE_REMOVAL_WORDS = 2
MIN_EDGE_REMOVAL_SEC = 0.8
REMOVAL_KEEP_PADDING_SEC = 0.12

MIN_SILENCE_LEN_MS = 1200
SILENCE_THRESH_RELATIVE_DB = 16
KEEP_SILENCE_MS = 250

MIN_MATCH_RATIO_FOR_ANCHOR = 70

# Safety gates for large-scale batch processing.
# If the original transcript and Whisper transcript do not match well enough,
# the pair is skipped instead of exporting misleading MFA snippets.
MIN_MATCH_COVERAGE_FOR_EXPORT = 0.70
MIN_MATCH_COVERAGE_FOR_CLEANING = 0.75
MAX_REMOVED_AUDIO_FRACTION = 0.25

AUDIO_EXTENSIONS = [".wav", ".mp3"]
OUTPUT_SAMPLE_RATE = 16000
OUTPUT_CHANNELS = 1

# Preflight audio/text mapping settings.
# These values are deliberately broad because audiobook speaking rates vary.
PREFLIGHT_TARGET_WPM = 155.0
PREFLIGHT_MIN_ACCEPTABLE_WPM = 85.0
PREFLIGHT_MAX_ACCEPTABLE_WPM = 220.0
PREFLIGHT_MAX_AUDIO_FILES_PER_TEXT = 4

# Important safety switch:
# The old preflight repair mode can concatenate consecutive audio files when
# the WPM value looks implausible. This is useful only for rare LibriVox cases
# where one transcript genuinely belongs to multiple consecutive audio files.
# For the current corpus, file stems are expected to match. Therefore the safe
# default is exact stem-based mapping.
ENABLE_PREFLIGHT_MULTI_AUDIO_REPAIR = False

# If multiple audio files with the same stem exist, prefer WAV over MP3 because
# WAV is usually the preprocessed pipeline-ready version.
AUDIO_EXTENSION_PRIORITY = [".wav", ".WAV", ".wave", ".WAVE", ".mp3", ".MP3"]


@dataclass
class WhisperWord:
    """A single word recognized by faster-whisper with approximate timestamps."""
    word: str
    norm: str
    start: float
    end: float
    probability: Optional[float] = None


@dataclass
class OriginalToken:
    """A token from the original transcript with its original character span."""
    text: str
    norm: str
    start_char: int
    end_char: int


@dataclass
class MatchPair:
    """A matched original-token index and Whisper-word index."""
    original_index: int
    whisper_index: int
    score: float


@dataclass
class SentenceSpan:
    """A sentence or sentence-like unit in the original transcript."""
    text: str
    start_char: int
    end_char: int
    start_token_index: int
    end_token_index: int


@dataclass
class RemovalInterval:
    """A time interval in the original audio that should be removed."""
    start: float
    end: float
    first_word_index: int
    last_word_index: int
    word_count: int
    reason: str


@dataclass
class SegmentPlan:
    """A planned MFA segment before export."""
    index: int
    sentence_start_index: int
    sentence_end_index: int
    token_start_index: int
    token_end_index: int
    audio_start_sec: float
    audio_end_sec: float
    text: str


def select_input_folder_with_dialog() -> Path:
    """Open a folder selection dialog and return the selected folder."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title="Select folder containing audio/text pairs for MFA segmentation"
        )
        root.destroy()
        if not selected:
            raise SystemExit("No folder selected. Script cancelled.")
        return Path(selected).resolve()
    except Exception:
        print("WARNING: The folder dialog could not be opened.")
        typed = input("Please enter the input folder path manually: ").strip().strip('"')
        if not typed:
            raise SystemExit("No folder provided. Script cancelled.")
        return Path(typed).resolve()


def normalize_word(word: str) -> str:
    """Normalize a word for robust matching while preserving Polish diacritics."""
    word = word.lower().replace("’", "'").replace("`", "'")
    word = re.sub(r"[^\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", "", word, flags=re.UNICODE)
    return word.strip("_")


def tokenize_original_text(text: str) -> List[OriginalToken]:
    """Tokenize the original transcript while preserving character offsets."""
    pattern = re.compile(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", flags=re.UNICODE)
    tokens: List[OriginalToken] = []
    for match in pattern.finditer(text):
        raw = match.group(0)
        norm = normalize_word(raw)
        if norm:
            tokens.append(OriginalToken(raw, norm, match.start(), match.end()))
    return tokens


def split_original_text_into_sentences(text: str, tokens: List[OriginalToken]) -> List[SentenceSpan]:
    """Split the transcript into sentence-like units using punctuation."""
    sentence_pattern = re.compile(r"[^.!?…]+(?:[.!?…]+|$)", flags=re.UNICODE)
    token_positions = [(t.start_char, t.end_char) for t in tokens]
    sentences: List[SentenceSpan] = []
    for match in sentence_pattern.finditer(text):
        sentence_text = match.group(0).strip()
        if not sentence_text:
            continue
        start_char, end_char = match.start(), match.end()
        token_indices = [i for i, (s, e) in enumerate(token_positions) if s >= start_char and e <= end_char]
        if token_indices:
            sentences.append(SentenceSpan(sentence_text, start_char, end_char, token_indices[0], token_indices[-1]))
    return sentences


def find_audio_text_pairs(input_folder: Path) -> List[Tuple[Path, Path]]:
    """Find audio/text pairs based on equal base filenames.

    This function is still available as a simple fallback. The main script uses
    build_preflight_audio_text_mapping(...) because LibriVox may split one book
    chapter across several audio files while the reference transcript is stored
    as one text file.
    """
    pairs: List[Tuple[Path, Path]] = []
    for txt_path in sorted(input_folder.glob("*.txt")):
        for ext in AUDIO_EXTENSIONS:
            audio_path = input_folder / f"{txt_path.stem}{ext}"
            if audio_path.exists():
                pairs.append((audio_path, txt_path))
                break
    return pairs


def find_all_audio_files(input_folder: Path) -> List[Path]:
    """Return all supported audio files in stable filename order."""
    audio_files: List[Path] = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(input_folder.glob(f"*{ext}"))
        audio_files.extend(input_folder.glob(f"*{ext.upper()}"))
    return sorted(set(audio_files), key=lambda p: p.name.lower())


def get_audio_duration_sec(path: Path) -> float:
    """Return the audio duration in seconds using pydub/ffmpeg."""
    return len(AudioSegment.from_file(path)) / 1000.0


def count_text_tokens(path: Path) -> int:
    """Count normalized tokens in a transcript file."""
    return len(tokenize_original_text(path.read_text(encoding="utf-8-sig")))


def concatenate_audio_files(audio_files: List[Path], output_path: Path) -> Path:
    """Concatenate multiple audio files into one temporary WAV for Whisper."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined = AudioSegment.empty()
    for audio_path in audio_files:
        combined += load_audio(audio_path)
    export_wav(combined, output_path)
    return output_path


def write_preflight_mapping_csv(path: Path, rows: List[List[str]]) -> None:
    """Write the preflight mapping report for manual inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "text_file", "audio_files", "text_words", "audio_duration_sec",
            "words_per_minute", "mapping_status", "note"
        ])
        writer.writerows(rows)


def build_preflight_audio_text_mapping(input_folder: Path, debug_output_root: Path) -> List[Tuple[Path, Path]]:
    """Build a safe audio/text mapping before Whisper is started.

    Default behavior:
        Exact stem-based mapping only.
        Example:
            chapter_01.txt -> chapter_01.wav / chapter_01.mp3

    This avoids the dangerous global-list behavior where one repaired mapping
    can shift all following audio/text pairs. A legacy multi-audio repair mode
    is still available via ENABLE_PREFLIGHT_MULTI_AUDIO_REPAIR=True, but should
    only be used deliberately for special cases.
    """
    text_files = sorted(input_folder.glob("*.txt"), key=lambda p: p.name.lower())
    audio_files = find_all_audio_files(input_folder)
    if not text_files or not audio_files:
        return []

    print("\nPreflight audio/text mapping check", flush=True)
    print("==================================", flush=True)
    print("Using exact stem-based audio/text mapping by default.", flush=True)

    logs_folder = debug_output_root / "logs"
    combined_folder = debug_output_root / "combined_audio_for_mapping"

    audio_durations = {audio_path: get_audio_duration_sec(audio_path) for audio_path in audio_files}

    # Build stem -> audio candidates. Case-insensitive stem matching makes the
    # mapping robust against inconsistent capitalization.
    audio_by_stem: Dict[str, List[Path]] = {}
    for audio_path in audio_files:
        audio_by_stem.setdefault(audio_path.stem.lower(), []).append(audio_path)

    def choose_audio_for_stem(stem: str) -> Optional[Path]:
        candidates = audio_by_stem.get(stem.lower(), [])
        if not candidates:
            return None
        # Prefer WAV/WAVE over MP3 if both exist for the same chapter.
        by_suffix = {p.suffix: p for p in sorted(candidates, key=lambda p: p.name.lower())}
        for suffix in AUDIO_EXTENSION_PRIORITY:
            if suffix in by_suffix:
                return by_suffix[suffix]
        return sorted(candidates, key=lambda p: p.name.lower())[0]

    pairs: List[Tuple[Path, Path]] = []
    report_rows: List[List[str]] = []
    used_audio_paths: set[Path] = set()
    warning_count = 0
    repaired_count = 0

    for text_path in text_files:
        word_count = count_text_tokens(text_path)
        mapped_audio = choose_audio_for_stem(text_path.stem)

        if mapped_audio is None:
            warning_count += 1
            report_rows.append([
                text_path.name,
                "",
                str(word_count),
                "0.000",
                "NV",
                "NO_EXACT_AUDIO_MATCH",
                "No audio file with the same stem was found for this transcript.",
            ])
            print(f"WARNING: No exact audio match for {text_path.name}", flush=True)
            continue

        used_audio_paths.add(mapped_audio)
        duration = audio_durations[mapped_audio]
        wpm = word_count / max(0.001, duration / 60.0)

        if PREFLIGHT_MIN_ACCEPTABLE_WPM <= wpm <= PREFLIGHT_MAX_ACCEPTABLE_WPM:
            status = "OK_EXACT_STEM"
            note = "Exact stem-based mapping."
        else:
            warning_count += 1
            status = "WARNING_EXACT_STEM_IMPLAUSIBLE_WPM"
            note = (
                "Exact stem-based mapping, but WPM is outside the expected range. "
                "This may be normal for unusual chapters, very short files, intros/outros, or slow/fast reading. "
                "The mapping was NOT shifted or repaired automatically."
            )

        pairs.append((mapped_audio, text_path))
        report_rows.append([
            text_path.name,
            mapped_audio.name,
            str(word_count),
            f"{duration:.3f}",
            f"{wpm:.1f}",
            status,
            note,
        ])
        print(
            f"  {text_path.name}: {mapped_audio.name} | {word_count} words | "
            f"{duration/60:.1f} min | {wpm:.1f} WPM | {status}",
            flush=True,
        )

    # Optional legacy mode for rare cases where one transcript truly belongs to
    # multiple consecutive audio files. This is deliberately not used by default.
    if ENABLE_PREFLIGHT_MULTI_AUDIO_REPAIR:
        print(
            "\nWARNING: ENABLE_PREFLIGHT_MULTI_AUDIO_REPAIR is True. "
            "Legacy greedy repair mode is not recommended for mixed audiobook corpora.",
            flush=True,
        )
        # Keep the exact-mapping results. The legacy repair is intentionally not
        # executed automatically here to avoid silent global shifts. If needed,
        # handle such books in a separate folder or create an explicit mapping.

    unused_audio = [p for p in audio_files if p not in used_audio_paths]
    for path in unused_audio:
        report_rows.append([
            "",
            path.name,
            "0",
            f"{audio_durations[path]:.3f}",
            "NV",
            "UNUSED_AUDIO",
            "Audio file was not assigned to any transcript by exact stem matching.",
        ])

    if unused_audio:
        warning_count += len(unused_audio)
        print(
            "WARNING: Unused audio files: " + ", ".join(p.name for p in unused_audio[:20]) +
            (" ..." if len(unused_audio) > 20 else ""),
            flush=True,
        )

    write_preflight_mapping_csv(logs_folder / "preflight_audio_text_mapping.csv", report_rows)
    print(f"\nPreflight report written to: {logs_folder / 'preflight_audio_text_mapping.csv'}", flush=True)
    print(f"Exact pairs: {len(pairs)}", flush=True)
    print(f"Warnings: {warning_count}", flush=True)

    if warning_count:
        print("\nThe script found preflight warnings.", flush=True)
        print("Exact stem-based mappings are kept, but warnings should be checked in the preflight CSV.", flush=True)
        while True:
            answer = input("Do you want to continue with this mapping? (y/n): ").strip().lower()
            if answer in {"y", "yes", "j", "ja"}:
                break
            if answer in {"n", "no", "nein"}:
                raise SystemExit("Batch interrupted by user after preflight mapping check.")
            print("Please answer with y or n.", flush=True)

    return pairs


def load_audio(path: Path) -> AudioSegment:
    """Load audio and convert it to mono 16 kHz for consistent MFA export."""
    return AudioSegment.from_file(path).set_channels(OUTPUT_CHANNELS).set_frame_rate(OUTPUT_SAMPLE_RATE)


def export_wav(audio: AudioSegment, path: Path) -> None:
    """Export an AudioSegment as WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(path, format="wav")


def transcribe_with_faster_whisper(audio_path: Path, model: WhisperModel) -> List[WhisperWord]:
    """Transcribe audio with faster-whisper and return word-level timestamps.

    Important implementation detail:
    faster-whisper returns a lazy generator. The actual transcription does not
    start at the model.transcribe(...) call, but only when the segments are
    iterated. Therefore this function prints explicit progress messages while
    consuming the generator.
    """
    print(f"  Transcribing with faster-whisper: {audio_path.name}", flush=True)
    transcribe_start = time.perf_counter()
    segments, info = model.transcribe(
        str(audio_path),
        language=WHISPER_LANGUAGE,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 700},
        beam_size=3,
        condition_on_previous_text=False,
    )
    print(
        f"  Whisper generator created. Detected language: {getattr(info, 'language', 'unknown')}. "
        "Consuming segments now...",
        flush=True,
    )
    words: List[WhisperWord] = []
    segment_count = 0
    for segment in segments:
        segment_count += 1
        if segment_count == 1 or segment_count % 10 == 0:
            print(
                f"    segment {segment_count:04d}: "
                f"{float(segment.start):.1f}-{float(segment.end):.1f} sec, "
                f"words collected: {len(words)}",
                flush=True,
            )
        if not segment.words:
            continue
        for word in segment.words:
            raw = word.word.strip()
            norm = normalize_word(raw)
            if norm:
                words.append(WhisperWord(raw, norm, float(word.start), float(word.end), getattr(word, "probability", None)))
    print(
        f"  Transcription finished in {time.perf_counter() - transcribe_start:.1f} seconds. "
        f"Segments: {segment_count}, words: {len(words)}",
        flush=True,
    )
    return words


def match_original_to_whisper(original_tokens: List[OriginalToken], whisper_words: List[WhisperWord]) -> List[MatchPair]:
    """Align normalized original tokens with normalized Whisper words."""
    original_norm = [t.norm for t in original_tokens]
    whisper_norm = [w.norm for w in whisper_words]
    matcher = difflib.SequenceMatcher(a=original_norm, b=whisper_norm, autojunk=False)
    pairs: List[MatchPair] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                pairs.append(MatchPair(i1 + offset, j1 + offset, 100.0))
        elif tag == "replace":
            pairs.extend(fuzzy_match_local_region(original_tokens[i1:i2], whisper_words[j1:j2], i1, j1))
    return remove_crossing_matches(pairs)


def fuzzy_match_local_region(original_region: List[OriginalToken], whisper_region: List[WhisperWord], original_offset: int, whisper_offset: int) -> List[MatchPair]:
    """Fuzzy-match a local replacement region to tolerate single-word ASR errors."""
    pairs: List[MatchPair] = []
    used_whisper: set[int] = set()
    for i, token in enumerate(original_region):
        best_j = None
        best_score = 0.0
        for j, word in enumerate(whisper_region):
            if j in used_whisper:
                continue
            score = fuzz.ratio(token.norm, word.norm)
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is not None and best_score >= MIN_MATCH_RATIO_FOR_ANCHOR:
            used_whisper.add(best_j)
            pairs.append(MatchPair(original_offset + i, whisper_offset + best_j, float(best_score)))
    return pairs


def remove_crossing_matches(pairs: List[MatchPair]) -> List[MatchPair]:
    """Keep only left-to-right monotonic matches."""
    cleaned: List[MatchPair] = []
    last_original = -1
    last_whisper = -1
    for pair in sorted(pairs, key=lambda p: (p.original_index, p.whisper_index)):
        if pair.original_index > last_original and pair.whisper_index > last_whisper:
            cleaned.append(pair)
            last_original = pair.original_index
            last_whisper = pair.whisper_index
    return cleaned


def build_token_to_time_map(original_tokens: List[OriginalToken], whisper_words: List[WhisperWord], matches: List[MatchPair]) -> Dict[int, Tuple[float, float]]:
    """Map original token indices to approximate audio timestamps."""
    token_time: Dict[int, Tuple[float, float]] = {}
    if not matches:
        return token_time
    for pair in matches:
        word = whisper_words[pair.whisper_index]
        token_time[pair.original_index] = (word.start, word.end)
    sorted_matches = sorted(matches, key=lambda p: p.original_index)
    for left, right in zip(sorted_matches, sorted_matches[1:]):
        oi_left, oi_right = left.original_index, right.original_index
        wi_left, wi_right = left.whisper_index, right.whisper_index
        if oi_right <= oi_left + 1:
            continue
        t_left = whisper_words[wi_left].end
        t_right = whisper_words[wi_right].start
        if t_right <= t_left:
            continue
        gap_tokens = oi_right - oi_left
        step = (t_right - t_left) / gap_tokens
        for k in range(1, gap_tokens):
            idx = oi_left + k
            token_time[idx] = (t_left + step * (k - 0.2), t_left + step * (k + 0.2))
    return token_time


def infer_trim_range_from_matches(whisper_words: List[WhisperWord], matches: List[MatchPair], audio_duration_sec: float) -> Tuple[float, float]:
    """Infer the audio region corresponding to the original transcript."""
    if not matches:
        return 0.0, audio_duration_sec
    first = matches[0]
    last = matches[-1]
    start_sec = max(0.0, whisper_words[first.whisper_index].start - TRIM_PADDING_MS / 1000)
    end_sec = min(audio_duration_sec, whisper_words[last.whisper_index].end + TRIM_PADDING_MS / 1000)
    if end_sec <= start_sec:
        return 0.0, audio_duration_sec
    return start_sec, end_sec


def estimate_sentence_times(sentences: List[SentenceSpan], token_time: Dict[int, Tuple[float, float]]) -> List[Tuple[Optional[float], Optional[float]]]:
    """Estimate start and end times for each original sentence."""
    sentence_times: List[Tuple[Optional[float], Optional[float]]] = []
    for sentence in sentences:
        times = [token_time[i] for i in range(sentence.start_token_index, sentence.end_token_index + 1) if i in token_time]
        if not times:
            sentence_times.append((None, None))
        else:
            sentence_times.append((min(t[0] for t in times), max(t[1] for t in times)))
    return sentence_times


def approximate_missing_sentence_time(index: int, sentence_times: List[Tuple[Optional[float], Optional[float]]], trim_start_sec: float, trim_end_sec: float) -> Tuple[float, float]:
    """Estimate missing sentence times from neighboring sentence timestamps."""
    previous_end = trim_start_sec
    next_start = trim_end_sec
    for j in range(index - 1, -1, -1):
        if sentence_times[j][1] is not None:
            previous_end = float(sentence_times[j][1])
            break
    for j in range(index + 1, len(sentence_times)):
        if sentence_times[j][0] is not None:
            next_start = float(sentence_times[j][0])
            break
    if next_start <= previous_end:
        return previous_end, min(trim_end_sec, previous_end + 5.0)
    return previous_end, next_start



def estimate_complete_token_times(
    original_tokens: List[OriginalToken],
    token_time: Dict[int, Tuple[float, float]],
    trim_start_sec: float,
    trim_end_sec: float,
) -> Dict[int, Tuple[float, float]]:
    """Return approximate start/end times for every original token.

    Matched tokens keep their Whisper-derived timestamps. Missing tokens are
    interpolated between the nearest known anchors. This is important because
    the original MFA text snippets must be written from the original transcript,
    even when Whisper misrecognized or skipped individual words.
    """
    n = len(original_tokens)
    if n == 0:
        return {}
    complete: Dict[int, Tuple[float, float]] = dict(token_time)
    known = sorted(complete.keys())
    if not known:
        total = max(0.001, trim_end_sec - trim_start_sec)
        step = total / n
        for i in range(n):
            complete[i] = (trim_start_sec + i * step, trim_start_sec + (i + 0.8) * step)
        return complete

    # Fill tokens before the first known timestamp.
    first = known[0]
    first_start = complete[first][0]
    if first > 0:
        available = max(0.001, first_start - trim_start_sec)
        step = available / first
        for i in range(first):
            complete[i] = (trim_start_sec + i * step, trim_start_sec + (i + 0.8) * step)

    # Fill gaps between known timestamps.
    known = sorted(complete.keys())
    for left, right in zip(known, known[1:]):
        if right <= left + 1:
            continue
        left_end = complete[left][1]
        right_start = complete[right][0]
        if right_start <= left_end:
            # If timestamps overlap, assign a very small artificial spacing.
            step = 0.02
            for k, i in enumerate(range(left + 1, right), start=1):
                complete[i] = (left_end + (k - 1) * step, left_end + k * step)
        else:
            missing = right - left - 1
            step = (right_start - left_end) / (missing + 1)
            for k, i in enumerate(range(left + 1, right), start=1):
                center = left_end + k * step
                complete[i] = (max(left_end, center - step * 0.35), min(right_start, center + step * 0.35))

    # Fill tokens after the last known timestamp.
    known = sorted(complete.keys())
    last = known[-1]
    last_end = complete[last][1]
    if last < n - 1:
        remaining = n - last - 1
        available = max(0.001, trim_end_sec - last_end)
        step = available / remaining
        for k, i in enumerate(range(last + 1, n), start=1):
            complete[i] = (last_end + (k - 1) * step, last_end + k * step * 0.8)

    # Clamp to trim range and enforce monotonicity.
    previous_end = trim_start_sec
    for i in range(n):
        start, end = complete[i]
        start = max(trim_start_sec, min(trim_end_sec, float(start)))
        end = max(start + 0.001, min(trim_end_sec, float(end)))
        if start < previous_end:
            start = previous_end
            end = max(start + 0.001, end)
        complete[i] = (start, min(trim_end_sec, end))
        previous_end = complete[i][1]
    return complete


def text_for_token_span(original_text: str, original_tokens: List[OriginalToken], start_token: int, end_token: int) -> str:
    """Extract original transcript text for an inclusive token span."""
    start_char = original_tokens[start_token].start_char
    end_char = original_tokens[end_token].end_char
    return original_text[start_char:end_char].strip()


def choose_pause_aware_cut(
    token_times: Dict[int, Tuple[float, float]],
    start_token: int,
    last_token: int,
    segment_start_sec: float,
) -> Tuple[int, float, str, float]:
    """Choose the best word-boundary cut for one MFA segment.

    The function never cuts inside a token. It searches for the best boundary
    between two neighboring tokens whose cut time keeps the segment between
    MIN_SEGMENT_SEC and MAX_SEGMENT_SEC. The preferred cut is the midpoint of a
    pause between words. Long pauses are favored, but the selected duration is
    also kept close to TARGET_SEGMENT_SEC.

    Returns
    -------
    cut_token_index:
        The last token included in the segment.
    cut_time_sec:
        The audio cut point, usually the middle of a pause after the token.
    decision:
        Human-readable decision label for the log file.
    pause_sec:
        The pause duration after cut_token_index, if available.
    """
    best = None
    fallback_under_max = None

    for i in range(start_token, last_token):
        current_end = token_times[i][1]
        next_start = token_times[i + 1][0]
        pause = max(0.0, next_start - current_end)
        cut_time = current_end + pause / 2.0
        duration = cut_time - segment_start_sec

        if duration <= MAX_SEGMENT_SEC:
            fallback_under_max = (i, cut_time, "fallback_last_boundary_under_max", pause, duration)

        if MIN_SEGMENT_SEC <= duration <= MAX_SEGMENT_SEC:
            target_penalty = abs(duration - TARGET_SEGMENT_SEC) / max(1.0, TARGET_SEGMENT_SEC)
            pause_bonus = pause * PAUSE_SCORE_WEIGHT
            preferred_pause_bonus = 1.0 if pause >= MIN_PREFERRED_WORD_PAUSE_SEC else 0.0
            score = pause_bonus + preferred_pause_bonus - TARGET_SCORE_WEIGHT * target_penalty
            candidate = (score, i, cut_time, pause, duration)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is not None:
        _score, i, cut_time, pause, duration = best
        label = "preferred_pause_cut" if pause >= MIN_PREFERRED_WORD_PAUSE_SEC else "word_boundary_cut"
        return i, cut_time, label, pause

    # If no legal 30-60 s segment is possible, use the last word boundary below
    # the maximum. This can happen at the beginning/end or with unusual timing.
    if fallback_under_max is not None:
        i, cut_time, label, pause, _duration = fallback_under_max
        return i, cut_time, label, pause

    # Emergency fallback: include at least one token. This still avoids cutting
    # inside a word; the cut is placed at the token end.
    i = min(start_token, last_token)
    return i, token_times[i][1], "emergency_single_token", 0.0


def plan_mfa_segments_pause_aware(
    original_text: str,
    original_tokens: List[OriginalToken],
    token_time: Dict[int, Tuple[float, float]],
    trim_start_sec: float,
    trim_end_sec: float,
) -> List[SegmentPlan]:
    """Create MFA segments using pause-aware word-level cutting.

    This is the main segmentation strategy for preprocessed transcripts without
    punctuation. It does not require sentence boundaries. Instead, it cuts only
    between original-text tokens and prefers boundaries where faster-whisper
    detected an acoustic pause between neighboring words. This prevents cutting
    through individual sound realizations while still producing short segments
    that are much easier for MFA to align.
    """
    if not original_tokens:
        return []

    token_times = estimate_complete_token_times(original_tokens, token_time, trim_start_sec, trim_end_sec)
    plans: List[SegmentPlan] = []
    current_token = 0
    current_audio_start = trim_start_sec
    last_token = len(original_tokens) - 1
    segment_index = 1

    while current_token <= last_token:
        remaining_duration = trim_end_sec - current_audio_start
        if remaining_duration <= MAX_SEGMENT_SEC:
            end_token = last_token
            end_time = trim_end_sec
        else:
            end_token, end_time, _decision, _pause = choose_pause_aware_cut(
                token_times, current_token, last_token, current_audio_start
            )
            if end_token < current_token:
                end_token = current_token
                end_time = token_times[current_token][1]

        chunk_text = text_for_token_span(original_text, original_tokens, current_token, end_token)
        if chunk_text:
            plans.append(SegmentPlan(
                index=segment_index,
                sentence_start_index=-1,
                sentence_end_index=-1,
                token_start_index=current_token,
                token_end_index=end_token,
                audio_start_sec=max(trim_start_sec, current_audio_start),
                audio_end_sec=min(trim_end_sec, end_time),
                text=chunk_text,
            ))
            segment_index += 1

        current_token = end_token + 1
        current_audio_start = end_time

    return plans




def detect_large_unmatched_whisper_intervals(
    whisper_words: List[WhisperWord],
    matches: List[MatchPair],
    audio_duration_sec: float,
) -> List[RemovalInterval]:
    """Detect long Whisper-only passages that are not represented in the reference.

    This function uses the stable v5 alignment as its basis. It does NOT delete
    isolated mismatched words. Instead, it searches for continuous stretches of
    Whisper words that were not aligned to any original-text token. A stretch is
    removed only if it is long enough in words and time. This is intended for
    LibriVox intros/outros, repeated chapter announcements, accidental comments,
    or other longer non-reference speech.
    """
    matched_whisper_indices = {m.whisper_index for m in matches}
    intervals: List[RemovalInterval] = []

    start_idx: Optional[int] = None
    prev_idx: Optional[int] = None

    def close_block(block_start: int, block_end: int) -> None:
        first = whisper_words[block_start]
        last = whisper_words[block_end]
        word_count = block_end - block_start + 1
        duration = max(0.0, last.end - first.start)
        is_edge = block_start == 0 or block_end == len(whisper_words) - 1

        if is_edge:
            removable = word_count >= MIN_EDGE_REMOVAL_WORDS and duration >= MIN_EDGE_REMOVAL_SEC
            reason = "edge_whisper_only_block"
        else:
            removable = word_count >= MIN_INTERNAL_REMOVAL_WORDS and duration >= MIN_INTERNAL_REMOVAL_SEC
            reason = "internal_whisper_only_block"

        if not removable:
            return

        remove_start = max(0.0, first.start - REMOVAL_KEEP_PADDING_SEC)
        remove_end = min(audio_duration_sec, last.end + REMOVAL_KEEP_PADDING_SEC)

        # Never cross into neighboring matched words.
        for j in range(block_start - 1, -1, -1):
            if j in matched_whisper_indices:
                remove_start = max(remove_start, whisper_words[j].end + 0.02)
                break
        for j in range(block_end + 1, len(whisper_words)):
            if j in matched_whisper_indices:
                remove_end = min(remove_end, whisper_words[j].start - 0.02)
                break

        if remove_end - remove_start >= 0.2:
            intervals.append(RemovalInterval(remove_start, remove_end, block_start, block_end, word_count, reason))

    for idx, _word in enumerate(whisper_words):
        unmatched = idx not in matched_whisper_indices
        if unmatched and start_idx is None:
            start_idx = idx
        if not unmatched and start_idx is not None:
            close_block(start_idx, prev_idx if prev_idx is not None else start_idx)
            start_idx = None
        prev_idx = idx

    if start_idx is not None:
        close_block(start_idx, prev_idx if prev_idx is not None else start_idx)

    return merge_removal_intervals(intervals)


def merge_removal_intervals(intervals: List[RemovalInterval]) -> List[RemovalInterval]:
    """Merge overlapping or nearly adjacent removal intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x.start)
    merged: List[RemovalInterval] = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current.start <= last.end + 0.15:
            merged[-1] = RemovalInterval(
                start=last.start,
                end=max(last.end, current.end),
                first_word_index=last.first_word_index,
                last_word_index=current.last_word_index,
                word_count=last.word_count + current.word_count,
                reason=last.reason + "+" + current.reason,
            )
        else:
            merged.append(current)
    return merged


def apply_removal_intervals_to_audio(audio: AudioSegment, intervals: List[RemovalInterval]) -> Tuple[AudioSegment, List[Tuple[float, float, float, float]]]:
    """Remove intervals from audio and return a time-warp map.

    The returned map contains tuples of:
        original_start, original_end, cleaned_start, cleaned_end
    for every kept audio chunk. It allows original Whisper timestamps to be
    transformed into cleaned-audio timestamps without running Whisper again.
    """
    cleaned = AudioSegment.empty()
    kept_chunks: List[Tuple[float, float, float, float]] = []
    cursor = 0.0
    cleaned_cursor = 0.0
    audio_duration = len(audio) / 1000.0

    for interval in intervals:
        keep_start = cursor
        keep_end = max(cursor, interval.start)
        if keep_end > keep_start:
            chunk = audio[int(keep_start * 1000): int(keep_end * 1000)]
            cleaned += chunk
            dur = keep_end - keep_start
            kept_chunks.append((keep_start, keep_end, cleaned_cursor, cleaned_cursor + dur))
            cleaned_cursor += dur
        cursor = max(cursor, interval.end)

    if cursor < audio_duration:
        chunk = audio[int(cursor * 1000):]
        cleaned += chunk
        dur = audio_duration - cursor
        kept_chunks.append((cursor, audio_duration, cleaned_cursor, cleaned_cursor + dur))

    return cleaned, kept_chunks


def transform_original_time_to_cleaned_time(t: float, kept_chunks: List[Tuple[float, float, float, float]]) -> float:
    """Map a timestamp from original audio time to cleaned audio time."""
    if not kept_chunks:
        return t
    for original_start, original_end, cleaned_start, cleaned_end in kept_chunks:
        if original_start <= t <= original_end:
            return cleaned_start + (t - original_start)
        if t < original_start:
            return cleaned_start
    return kept_chunks[-1][3]


def transform_token_times_to_cleaned_audio(
    token_time: Dict[int, Tuple[float, float]],
    kept_chunks: List[Tuple[float, float, float, float]],
) -> Dict[int, Tuple[float, float]]:
    """Transform all token timestamps after audio cleaning."""
    transformed: Dict[int, Tuple[float, float]] = {}
    for idx, (start, end) in token_time.items():
        new_start = transform_original_time_to_cleaned_time(start, kept_chunks)
        new_end = transform_original_time_to_cleaned_time(end, kept_chunks)
        if new_end < new_start:
            new_end = new_start + 0.001
        transformed[idx] = (new_start, new_end)
    return transformed


def write_removal_intervals_csv(path: Path, intervals: List[RemovalInterval], whisper_words: List[WhisperWord]) -> None:
    """Write removed non-reference intervals to CSV for manual quality control."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["start_sec", "end_sec", "duration_sec", "first_word_index", "last_word_index", "word_count", "reason", "text_preview"])
        for interval in intervals:
            preview_words = [w.word for w in whisper_words[interval.first_word_index: interval.last_word_index + 1]]
            preview = re.sub(r"\s+", " ", " ".join(preview_words))[:220]
            writer.writerow([
                f"{interval.start:.3f}",
                f"{interval.end:.3f}",
                f"{interval.end - interval.start:.3f}",
                interval.first_word_index,
                interval.last_word_index,
                interval.word_count,
                interval.reason,
                preview,
            ])

def write_pause_candidates_csv(path: Path, original_tokens: List[OriginalToken], token_time: Dict[int, Tuple[float, float]]) -> None:
    """Write all token-boundary pauses to CSV for quality control."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["left_token_index", "left_token", "right_token_index", "right_token", "left_end_sec", "right_start_sec", "pause_sec", "candidate_cut_sec"])
        for i in range(len(original_tokens) - 1):
            left_end = token_time[i][1]
            right_start = token_time[i + 1][0]
            pause = max(0.0, right_start - left_end)
            cut = left_end + pause / 2.0
            writer.writerow([i, original_tokens[i].text, i + 1, original_tokens[i + 1].text, f"{left_end:.3f}", f"{right_start:.3f}", f"{pause:.3f}", f"{cut:.3f}"])

def write_whisper_words_csv(path: Path, words: List[WhisperWord]) -> None:
    """Write Whisper words and timestamps to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["index", "word", "normalized_word", "start_sec", "end_sec", "probability"])
        for i, word in enumerate(words):
            writer.writerow([i, word.word, word.norm, f"{word.start:.3f}", f"{word.end:.3f}", word.probability])


def write_alignment_matches_csv(path: Path, original_tokens: List[OriginalToken], whisper_words: List[WhisperWord], matches: List[MatchPair]) -> None:
    """Write original-token to Whisper-word matches to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["original_index", "original_token", "whisper_index", "whisper_word", "whisper_start_sec", "whisper_end_sec", "match_score"])
        for pair in matches:
            original = original_tokens[pair.original_index]
            whisper = whisper_words[pair.whisper_index]
            writer.writerow([pair.original_index, original.text, pair.whisper_index, whisper.word, f"{whisper.start:.3f}", f"{whisper.end:.3f}", f"{pair.score:.1f}"])


def write_segment_log_csv(path: Path, plans: List[SegmentPlan]) -> None:
    """Write the final MFA segment plan to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["segment_index", "audio_start_sec", "audio_end_sec", "duration_sec", "sentence_start_index", "sentence_end_index", "token_start_index", "token_end_index", "text_preview"])
        for plan in plans:
            preview = re.sub(r"\s+", " ", plan.text)[:120]
            writer.writerow([plan.index, f"{plan.audio_start_sec:.3f}", f"{plan.audio_end_sec:.3f}", f"{plan.audio_end_sec - plan.audio_start_sec:.3f}", plan.sentence_start_index, plan.sentence_end_index, plan.token_start_index, plan.token_end_index, preview])


def export_trimmed_audio_and_text(audio: AudioSegment, original_text: str, output_audio_path: Path, output_text_path: Path, trim_start_sec: float, trim_end_sec: float) -> None:
    """Export the trimmed full chapter audio and original transcript."""
    start_ms = max(0, int(trim_start_sec * 1000))
    end_ms = min(len(audio), int(trim_end_sec * 1000))
    export_wav(audio[start_ms:end_ms], output_audio_path)
    output_text_path.parent.mkdir(parents=True, exist_ok=True)
    output_text_path.write_text(original_text.strip() + "\n", encoding="utf-8")


def export_mfa_segments(
    audio: AudioSegment,
    plans: List[SegmentPlan],
    mfa_output_folder: Path,
    base_name: str,
) -> None:
    """Export WAV/TXT segment pairs to the configured MFA input folder."""
    mfa_output_folder.mkdir(parents=True, exist_ok=True)

    for plan in plans:
        start_ms = max(0, int(plan.audio_start_sec * 1000) - SEGMENT_PADDING_MS)
        end_ms = min(len(audio), int(plan.audio_end_sec * 1000) + SEGMENT_PADDING_MS)
        snippet = audio[start_ms:end_ms]
        out_base = f"{base_name}_seg_{plan.index:04d}"

        export_wav(snippet, mfa_output_folder / f"{out_base}.wav")
        (mfa_output_folder / f"{out_base}.txt").write_text(
            plan.text.strip() + "\n",
            encoding="utf-8",
        )



def delete_existing_outputs_for_base(base_name: str, mfa_output_root: Path, debug_output_root: Path) -> None:
    """Delete stale output files for one base name before reprocessing.

    This prevents old bad segment pairs from remaining in 04___mfa_input when a
    later safety check skips the pair.
    """
    patterns = [
        (mfa_output_root, f"{base_name}_seg_*.wav"),
        (mfa_output_root, f"{base_name}_seg_*.txt"),
        (debug_output_root / "trimmed", f"{base_name}_cleaned.wav"),
        (debug_output_root / "trimmed", f"{base_name}_cleaned.txt"),
        (debug_output_root / "logs", f"{base_name}_*.csv"),
    ]
    for folder, pattern in patterns:
        if not folder.exists():
            continue
        for path in folder.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def ask_skip_or_interrupt(reason_title: str, details: str, summary_lines: List[str]) -> bool:
    """Ask the user whether a problematic chapter should be skipped.

    Returns True if the chapter should be skipped. Raises SystemExit if the
    user chooses not to skip, so the complete batch run is interrupted.
    """
    winsound.Beep(300, 1000)
    winsound.Beep(300, 1200)
    winsound.Beep(300, 1300)

    print("\n" + "=" * 78, flush=True)
    print(f"SAFETY WARNING: {reason_title}", flush=True)
    print(details, flush=True)
    print("\nThe script can interrupt here. Alternatively, this chapter can be skipped and the next chapter will be analysed.", flush=True)
    while True:
        answer = input("Do you want to skip this chapter? (y/n): ").strip().lower()
        if answer in {"y", "yes", "j", "ja"}:
            print("Chapter skipped. Continuing with the next pair...", flush=True)
            summary_lines.append(f"SKIPPED BY USER: {reason_title}")
            summary_lines.append(details)
            print("=" * 78 + "\n", flush=True)
            return True
        if answer in {"n", "no", "nein"}:
            summary_lines.append(f"INTERRUPTED BY USER: {reason_title}")
            summary_lines.append(details)
            raise SystemExit("Batch interrupted by user after safety warning.")
        print("Please answer with y or n.", flush=True)

def process_pair(audio_path: Path, text_path: Path, model: WhisperModel, mfa_output_root: Path, debug_output_root: Path, summary_lines: List[str]) -> None:
    """Process one audio/text pair."""
    base_name = text_path.stem
    trimmed_folder = debug_output_root / "trimmed"
    segments_folder = mfa_output_root
    logs_folder = debug_output_root / "logs"
    delete_existing_outputs_for_base(base_name, mfa_output_root, debug_output_root)
    summary_lines.append(f"\n=== Processing: {base_name} ===")
    summary_lines.append(f"Audio: {audio_path}")
    summary_lines.append(f"Text:  {text_path}")
    print(f"\nProcessing pair: {base_name}", flush=True)
    print("  Reading original transcript...", flush=True)
    original_text = text_path.read_text(encoding="utf-8-sig")
    original_tokens = tokenize_original_text(original_text)
    sentences = split_original_text_into_sentences(original_text, original_tokens)
    if not original_tokens:
        raise ValueError(f"No valid tokens found in text file: {text_path}")
    if not sentences:
        raise ValueError(f"No sentence-like units found in text file: {text_path}")
    print("  Loading audio with pydub/ffmpeg...", flush=True)
    audio = load_audio(audio_path)
    audio_duration_sec = len(audio) / 1000.0
    print(f"  Audio loaded. Duration: {audio_duration_sec:.2f} sec", flush=True)
    summary_lines.append(f"Audio duration: {audio_duration_sec:.2f} sec")
    summary_lines.append(f"Original tokens: {len(original_tokens)}")
    summary_lines.append(f"Sentence units: {len(sentences)}")
    whisper_words = transcribe_with_faster_whisper(audio_path, model)
    if not whisper_words:
        raise ValueError(f"Whisper did not return any words for: {audio_path}")
    write_whisper_words_csv(logs_folder / f"{base_name}_whisper_words.csv", whisper_words)
    print("  Matching original text to Whisper word timestamps...", flush=True)
    matches = match_original_to_whisper(original_tokens, whisper_words)
    write_alignment_matches_csv(logs_folder / f"{base_name}_alignment_matches.csv", original_tokens, whisper_words, matches)
    match_coverage = len(matches) / max(1, len(original_tokens))
    summary_lines.append(f"Whisper words: {len(whisper_words)}")
    summary_lines.append(f"Matched original tokens: {len(matches)} ({match_coverage:.1%})")
    if not matches:
        raise ValueError("No usable matches between original text and Whisper transcript. Check whether the correct text file was paired with the audio.")
    if match_coverage < MIN_MATCH_COVERAGE_FOR_EXPORT:
        details = (
            "Less than 70% of the script data could be matched to the corresponding audio file for this chapter.\n"
            f"Matched original tokens: {len(matches)} of {len(original_tokens)} ({match_coverage:.1%}).\n"
            "Please check the script file and audio file manually. This usually means that the audio and text files are not the same chapter/part.\n"
            "No MFA segments will be exported for this chapter unless you fix the source files and run the script again."
        )
        if ask_skip_or_interrupt("Low transcript/audio match coverage", details, summary_lines):
            return
    print("  Detecting long Whisper-only passages for internal audio cleaning...", flush=True)
    removal_intervals = detect_large_unmatched_whisper_intervals(whisper_words, matches, audio_duration_sec)
    write_removal_intervals_csv(logs_folder / f"{base_name}_removed_intervals.csv", removal_intervals, whisper_words)
    removed_duration = sum(interval.end - interval.start for interval in removal_intervals)
    print(f"  Removal intervals: {len(removal_intervals)}; removed duration: {removed_duration:.2f} sec", flush=True)
    summary_lines.append(f"Removal intervals: {len(removal_intervals)}")
    summary_lines.append(f"Removed duration: {removed_duration:.2f} sec")
    removed_fraction = removed_duration / max(0.001, audio_duration_sec)
    summary_lines.append(f"Removed fraction: {removed_fraction:.1%}")

    if match_coverage < MIN_MATCH_COVERAGE_FOR_CLEANING and removal_intervals:
        details = (
            "The script detected removable Whisper-only passages, but the transcript/audio match coverage is too low for safe cleaning.\n"
            f"Matched original tokens: {len(matches)} of {len(original_tokens)} ({match_coverage:.1%}; required at least {MIN_MATCH_COVERAGE_FOR_CLEANING:.0%}).\n"
            "Please check the script file and audio file manually. No MFA segments will be exported for this chapter unless you fix the source files and run the script again."
        )
        if ask_skip_or_interrupt("Unsafe cleaning because match coverage is too low", details, summary_lines):
            return
    if removed_fraction > MAX_REMOVED_AUDIO_FRACTION:
        details = (
            "The script would remove an unusually large part of the audio for this chapter.\n"
            f"Proposed removed duration: {removed_duration:.2f} sec of {audio_duration_sec:.2f} sec ({removed_fraction:.1%}; allowed maximum {MAX_REMOVED_AUDIO_FRACTION:.0%}).\n"
            "This strongly suggests a wrong audio/text pair or a failed ASR alignment. Please check the script file manually.\n"
            "No MFA segments will be exported for this chapter unless you fix the source files and run the script again."
        )
        if ask_skip_or_interrupt("Too much proposed audio removal", details, summary_lines):
            return

    print("  Applying internal audio cleaning...", flush=True)
    cleaned_audio, kept_chunks = apply_removal_intervals_to_audio(audio, removal_intervals)
    cleaned_duration_sec = len(cleaned_audio) / 1000.0
    cleaned_trim_start_sec = 0.0
    cleaned_trim_end_sec = cleaned_duration_sec
    summary_lines.append(f"Cleaned audio duration: {cleaned_duration_sec:.2f} sec")

    print("  Exporting cleaned chapter audio/text...", flush=True)
    export_trimmed_audio_and_text(cleaned_audio, original_text, trimmed_folder / f"{base_name}_cleaned.wav", trimmed_folder / f"{base_name}_cleaned.txt", cleaned_trim_start_sec, cleaned_trim_end_sec)

    token_time = build_token_to_time_map(original_tokens, whisper_words, matches)
    complete_token_time_original = estimate_complete_token_times(original_tokens, token_time, 0.0, audio_duration_sec)
    complete_token_time = transform_token_times_to_cleaned_audio(complete_token_time_original, kept_chunks)
    write_pause_candidates_csv(logs_folder / f"{base_name}_pause_candidates.csv", original_tokens, complete_token_time)

    print("  Planning MFA segments with pause-aware word-boundary cuts...", flush=True)
    plans = plan_mfa_segments_pause_aware(original_text, original_tokens, complete_token_time, cleaned_trim_start_sec, cleaned_trim_end_sec)
    write_segment_log_csv(logs_folder / f"{base_name}_segment_log.csv", plans)
    print(f"  Exporting {len(plans)} MFA audio/text snippets...", flush=True)
    export_mfa_segments(cleaned_audio, plans, segments_folder, base_name)
    summary_lines.append(f"Exported MFA segments: {len(plans)}")


def main() -> None:
    """Run the complete preprocessing pipeline."""
    print("\nMA Whisper Segmenter for MFA")
    print("============================\n")
    dataset_suffix = DATASET_SUFFIX.strip().lower()
    valid_suffixes = {"training", "validation", "test"}
    if dataset_suffix not in valid_suffixes:
        raise SystemExit(
            f"Invalid DATASET_SUFFIX: {DATASET_SUFFIX!r}. "
            "Allowed values are: training, validation, test."
        )

    input_folder = config.PREPARED_WHISPER_INPUT_DIR / dataset_suffix
    mfa_output_root = config.MFA_INPUT_DIR / dataset_suffix
    debug_output_root = config.WHISPER_DEBUG_DIR / dataset_suffix

    if not input_folder.exists():
        raise SystemExit(f"Input folder does not exist: {input_folder}")

    mfa_output_root.mkdir(parents=True, exist_ok=True)
    debug_output_root.mkdir(parents=True, exist_ok=True)
    pairs = build_preflight_audio_text_mapping(input_folder, debug_output_root)
    if not pairs:
        raise SystemExit("No usable audio/text mapping found. Expected transcript TXT files and supported audio files in the selected folder.")
    print(f"Selected input folder: {input_folder}", flush=True)
    print(f"MFA input folder:      {mfa_output_root}", flush=True)
    print(f"Whisper debug folder:  {debug_output_root}", flush=True)
    print(f"Mapped text units:     {len(pairs)}", flush=True)
    print("\nInput recommendation: use preprocessed WAV/TXT pairs, not raw source material.", flush=True)
    WHISPER_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Whisper model directory: {WHISPER_MODEL_DIRECTORY}", flush=True)
    print("Loading faster-whisper model...", flush=True)
    load_start = time.perf_counter()
    try:
        model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=str(WHISPER_MODEL_DIRECTORY),
            local_files_only=LOCAL_MODEL_FILES_ONLY,
        )
        print(f"Model loaded in {time.perf_counter() - load_start:.1f} seconds.", flush=True)
    except Exception as cuda_error:
        if WHISPER_DEVICE == "cuda":
            print("\nWARNING: CUDA model loading failed. Falling back to CPU/int8.")
            print(f"CUDA error: {cuda_error}\n")
            model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", download_root=str(WHISPER_MODEL_DIRECTORY), local_files_only=LOCAL_MODEL_FILES_ONLY)
            print(f"CPU fallback model loaded in {time.perf_counter() - load_start:.1f} seconds.", flush=True)
        else:
            raise
    summary_lines: List[str] = [
        "MA Whisper Segmenter for MFA",
        "============================",
        f"Input folder: {input_folder}",
        f"MFA input folder: {mfa_output_root}",
        f"Whisper debug folder: {debug_output_root}",
        f"Whisper model: {WHISPER_MODEL_SIZE}",
        f"Whisper model directory: {WHISPER_MODEL_DIRECTORY}",
        f"Requested device: {WHISPER_DEVICE}",
        f"Requested compute type: {WHISPER_COMPUTE_TYPE}",
        f"Language: {WHISPER_LANGUAGE}",
        f"Min segment duration: {MIN_SEGMENT_SEC}",
        f"Target segment duration: {TARGET_SEGMENT_SEC}",
        f"Max segment duration: {MAX_SEGMENT_SEC}",
        f"Minimum preferred word pause: {MIN_PREFERRED_WORD_PAUSE_SEC}",
    ]
    for audio_path, text_path in pairs:
        try:
            process_pair(audio_path, text_path, model, mfa_output_root, debug_output_root, summary_lines)
            print(f"Finished: {text_path.stem}", flush=True)
        except Exception as exc:
            print(f"ERROR while processing {text_path.stem}: {exc}")
            summary_lines.append(f"\n=== ERROR: {text_path.stem} ===")
            summary_lines.append(str(exc))
            summary_lines.append(traceback.format_exc())
    logs_folder = debug_output_root / "logs"
    logs_folder.mkdir(parents=True, exist_ok=True)
    (logs_folder / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\nDone.", flush=True)
    print(f"MFA input written to:  {mfa_output_root}", flush=True)
    print(f"Debug output written to: {debug_output_root}", flush=True)
    print(f"Run summary:       {logs_folder / 'run_summary.txt'}", flush=True)


if __name__ == "__main__":
    main()
    winsound.Beep(600, 500)
    winsound.Beep(300, 1200)
    winsound.Beep(600, 500)
    winsound.Beep(300, 1300)
