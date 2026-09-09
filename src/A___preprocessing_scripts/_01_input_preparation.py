"""
_01_input_preparation.py
===============================

Copyright © 2026 Till Preidt (GitHub: MrNemesis98).
This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License
(CC BY-NC 4.0). Reuse and adaptation are permitted for non-commercial purposes,
provided appropriate credit is given.

Processing behaviour:
    - Audio: WAV, mono, 16 kHz, signed 16-bit PCM via FFmpeg
    - Text: Unicode NFC normalization, number conversion, punctuation cleanup,
      whitespace normalization, and lowercase output for polish_mfa

Paths are obtained from:
    src/_00_config.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tkinter as tk
import unicodedata
from pathlib import Path
from tkinter import filedialog, messagebox

from num2words import num2words


# =============================================================================
# Load project configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import _00_config as config  # noqa: E402


# =============================================================================
# Script-specific settings
# =============================================================================

SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]

AUDIO_CHANNELS = "1"
AUDIO_SAMPLE_RATE = "16000"
AUDIO_SAMPLE_FORMAT = "s16"

LANGUAGE = "pl"
USE_UPPERCASE = False  # Keep False for polish_mfa for now.


# =============================================================================
# Folder selection
# =============================================================================

def select_input_folder() -> Path | None:
    """Select a folder inside the configured raw-input directory."""
    root = tk.Tk()
    root.withdraw()

    selected_folder = filedialog.askdirectory(
        title="Select raw-input folder containing audio and/or TXT files",
        initialdir=str(config.RAW_INPUT_DIR),
    )

    root.destroy()

    if not selected_folder:
        return None

    return Path(selected_folder)


def determine_output_folder(input_dir: Path) -> Path:
    """
    Mirror the selected raw-input subfolder below PREPARED_WHISPER_INPUT_DIR.

    Example:
        RAW_INPUT_DIR/training/book_01
        -> PREPARED_WHISPER_INPUT_DIR/training/book_01
    """
    try:
        relative_path = input_dir.resolve().relative_to(
            config.RAW_INPUT_DIR.resolve()
        )
    except ValueError as exc:
        raise ValueError(
            "The selected folder must be located inside the configured "
            f"raw-input directory:\n{config.RAW_INPUT_DIR}"
        ) from exc

    return config.PREPARED_WHISPER_INPUT_DIR / relative_path


# =============================================================================
# Audio preprocessing
# =============================================================================

def convert_audio_file(input_path: Path, output_path: Path) -> None:
    """Convert one supported audio file to mono, 16-kHz, 16-bit PCM WAV."""
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        AUDIO_CHANNELS,
        "-ar",
        AUDIO_SAMPLE_RATE,
        "-sample_fmt",
        AUDIO_SAMPLE_FORMAT,
        str(output_path),
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def preprocess_audio_files(
    input_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Process all directly contained supported audio files."""
    audio_files = [
        file
        for file in input_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_AUDIO_FORMATS
    ]

    converted_files: list[str] = []

    for file_counter, audio_file in enumerate(audio_files, start=1):
        print(f"Processing audio file nr. {file_counter}: {audio_file.name}")

        output_file = output_dir / f"{audio_file.stem}.wav"
        convert_audio_file(audio_file, output_file)
        converted_files.append(audio_file.name)

    return converted_files


# =============================================================================
# Transcript preprocessing
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize Unicode and replace typographic punctuation."""
    text = unicodedata.normalize("NFC", text)

    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "—": " ",
        "–": " ",
        "-": " ",
        "…": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def replace_numbers(text: str) -> str:
    """Replace standalone integers with Polish number words."""

    def convert(match: re.Match[str]) -> str:
        number = match.group()

        try:
            return num2words(int(number), lang=LANGUAGE)
        except Exception:
            return number

    return re.sub(r"\b\d+\b", convert, text)


def clean_for_mfa(text: str) -> str:
    """Apply the original transcript-cleaning rules."""
    text = normalize_text(text)
    text = replace_numbers(text)

    # Remove punctuation while retaining Polish letters.
    text = re.sub(
        r"[^\w\sąćęłńóśźżĄĆĘŁŃÓŚŹŻ]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if USE_UPPERCASE:
        return text.upper()

    return text.lower()


def preprocess_text_file(input_path: Path, output_path: Path) -> None:
    """Clean one UTF-8 transcript and write the prepared TXT file."""
    text = input_path.read_text(encoding="utf-8", errors="replace")
    cleaned_text = clean_for_mfa(text)
    output_path.write_text(cleaned_text, encoding="utf-8")


def preprocess_text_files(
    input_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Process all directly contained TXT files."""
    txt_files = [
        file
        for file in input_dir.iterdir()
        if file.is_file() and file.suffix.lower() == ".txt"
    ]

    processed_files: list[str] = []

    for file_counter, txt_file in enumerate(txt_files, start=1):
        print(f"Processing TXT file nr. {file_counter}: {txt_file.name}")

        output_file = output_dir / txt_file.name
        preprocess_text_file(txt_file, output_file)
        processed_files.append(txt_file.name)

    return processed_files


# =============================================================================
# Main program
# =============================================================================

def main() -> None:
    """Select one raw-input folder and prepare its audio and TXT files."""
    input_dir = select_input_folder()

    if input_dir is None:
        print("No input folder selected. Program cancelled.")
        return

    try:
        output_dir = determine_output_folder(input_dir)
    except ValueError as error:
        messagebox.showerror("Invalid input folder", str(error))
        print(error)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [
        file
        for file in input_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_AUDIO_FORMATS
    ]
    txt_files = [
        file
        for file in input_dir.iterdir()
        if file.is_file() and file.suffix.lower() == ".txt"
    ]

    if not audio_files and not txt_files:
        messagebox.showwarning(
            "No supported files found",
            "The selected folder contains no supported audio files "
            "and no TXT files.",
        )
        return

    messagebox.showinfo(
        "Input preparation initiated",
        f"Selected input folder:\n{input_dir}\n\n"
        f"Output folder:\n{output_dir}\n\n"
        f"Audio files found: {len(audio_files)}\n"
        f"TXT files found: {len(txt_files)}",
    )

    converted_audio_files = preprocess_audio_files(input_dir, output_dir)
    processed_text_files = preprocess_text_files(input_dir, output_dir)

    audio_summary = (
        "\n".join(converted_audio_files)
        if converted_audio_files
        else "No audio files processed."
    )
    text_summary = (
        "\n".join(processed_text_files)
        if processed_text_files
        else "No TXT files processed."
    )

    messagebox.showinfo(
        "Process finished with exit code 0",
        f"Converted audio files:\n{audio_summary}\n\n"
        f"Processed TXT files:\n{text_summary}\n\n"
        f"Processed files saved to:\n{output_dir}",
    )


if __name__ == "__main__":
    main()
