"""
_03_mfa_batch_manager.py
===============================

Copyright © 2026 Till Preidt (GitHub: MrNemesis98).
This work is licensed under the Creative Commons Attribution–NonCommercial 4.0 International License
(CC BY-NC 4.0). Reuse and adaptation are permitted for non-commercial purposes,
provided appropriate credit is given.

Storage-efficient sequential batch manager for Montreal Forced Aligner (MFA).

Only one temporary input batch and one temporary MFA output batch exist at a
 time. After a successful batch, its TextGrids are moved directly into the
final feature-extraction input folder and both temporary batch folders are
removed. WAV files remain only in the original MFA input directory. Failed or incomplete batches are retained for debugging.

MFA is always launched from the project-local environment and model root:
    <project_root>/env_mfa/
    <project_root>/models/00___preprocessing/montreal_forced_aligner/

There is no fallback to a global MFA installation.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import _00_config as config  # noqa: E402


# ============================================================
# USER SETTINGS
# ============================================================

DATASET_SUFFIX = "training"  # training, validation, test
BATCH_SIZE = 100

ACOUSTIC_MODEL = config.MFA_ACOUSTIC_MODEL_NAME
DICTIONARY = config.MFA_DICTIONARY_NAME
NUM_JOBS = 8
USE_SINGLE_SPEAKER = True
USE_CLEAN = True
BEAM: Optional[int] = 100           # debug: 30000
RETRY_BEAM: Optional[int] = 400     # debug: 120000

# WAV files remain exclusively in the MFA input directory. The Praat
# feature-extraction script reads WAVs there and TextGrids from the final
# MFA output directory.

# Resume behavior.
SKIP_COMPLETED_PAIRS = True
STOP_ON_FIRST_FAILED_BATCH = False

# Storage behavior.
DELETE_SUCCESSFUL_TEMP_BATCHES = True
KEEP_FAILED_TEMP_BATCHES = True


@dataclass(frozen=True)
class FilePair:
    stem: str
    wav: Path
    txt: Path


@dataclass
class BatchResult:
    batch_name: str
    expected_pairs: int
    textgrids_moved: int
    missing_textgrids: int
    return_code: int
    status: str
    runtime_seconds: float
    temp_input_dir: Path
    temp_output_dir: Path
    log_file: Path
    warning: str = ""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_print(text: str = "") -> None:
    print(text, flush=True)


def chunked(items: Sequence[FilePair], size: int) -> Iterable[List[FilePair]]:
    if size <= 0:
        raise ValueError("BATCH_SIZE must be greater than zero.")
    for start in range(0, len(items), size):
        yield list(items[start:start + size])


def get_dataset_paths() -> Tuple[Path, Path, Path, Path, Path]:
    suffix = config.validate_dataset_suffix(DATASET_SUFFIX)
    mfa_input_dir = config.dataset_directory(config.MFA_INPUT_DIR, suffix)
    mfa_debug_dir = config.dataset_directory(config.MFA_DEBUG_DIR, suffix)
    temp_batches_root = mfa_debug_dir / "temporary_batches"
    temp_outputs_root = mfa_debug_dir / "temporary_outputs"
    logs_dir = mfa_debug_dir / "batch_logs"
    feature_input_dir = config.dataset_directory(
        config.FEATURE_EXTRACTION_INPUT_DIR, suffix
    )
    return (
        mfa_input_dir,
        temp_batches_root,
        temp_outputs_root,
        logs_dir,
        feature_input_dir,
    )


def validate_runtime() -> List[str]:
    errors: List[str] = []
    if not config.MFA_ENV_DIR.is_dir():
        errors.append(f"Project-local MFA environment is missing: {config.MFA_ENV_DIR}")
    if not config.MFA_ENV_PYTHON_EXE.is_file():
        errors.append(f"Project-local MFA Python is missing: {config.MFA_ENV_PYTHON_EXE}")
    if not config.MFA_ENV_MFA_EXE.is_file():
        errors.append(f"Project-local MFA executable is missing: {config.MFA_ENV_MFA_EXE}")
    try:
        config.MFA_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errors.append(f"MFA root is not writable: {config.MFA_ROOT_DIR} ({exc})")
    return errors


def find_file_pairs(input_dir: Path) -> Tuple[List[FilePair], List[Path], List[Path]]:
    wavs: Dict[str, Path] = {p.stem: p for p in sorted(input_dir.glob("*.wav"))}
    txts: Dict[str, Path] = {p.stem: p for p in sorted(input_dir.glob("*.txt"))}
    common = sorted(set(wavs) & set(txts))
    pairs = [FilePair(stem=s, wav=wavs[s], txt=txts[s]) for s in common]
    wav_without_txt = [wavs[s] for s in sorted(set(wavs) - set(txts))]
    txt_without_wav = [txts[s] for s in sorted(set(txts) - set(wavs))]
    return pairs, wav_without_txt, txt_without_wav


def pair_is_complete(pair: FilePair, final_dir: Path) -> bool:
    """Return True when the final TextGrid already exists."""
    tg_upper = final_dir / f"{pair.stem}.TextGrid"
    tg_lower = final_dir / f"{pair.stem}.textgrid"
    return tg_upper.is_file() or tg_lower.is_file()


def prepare_temporary_batch(
    batch_name: str,
    pairs: Sequence[FilePair],
    temp_batches_root: Path,
) -> Path:
    batch_dir = temp_batches_root / batch_name
    remove_tree(batch_dir)
    ensure_dir(batch_dir)

    for pair in pairs:
        shutil.copy2(pair.wav, batch_dir / pair.wav.name)
        shutil.copy2(pair.txt, batch_dir / pair.txt.name)

    manifest = batch_dir / "_batch_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["stem", "source_wav", "source_txt"])
        for pair in pairs:
            writer.writerow([pair.stem, str(pair.wav), str(pair.txt)])

    return batch_dir


def build_mfa_arguments(input_dir: Path, output_dir: Path) -> List[str]:
    args = [
        "align",
        str(input_dir),
        DICTIONARY,
        ACOUSTIC_MODEL,
        str(output_dir),
    ]
    if USE_CLEAN:
        args.append("--clean")
    if USE_SINGLE_SPEAKER:
        args.append("--single_speaker")
    if BEAM is not None:
        args.extend(["--beam", str(BEAM)])
    if RETRY_BEAM is not None:
        args.extend(["--retry_beam", str(RETRY_BEAM)])
    if NUM_JOBS > 0:
        args.extend(["-j", str(NUM_JOBS)])
    return args


def build_mfa_command(input_dir: Path, output_dir: Path) -> List[str]:
    launcher = (
        "from montreal_forced_aligner.command_line.mfa "
        "import mfa_cli; mfa_cli()"
    )
    return [
        str(config.MFA_ENV_PYTHON_EXE),
        "-c",
        launcher,
        *build_mfa_arguments(input_dir, output_dir),
    ]


def run_mfa(
    batch_name: str,
    temp_input_dir: Path,
    temp_output_dir: Path,
    logs_dir: Path,
) -> Tuple[int, float, Path]:
    ensure_dir(logs_dir)
    remove_tree(temp_output_dir)
    ensure_dir(temp_output_dir)

    log_file = logs_dir / f"{batch_name}_mfa_log.txt"
    command = build_mfa_command(temp_input_dir, temp_output_dir)
    environment = dict(os.environ)
    environment["MFA_ROOT_DIR"] = str(config.MFA_ROOT_DIR)

    start = time.time()
    with log_file.open("w", encoding="utf-8", errors="replace") as log:
        log.write("MFA sequential batch-manager log\n")
        log.write(f"Batch: {batch_name}\n")
        log.write(f"Started: {timestamp()}\n")
        log.write(f"MFA environment: {config.MFA_ENV_DIR}\n")
        log.write(f"MFA Python: {config.MFA_ENV_PYTHON_EXE}\n")
        log.write(f"MFA root: {config.MFA_ROOT_DIR}\n")
        log.write(f"Dictionary: {DICTIONARY}\n")
        log.write(f"Acoustic model: {ACOUSTIC_MODEL}\n")
        log.write(f"Temporary input: {temp_input_dir}\n")
        log.write(f"Temporary output: {temp_output_dir}\n")
        log.write("\n--- MFA OUTPUT ---\n\n")
        log.flush()

        try:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=environment,
            )
            return_code = process.returncode
        except OSError as exc:
            log.write(f"\nERROR while launching project-local MFA: {exc}\n")
            return_code = -1

        log.write("\n--- END MFA OUTPUT ---\n")
        log.write(f"Finished: {timestamp()}\n")
        log.write(f"Return code: {return_code}\n")

    return return_code, time.time() - start, log_file


def find_textgrids(directory: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    if not directory.exists():
        return found
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".textgrid":
            found[path.stem.casefold()] = path
    return found


def move_batch_outputs(
    pairs: Sequence[FilePair],
    temp_output_dir: Path,
    final_dir: Path,
) -> Tuple[int, List[str]]:
    """Move TextGrids directly to the final folder without copying WAVs."""
    ensure_dir(final_dir)
    textgrids = find_textgrids(temp_output_dir)
    moved = 0
    missing: List[str] = []

    for pair in pairs:
        source_tg = textgrids.get(pair.stem.casefold())
        if source_tg is None:
            missing.append(pair.stem)
            continue

        target_tg = final_dir / f"{pair.stem}.TextGrid"
        if target_tg.exists():
            target_tg.unlink()
        shutil.move(str(source_tg), str(target_tg))
        moved += 1

    return moved, missing


def process_batch(
    batch_index: int,
    total_batches: int,
    pairs: Sequence[FilePair],
    temp_batches_root: Path,
    temp_outputs_root: Path,
    logs_dir: Path,
    final_dir: Path,
) -> BatchResult:
    batch_name = f"batch_{batch_index:04d}"
    temp_input_dir = temp_batches_root / batch_name
    temp_output_dir = temp_outputs_root / batch_name

    safe_print(f"\n========== {batch_name} ({batch_index}/{total_batches}) ==========")
    safe_print(f"Preparing {len(pairs)} pair(s)...")
    prepare_temporary_batch(batch_name, pairs, temp_batches_root)

    return_code, runtime_seconds, log_file = run_mfa(
        batch_name,
        temp_input_dir,
        temp_output_dir,
        logs_dir,
    )

    moved = 0
    missing = [pair.stem for pair in pairs]
    if return_code == 0:
        moved, missing = move_batch_outputs(pairs, temp_output_dir, final_dir)

    expected = len(pairs)
    if return_code != 0:
        status = "FAILED_RETURN_CODE"
        warning = f"MFA returned code {return_code}. Check {log_file}."
    elif moved == expected:
        status = "OK"
        warning = ""
    elif moved == 0:
        status = "FAILED_NO_TEXTGRIDS"
        warning = "MFA returned successfully but produced no matching TextGrids."
    else:
        status = "WARNING_INCOMPLETE"
        shown = ", ".join(missing[:10])
        warning = f"Moved only {moved}/{expected} TextGrids. Missing: {shown}"
        if len(missing) > 10:
            warning += f" ... and {len(missing) - 10} more"

    successful = status == "OK"
    if successful and DELETE_SUCCESSFUL_TEMP_BATCHES:
        remove_tree(temp_input_dir)
        remove_tree(temp_output_dir)
    elif not successful and not KEEP_FAILED_TEMP_BATCHES:
        remove_tree(temp_input_dir)
        remove_tree(temp_output_dir)

    safe_print(
        f"{batch_name}: {status} | TextGrids moved {moved}/{expected} | "
        f"{runtime_seconds / 60:.2f} min"
    )
    if warning:
        safe_print(f"WARNING: {warning}")

    return BatchResult(
        batch_name=batch_name,
        expected_pairs=expected,
        textgrids_moved=moved,
        missing_textgrids=len(missing),
        return_code=return_code,
        status=status,
        runtime_seconds=runtime_seconds,
        temp_input_dir=temp_input_dir,
        temp_output_dir=temp_output_dir,
        log_file=log_file,
        warning=warning,
    )


def write_reports(
    results: Sequence[BatchResult],
    logs_dir: Path,
    wav_without_txt: Sequence[Path],
    txt_without_wav: Sequence[Path],
) -> Path:
    ensure_dir(logs_dir)
    summary = logs_dir / "mfa_batch_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow([
            "batch", "status", "expected_pairs", "textgrids_moved",
            "missing_textgrids", "return_code", "runtime_seconds",
            "runtime_minutes", "temporary_input_dir", "temporary_output_dir",
            "log_file", "warning",
        ])
        for r in results:
            writer.writerow([
                r.batch_name, r.status, r.expected_pairs, r.textgrids_moved,
                r.missing_textgrids, r.return_code,
                round(r.runtime_seconds, 2), round(r.runtime_seconds / 60, 2),
                str(r.temp_input_dir), str(r.temp_output_dir),
                str(r.log_file), r.warning,
            ])

    missing_report = logs_dir / "mfa_missing_pairs_report.txt"
    with missing_report.open("w", encoding="utf-8") as handle:
        handle.write("Missing WAV/TXT pair report\n")
        handle.write(f"Created: {timestamp()}\n\n")
        handle.write(f"WAV without TXT: {len(wav_without_txt)}\n")
        for path in wav_without_txt:
            handle.write(f"  {path}\n")
        handle.write(f"\nTXT without WAV: {len(txt_without_wav)}\n")
        for path in txt_without_wav:
            handle.write(f"  {path}\n")
    return summary


def print_menu() -> None:
    safe_print("\n_03_mfa_batch_manager.py")
    safe_print("=======================")
    safe_print("[1] Run complete sequential MFA processing")
    safe_print("[2] Validate WAV/TXT input pairs only")
    safe_print("[q] Quit")


def ask_choice() -> str:
    while True:
        print_menu()
        choice = input("Choose option: ").strip().lower()
        if choice in {"1", "2", "q"}:
            return choice
        safe_print("Invalid selection. Choose 1, 2, or q.")


def main() -> int:
    try:
        (
            mfa_input_dir,
            temp_batches_root,
            temp_outputs_root,
            logs_dir,
            feature_input_dir,
        ) = get_dataset_paths()
    except ValueError as exc:
        safe_print(f"ERROR: {exc}")
        return 1

    safe_print("VERSION: PROJECT_LOCAL_MFA_SEQUENTIAL_v1")
    safe_print(f"Dataset split : {DATASET_SUFFIX}")
    safe_print(f"Project root  : {config.PROJECT_ROOT}")
    safe_print(f"MFA input     : {mfa_input_dir}")
    safe_print(f"Final output  : {feature_input_dir}")
    safe_print(f"MFA env       : {config.MFA_ENV_DIR}")
    safe_print(f"MFA Python    : {config.MFA_ENV_PYTHON_EXE}")
    safe_print(f"MFA root      : {config.MFA_ROOT_DIR}")
    safe_print(f"Dictionary    : {DICTIONARY}")
    safe_print(f"Acoustic model: {ACOUSTIC_MODEL}")

    runtime_errors = validate_runtime()
    if runtime_errors:
        safe_print("\nERROR: The project-local MFA runtime is not ready:")
        for error in runtime_errors:
            safe_print(f"  - {error}")
        return 1

    if not mfa_input_dir.is_dir():
        safe_print(f"ERROR: MFA input directory does not exist: {mfa_input_dir}")
        return 1

    pairs, wav_without_txt, txt_without_wav = find_file_pairs(mfa_input_dir)
    safe_print(f"\nValid WAV/TXT pairs: {len(pairs)}")
    safe_print(f"WAV without TXT    : {len(wav_without_txt)}")
    safe_print(f"TXT without WAV    : {len(txt_without_wav)}")

    choice = ask_choice()
    if choice == "q":
        return 0
    if choice == "2":
        report = write_reports([], logs_dir, wav_without_txt, txt_without_wav)
        safe_print(f"Validation report written: {report}")
        return 0
    if not pairs:
        safe_print("ERROR: No valid WAV/TXT pairs found.")
        return 1

    ensure_dir(temp_batches_root)
    ensure_dir(temp_outputs_root)
    ensure_dir(logs_dir)
    ensure_dir(feature_input_dir)

    if SKIP_COMPLETED_PAIRS:
        pending = [p for p in pairs if not pair_is_complete(p, feature_input_dir)]
        skipped = len(pairs) - len(pending)
    else:
        pending = pairs
        skipped = 0

    safe_print(f"\nAlready complete/skipped: {skipped}")
    safe_print(f"Pairs to process        : {len(pending)}")
    if not pending:
        safe_print("All pairs already have final TextGrid outputs.")
        return 0

    batches = list(chunked(pending, BATCH_SIZE))
    total_batches = len(batches)
    results: List[BatchResult] = []

    safe_print(
        f"\nStarting sequential processing: {total_batches} batch(es), "
        f"maximum {BATCH_SIZE} pair(s) each."
    )
    safe_print("Only one temporary input and output batch exist at a time.")

    for index, batch_pairs in enumerate(batches, start=1):
        result = process_batch(
            index,
            total_batches,
            batch_pairs,
            temp_batches_root,
            temp_outputs_root,
            logs_dir,
            feature_input_dir,
        )
        results.append(result)

        if STOP_ON_FIRST_FAILED_BATCH and result.status != "OK":
            safe_print(
                "\nStopping after the first failed or incomplete batch. "
                "Its temporary directories were retained for debugging."
            )
            break

    report = write_reports(results, logs_dir, wav_without_txt, txt_without_wav)
    successful = sum(r.status == "OK" for r in results)
    problematic = len(results) - successful

    safe_print("\n==================================================")
    safe_print("MFA processing finished")
    safe_print(f"Successful batches : {successful}")
    safe_print(f"Problematic batches: {problematic}")
    safe_print(f"Summary report     : {report}")
    safe_print(f"Final output       : {feature_input_dir}")
    safe_print("==================================================")

    return 0 if problematic == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        safe_print("\nInterrupted by user.")
        raise SystemExit(130)
