"""OCR pre-pass with multi-signal relevance scoring.

Tesseract runs on every extracted frame in parallel (ProcessPoolExecutor).
Each frame produces:
  - A .txt sidecar file on disk
  - A result dict with text, confidence, char_count, novelty, relevance_score, included_in_prompt

Usage:
    from ocr import run_ocr_pipeline
    results = run_ocr_pipeline(frame_paths, mode="chart")
"""
from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pytesseract
from PIL import Image

# Tesseract binary path for Windows; no-op if tesseract is already on PATH
_TESSERACT_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(_TESSERACT_DEFAULT):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_DEFAULT

# Patterns that boost relevance in chart mode
_CHART_PATTERNS = re.compile(
    r"\$?\d+\.\d{2}"          # prices: $420.50
    r"|[\+\-]?\d+\.?\d*%"     # percentages: +3.5%
    r"|\b(?:RSI|MACD|EMA|SMA|VWAP|ATR|BB|ADX|OBV|MFI|STOCH)\b",
    re.IGNORECASE
)

RELEVANCE_THRESHOLD = 5.0  # frames below this score are excluded from the prompt


def _ocr_single_frame(frame_path_str: str) -> dict:
    """Worker: runs in a subprocess via ProcessPoolExecutor."""
    import os
    import pytesseract
    from PIL import Image

    _TESSERACT_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_TESSERACT_DEFAULT):
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_DEFAULT

    path = Path(frame_path_str)
    try:
        image = Image.open(path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confs = [c for c in data["conf"] if c >= 0]
        words = [w for w, c in zip(data["text"], data["conf"]) if c >= 0 and w.strip()]
        text = " ".join(words)
        confidence_avg = sum(confs) / len(confs) if confs else 0.0
        char_count = len(text)
    except Exception as e:
        text = ""
        confidence_avg = 0.0
        char_count = 0

    return {
        "path": frame_path_str,
        "text": text,
        "confidence_avg": confidence_avg,
        "char_count": char_count,
    }


def _jaccard_novelty(curr_text: str, prev_text: str) -> float:
    """Returns 0.0 (identical) to 1.0 (completely novel)."""
    curr_words = set(curr_text.lower().split())
    prev_words = set(prev_text.lower().split())
    if not curr_words and not prev_words:
        return 0.0
    intersection = len(curr_words & prev_words)
    union = len(curr_words | prev_words)
    if union == 0:
        return 0.0
    return 1.0 - (intersection / union)


def _compute_mode_weight(text: str, mode: str) -> float:
    if mode == "chart" and _CHART_PATTERNS.search(text):
        return 2.0
    return 1.0


def run_ocr_parallel(frame_paths: list[Path], max_workers: int | None = None) -> list[dict]:
    """Run OCR on all frames in parallel. Returns results in input order."""
    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = max(1, cpu - 1)

    path_strs = [str(p) for p in frame_paths]

    # Use a dict to preserve ordering since as_completed doesn't guarantee it
    results_map: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_ocr_single_frame, p): p for p in path_strs}
        for future in as_completed(future_to_path):
            result = future.result()
            results_map[result["path"]] = result

    # Return in the original order
    return [results_map[p] for p in path_strs]


def compute_relevance_scores(ocr_results: list[dict], mode: str) -> list[dict]:
    """Add novelty, relevance_score, and included_in_prompt to each result in-place."""
    for i, r in enumerate(ocr_results):
        prev_text = ocr_results[i - 1]["text"] if i > 0 else ""
        r["novelty"] = _jaccard_novelty(r["text"], prev_text)
        mode_weight = _compute_mode_weight(r["text"], mode)
        base = r["char_count"] * (r["confidence_avg"] / 100.0)
        r["relevance_score"] = base * r["novelty"] * mode_weight
        r["included_in_prompt"] = r["relevance_score"] >= RELEVANCE_THRESHOLD
    return ocr_results


def write_sidecars(ocr_results: list[dict]) -> None:
    """Write a .txt sidecar next to each frame JPG with the OCR text."""
    for r in ocr_results:
        sidecar_path = Path(r["path"]).with_suffix(".txt")
        sidecar_path.write_text(r["text"], encoding="utf-8")


def run_ocr_pipeline(frame_paths: list[Path], mode: str = "regular") -> list[dict]:
    """Full OCR pipeline: parallel OCR → relevance scoring → sidecars.

    Returns list of result dicts with all fields populated.
    """
    print(f"[watch] OCR: processing {len(frame_paths)} frames (mode={mode})…", file=sys.stderr)
    results = run_ocr_parallel(frame_paths)
    compute_relevance_scores(results, mode)
    write_sidecars(results)
    included = sum(1 for r in results if r["included_in_prompt"])
    print(f"[watch] OCR: {included}/{len(results)} frames above relevance threshold", file=sys.stderr)
    return results
