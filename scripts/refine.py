"""On-demand frame refinement: re-extract dense high-res frames around a timestamp.

Usage:
    python scripts/refine.py <work_dir> --timestamp 2:13 [--window 5] [--resolution max] [--fps 4]

Outputs paths of refined frames (for Claude to Read) and their OCR text.
Exit 0 on success.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
_env = Path.home() / ".config" / "watch" / ".env"
if _env.exists():
    load_dotenv(_env)


def _parse_timestamp(ts: str) -> float:
    """Parse MM:SS or SS or HH:MM:SS into seconds."""
    parts = ts.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    else:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def _format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = seconds % 60
    return f"{m}:{s:05.2f}"


def _find_video_file(work_dir: Path) -> Path | None:
    """Find the video file in the work dir's download folder."""
    dl_dir = work_dir / "download"
    if not dl_dir.exists():
        return None
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"):
        candidates = list(dl_dir.glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    return None


def _find_video_from_db(work_dir: Path) -> tuple[Path | None, str | None]:
    """Look up source_url from DB via watch_meta.json in the work dir."""
    meta_file = work_dir / "watch_meta.json"
    if not meta_file.exists():
        return None, None
    import json
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    return None, meta.get("source_url")


def extract_refined_frames(video_path: Path, work_dir: Path,
                           timestamp: float, window: float = 5,
                           resolution: int | str = 1920, fps: float = 4) -> list[dict]:
    """Extract dense frames around timestamp ± window seconds.

    Returns list of {path, timestamp_str, timestamp_seconds}.
    """
    start = max(0.0, timestamp - window)
    end = timestamp + window

    ts_label = f"{int(timestamp)}"
    output_dir = work_dir / "refined" / ts_label
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolution: "max" means use 1920, otherwise int
    res_val = 1920 if str(resolution) == "max" else int(resolution)
    scale_filter = f"fps={fps},scale={res_val}:-2"

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(video_path),
        "-vf", scale_filter,
        "-q:v", "2",
        str(output_dir / "frame_%04d.jpg"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    # Build frame list with absolute timestamps
    frames = sorted(output_dir.glob("frame_*.jpg"))
    frame_data = []
    for i, fp in enumerate(frames):
        abs_ts = start + (i / fps)
        frame_data.append({
            "path": str(fp),
            "timestamp_seconds": abs_ts,
            "timestamp_str": _format_timestamp(abs_ts),
        })
    return frame_data


def main() -> int:
    ap = argparse.ArgumentParser(prog="refine")
    ap.add_argument("work_dir", help="Working directory from watch.py")
    ap.add_argument("--timestamp", required=True, help="Timestamp to zoom into (MM:SS or SS)")
    ap.add_argument("--window", type=float, default=5, help="Seconds before/after timestamp (default 5)")
    ap.add_argument("--resolution", default="1920", help="Frame width in px or 'max' (default 1920)")
    ap.add_argument("--fps", type=float, default=4, help="Frames per second for dense extraction (default 4)")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        print(f"[refine] Work dir not found: {work_dir}", file=sys.stderr)
        return 1

    timestamp = _parse_timestamp(args.timestamp)

    # Find video
    video_path = _find_video_file(work_dir)
    source_url = None
    if video_path is None:
        _, source_url = _find_video_from_db(work_dir)
        if source_url:
            print(f"[refine] Local video not found. Source URL: {source_url}", file=sys.stderr)
            print(f"[refine] To re-download: re-run watch.py with --out-dir {work_dir}", file=sys.stderr)
        else:
            print(f"[refine] No video found in {work_dir}/download/", file=sys.stderr)
        return 1

    print(f"[refine] Using video: {video_path}", file=sys.stderr)
    print(f"[refine] Extracting dense frames around {args.timestamp} ± {args.window}s at {args.fps}fps/{args.resolution}px…", file=sys.stderr)

    frames = extract_refined_frames(video_path, work_dir, timestamp, args.window, args.resolution, args.fps)

    if not frames:
        print("[refine] No frames extracted. Check timestamp is within video duration.", file=sys.stderr)
        return 1

    # Run OCR on refined frames
    from ocr import run_ocr_pipeline
    frame_paths = [Path(f["path"]) for f in frames]
    mode = "chart"  # refine is almost always used in chart context
    ocr_results = run_ocr_pipeline(frame_paths, mode=mode)

    # Merge OCR into frame data
    ocr_by_path = {r["path"]: r for r in ocr_results}
    for frame in frames:
        ocr = ocr_by_path.get(frame["path"], {})
        frame["ocr_text"] = ocr.get("text", "")
        frame["ocr_score"] = ocr.get("relevance_score", 0)

    # Output for Claude to Read
    print(f"\nRefined frames around {args.timestamp} (window ±{args.window}s):")
    for frame in frames:
        print(f"- `{frame['path']}` (t={frame['timestamp_str']})")
        if frame["ocr_text"].strip():
            preview = frame["ocr_text"][:200]
            print(f"  OCR: {preview}{'…' if len(frame['ocr_text']) > 200 else ''}")

    print(f"\nRead each frame path above with the Read tool to view the refined images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
