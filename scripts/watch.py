#!/usr/bin/env python3
"""/watch entry point: download video, extract frames, parse transcript.

Prints a markdown report to stdout listing frame paths + transcript. Claude
then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# Load .env from ~/.config/watch/.env before any local imports that read env
from dotenv import load_dotenv  # noqa: E402
_env_path = Path.home() / ".config" / "watch" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from download import download, is_url  # noqa: E402
from frames import MAX_FPS, auto_fps, auto_fps_focus, extract, format_time, get_metadata, parse_time  # noqa: E402
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
from whisper import load_api_key, transcribe_video  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="watch",
        description="Download a video, extract auto-scaled frames, and surface the transcript.",
    )
    ap.add_argument("source", nargs="?", default=None, help="Video URL or local file path")
    ap.add_argument("--max-frames", type=int, default=None, help="Cap on frame count")
    ap.add_argument("--resolution", type=int, default=None, help="Frame width in pixels")
    ap.add_argument("--fps", type=float, default=None, help="Override auto-fps")
    ap.add_argument("--start", type=str, default=None, help="Range start (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--end", type=str, default=None, help="Range end (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--out-dir", type=str, default=None, help="Working directory (default: tmp)")
    ap.add_argument(
        "--no-whisper",
        action="store_true",
        help="Disable Whisper fallback. Report frames-only if no captions available.",
    )
    ap.add_argument(
        "--whisper",
        choices=["groq", "openai"],
        default=None,
        help="Force a specific Whisper backend. Default: prefer Groq, fall back to OpenAI.",
    )
    ap.add_argument(
        "--mode",
        choices=["regular", "chart"],
        default="regular",
        help="Processing mode. chart uses higher resolution and more frames.",
    )
    ap.add_argument(
        "--keep",
        choices=["none", "transcript", "all"],
        default="none",
        help="Retention level for local raw files after 48h cleanup.",
    )
    ap.add_argument(
        "--provider",
        choices=["gemini", "claude", "auto"],
        default="auto",
        help="Force a specific vision model provider.",
    )
    ap.add_argument(
        "--health",
        action="store_true",
        help="Show system health dashboard and exit.",
    )
    ap.add_argument(
        "--doctor",
        action="store_true",
        help="Run sanity checks and explain any failures.",
    )
    args = ap.parse_args()

    # Handle observability commands (no source required)
    if args.health or args.doctor:
        try:
            from health import run_health, run_doctor
            if args.health:
                return run_health()
            else:
                return run_doctor()
        except ImportError as e:
            print(f"[watch] health module not available: {e}", file=sys.stderr)
            return 1

    if args.source is None:
        ap.error("source is required unless --health or --doctor is used")

    # Apply mode-based defaults (only if user didn't explicitly pass the flag)
    if args.mode == "chart":
        if args.resolution is None:
            args.resolution = 1920
        if args.max_frames is None:
            args.max_frames = 150
    else:  # regular
        if args.resolution is None:
            args.resolution = 512
        if args.max_frames is None:
            args.max_frames = 80

    # Persist mode/keep/provider into env so sub-scripts can read them
    os.environ["WATCH_MODE"] = args.mode
    os.environ["WATCH_KEEP"] = args.keep
    os.environ["WATCH_PROVIDER"] = args.provider

    if args.out_dir:
        work = Path(args.out_dir).expanduser().resolve()
    else:
        work = Path(tempfile.mkdtemp(prefix="watch-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[watch] working dir: {work}", file=sys.stderr)
    print(f"[watch] mode={args.mode} keep={args.keep} provider={args.provider}", file=sys.stderr)

    print(
        "[watch] downloading via yt-dlp…" if is_url(args.source) else "[watch] using local file…",
        file=sys.stderr,
    )
    dl = download(args.source, work / "download")
    video_path = dl["video_path"]

    meta = get_metadata(video_path)
    full_duration = meta["duration_seconds"]

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    if start_sec is not None and start_sec < 0:
        raise SystemExit("--start must be non-negative")
    if end_sec is not None and start_sec is not None and end_sec <= start_sec:
        raise SystemExit("--end must be greater than --start")
    if full_duration > 0 and start_sec is not None and start_sec >= full_duration:
        raise SystemExit(f"--start {start_sec:.1f}s is past end of video ({full_duration:.1f}s)")

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)
    focused = start_sec is not None or end_sec is not None

    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=args.max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=args.max_frames)
    if args.fps is not None:
        fps = min(args.fps, MAX_FPS)
        target = max(1, int(round(fps * effective_duration)))

    scope = (
        f"{format_time(effective_start)}-{format_time(effective_end)} ({effective_duration:.1f}s)"
        if focused else f"full {effective_duration:.1f}s"
    )
    print(f"[watch] extracting ~{target} frames at {fps:.3f} fps over {scope}…", file=sys.stderr)

    frames = extract(
        video_path,
        work / "frames",
        fps=fps,
        resolution=args.resolution,
        max_frames=args.max_frames,
        start_seconds=start_sec,
        end_seconds=end_sec,
    )

    # Run OCR pipeline — creates .txt sidecars next to each .jpg for answer.py
    ocr_results: list[dict] = []
    try:
        from ocr import run_ocr_pipeline
        frame_paths_for_ocr = [Path(f["path"]) for f in frames]
        print(f"[watch] running OCR on {len(frame_paths_for_ocr)} frames…", file=sys.stderr)
        ocr_results = run_ocr_pipeline(frame_paths_for_ocr, mode=args.mode)
        n_ocr_included = sum(1 for r in ocr_results if r.get("included_in_prompt"))
        print(f"[watch] OCR done: {n_ocr_included}/{len(ocr_results)} frames have relevant text", file=sys.stderr)
    except Exception as exc:
        print(f"[watch] OCR failed (non-fatal): {exc}", file=sys.stderr)

    transcript_segments: list[dict] = []
    transcript_text: str | None = None
    transcript_source: str | None = None
    if dl.get("subtitle_path"):
        try:
            all_segments = parse_vtt(dl["subtitle_path"])
            transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
            transcript_text = format_transcript(transcript_segments)
            transcript_source = "captions"
        except Exception as exc:
            print(f"[watch] subtitle parse failed: {exc}", file=sys.stderr)

    if not transcript_segments and not args.no_whisper:
        backend, api_key = load_api_key(args.whisper)
        if backend and api_key:
            try:
                all_segments, used_backend = transcribe_video(
                    video_path,
                    work / "audio.mp3",
                    backend=backend,
                    api_key=api_key,
                )
                transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
                transcript_text = format_transcript(transcript_segments)
                transcript_source = f"whisper ({used_backend})"
            except SystemExit as exc:
                print(f"[watch] whisper fallback failed: {exc}", file=sys.stderr)
        else:
            hint = (
                f"--whisper {args.whisper} was set but the matching API key is missing"
                if args.whisper else
                "no subtitles and no Whisper API key found"
            )
            setup_py = SCRIPT_DIR / "setup.py"
            print(
                f"[watch] {hint} — run `python {setup_py}` to enable the Whisper fallback",
                file=sys.stderr,
            )

    info = dl.get("info") or {}

    # Write watch_run.json so answer.py can call persist_all() after a successful vision call
    try:
        _frame_results = []
        for _i, _frame in enumerate(frames):
            _ocr = ocr_results[_i] if _i < len(ocr_results) else {}
            _frame_results.append({
                "path": _frame["path"],
                "timestamp_seconds": _frame["timestamp_seconds"],
                "text": _ocr.get("text", ""),
                "included_in_prompt": bool(_ocr.get("included_in_prompt", False)),
                "confidence_avg": _ocr.get("confidence_avg"),
                "relevance_score": _ocr.get("relevance_score"),
            })
        (work / "watch_run.json").write_text(
            json.dumps({
                "source_url": args.source,
                "title": info.get("title"),
                "creator": info.get("uploader"),
                "duration_seconds": full_duration,
                "mode": args.mode,
                "retention": args.keep,
                "frame_results": _frame_results,
                "transcript_segments": transcript_segments,
            }, indent=2),
            encoding="utf-8",
        )
    except Exception as _e:
        print(f"[watch] watch_run.json write failed (non-fatal): {_e}", file=sys.stderr)

    print()
    print("# watch: video report")
    print()
    print(f"- **Source:** {args.source}")
    if info.get("title"):
        print(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        print(f"- **Uploader:** {info['uploader']}")
    print(f"- **Duration:** {format_time(full_duration)} ({full_duration:.1f}s)")
    if focused:
        print(
            f"- **Focus range:** {format_time(effective_start)} → {format_time(effective_end)} "
            f"({effective_duration:.1f}s)"
        )
    if meta.get("width") and meta.get("height"):
        print(f"- **Resolution:** {meta['width']}x{meta['height']} ({meta.get('codec') or 'unknown codec'})")
    mode_label = "focused" if focused else "full"
    print(f"- **Frames:** {len(frames)} @ {fps:.3f} fps, {mode_label} mode (budget {target}, max {args.max_frames})")
    print(f"- **Frame size:** {args.resolution}px wide")
    if ocr_results:
        n_ocr_included = sum(1 for r in ocr_results if r.get("included_in_prompt"))
        print(f"- **OCR:** {n_ocr_included}/{len(ocr_results)} frames have relevant text (sidecars written)")
    print(f"- **Watch mode:** {args.mode} | **Provider:** {args.provider} | **Keep:** {args.keep}")
    if transcript_segments:
        in_range = " in range" if focused else ""
        print(
            f"- **Transcript:** {len(transcript_segments)} segments{in_range} "
            f"(via {transcript_source or 'captions'})"
        )
    else:
        print("- **Transcript:** none available")

    if not focused and full_duration > 600:
        mins = int(full_duration // 60)
        print()
        print(
            f"> **Warning:** This is a {mins}-minute video. Frame coverage is sparse at this length — "
            "accuracy degrades noticeably on anything over 10 minutes. For better results, "
            "re-run with `--start HH:MM:SS --end HH:MM:SS` to zoom into a specific section."
        )

    print()
    print("## Frames")
    print()
    print(f"Frames live at: `{work / 'frames'}`")
    print()
    print(
        "**Read each frame path below with the Read tool to view the image.** "
        "Frames are in chronological order; `t=MM:SS` is the absolute timestamp in the source video."
    )
    print()
    for frame in frames:
        print(f"- `{frame['path']}` (t={format_time(frame['timestamp_seconds'])})")

    print()
    print("## Transcript")
    print()
    if transcript_text:
        label = transcript_source or "captions"
        if focused:
            print(f"_Source: {label}. Filtered to {format_time(effective_start)} → {format_time(effective_end)}:_")
        else:
            print(f"_Source: {label}._")
        print()
        print("```")
        print(transcript_text)
        print("```")
    elif focused and dl.get("subtitle_path"):
        print(f"_No transcript lines fell inside {format_time(effective_start)} → {format_time(effective_end)}._")
    else:
        setup_py = SCRIPT_DIR / "setup.py"
        print(
            "_No transcript available — proceed with frames only. "
            "Captions were missing and the Whisper fallback was unavailable "
            "(no API key set, or `--no-whisper` was used). "
            f"Run `python {setup_py}` to enable Whisper, then re-run._"
        )

    print()
    print("---")
    print(f"_Work dir: `{work}` — mode={args.mode}, keep={args.keep}._")

    # Log telemetry (writer laptop only, best-effort)
    try:
        from telemetry import log_invocation
        _url = args.source.lower()
        _src_type = (
            "youtube" if ("youtube.com" in _url or "youtu.be" in _url) else
            "vimeo" if "vimeo.com" in _url else
            "tiktok" if "tiktok.com" in _url else
            "twitter" if ("twitter.com" in _url or "x.com" in _url) else
            "local" if not _url.startswith("http") else
            "other"
        )
        log_invocation(
            mode=args.mode,
            source_type=_src_type,
            duration_seconds=full_duration,
            provider=args.provider,
            n_frames=len(frames),
        )
    except Exception:
        pass

    # Run cleanup on old work dirs (best-effort, non-blocking)
    try:
        from cleanup import cleanup_work_dirs
        cleanup_work_dirs()
    except Exception:
        pass

    # Run sync-wiki --quick on each /watch invocation (catches edits made since last run)
    if os.environ.get("WATCH_IS_WRITER_LAPTOP", "false").lower() == "true":
        try:
            import subprocess
            subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "sync_wiki.py"), "--quick"],
                capture_output=True, timeout=10
            )
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
