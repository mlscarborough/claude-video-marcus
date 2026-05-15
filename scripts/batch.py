"""batch.py — process multiple videos in a single invocation.

Queues videos sequentially with per-video error isolation, quota pre-check,
and a combined summary report at the end.

CLI:
    python scripts/batch.py <url1> [url2 ...] [options]
    python scripts/batch.py --batch urls.txt [options]
    python scripts/batch.py https://youtube.com/@PatrickKenney --max-videos 10 --order newest

Options:
    --batch FILE       Plain text file, one URL per line (# comments ignored)
    --mode MODE        regular | chart  (default: regular)
    --delay N          Seconds between videos (default 5 — respects Gemini RPM)
    --max-workers N    Parallel workers (default 1 — sequential; max 3)
    --max-videos N     Cap on channel/playlist enumeration (default 20; 0 = unlimited)
    --order ORDER      newest | oldest | default (for channels; default: newest)
    --yes              Skip confirmation prompt (non-interactive / scripted)
    --question TEXT    Question to answer for each video (default: "Summarize this video.")
    --out-dir DIR      Parent directory for work dirs (default: system temp)
    --no-checkpoint    Disable resume-from-checkpoint (re-process all URLs)

Exit codes: 0 = all succeeded, 1 = one or more errors.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
_env = Path.home() / ".config" / "watch" / ".env"
if _env.exists():
    load_dotenv(_env)

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv"}

CHANNEL_CAP_DEFAULT = 20


# ── 16.5  Batch source detection and enumeration ──────────────────────────────

def _is_channel_url(url: str) -> bool:
    low = url.lower()
    return any(pat in low for pat in ["/@", "/channel/", "/user/", "/c/"])


def _is_playlist_url(url: str) -> bool:
    return "list=" in url and "youtube" in url.lower()


def _yt_dlp_enumerate(source: str, max_count: int = 0, order: str = "newest") -> list[str]:
    """Use yt-dlp --flat-playlist to list URLs without downloading.

    max_count=0 means no cap. order='oldest' reverses the playlist.
    """
    cmd = ["yt-dlp", "--flat-playlist", "--print", "url"]
    if order == "oldest":
        cmd.append("--playlist-reverse")
    if max_count > 0:
        cmd.extend(["--playlist-end", str(max_count)])
    cmd.append(source)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        urls = [u.strip() for u in result.stdout.splitlines() if u.strip()]
        return urls
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise RuntimeError(f"yt-dlp enumeration failed: {e}") from e


def _yt_dlp_get_title(source: str) -> str:
    """Fetch playlist/channel title via yt-dlp (best-effort)."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--flat-playlist", "--print", "%(playlist_title)s", "--playlist-end", "1", source],
            capture_output=True, text=True, timeout=30,
        )
        title = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        return title or source
    except Exception:
        return source


def _yt_dlp_count(source: str) -> int:
    """Count total videos in a channel/playlist without cap."""
    urls = _yt_dlp_enumerate(source, max_count=0)
    return len(urls)


def _print_video_list(urls: list[str], max_show: int = 5) -> None:
    for i, u in enumerate(urls[:max_show], 1):
        print(f"    {i}. {u}")
    if len(urls) > max_show:
        print(f"    ... ({len(urls) - max_show} more)")


def _confirm_batch(urls: list[str], source_label: str, yes: bool) -> list[str] | None:
    """Show what was found and ask for confirmation. Returns urls or None (abort)."""
    count = len(urls)
    print(f"\n[batch] {source_label}")
    print(f"[batch] Found {count} video(s).")
    _print_video_list(urls)

    if yes:
        print(f"[batch] --yes flag set — proceeding automatically.")
        return urls

    answer = input(f"\nProcess all {count} as a batch? [yes / no / list] -> ").strip().lower()
    if answer in ("yes", "y"):
        return urls
    elif answer == "list":
        for i, u in enumerate(urls, 1):
            print(f"  {i}. {u}")
        answer2 = input(f"\nProcess all {count}? [yes / no] -> ").strip().lower()
        if answer2 in ("yes", "y"):
            return urls
    print("[batch] Aborted.")
    return None


def _channel_cap_prompt(source: str, total: int, cap: int, order: str, yes: bool) -> list[str] | None:
    """Interactive channel cap menu when total > cap."""
    title = _yt_dlp_get_title(source)
    print(f"\n[batch] Auto-detected YouTube channel: \"{title}\"")
    print(f"[batch] Total videos on channel: {total}")

    if yes:
        # --yes accepts default cap (newest)
        print(f"[batch] --yes flag: processing {cap} most recent (default cap).")
        return _yt_dlp_enumerate(source, max_count=cap, order="newest")

    print(f"\nDefault cap is {cap} (most recent first). Would you like to:")
    print(f"  1. Process the {cap} most recent (default)")
    print(f"  2. Process the {cap} oldest")
    print(f"  3. Process all {total}")
    print(f"  4. Set a custom limit")
    print(f"  5. Cancel")

    choice = input("\n-> ").strip()
    if choice == "1" or choice == "":
        urls = _yt_dlp_enumerate(source, max_count=cap, order="newest")
    elif choice == "2":
        urls = _yt_dlp_enumerate(source, max_count=cap, order="oldest")
    elif choice == "3":
        urls = _yt_dlp_enumerate(source, max_count=0)
    elif choice == "4":
        raw = input("How many? (e.g. '30' or '30 oldest') -> ").strip()
        parts = raw.split()
        try:
            n = int(parts[0])
        except (IndexError, ValueError):
            print("[batch] Invalid input. Aborted.")
            return None
        ord2 = "oldest" if (len(parts) > 1 and "oldest" in parts[1].lower()) else "newest"
        urls = _yt_dlp_enumerate(source, max_count=n, order=ord2)
    else:
        print("[batch] Aborted.")
        return None

    print(f"[batch] Will process {len(urls)} video(s).")
    return urls


def detect_and_enumerate(
    source: str,
    max_videos: int = CHANNEL_CAP_DEFAULT,
    order: str = "newest",
    yes: bool = False,
) -> list[str] | None:
    """Detect if source is a batch (directory / playlist / channel).

    Returns list of URLs/paths, or None if it's a single video, or empty
    list if the user aborted. Raises RuntimeError on enumeration failure.
    """
    p = Path(source)

    # ── Local directory ────────────────────────────────────────────────────
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS)
        if not files:
            print(f"[batch] No video files found in {source}", file=sys.stderr)
            return []
        paths = [str(f) for f in files]
        return _confirm_batch(paths, f"Local directory: {source} ({len(paths)} video files)", yes)

    url_low = source.lower()

    # ── YouTube channel ────────────────────────────────────────────────────
    if _is_channel_url(source):
        print(f"[batch] Enumerating channel (this may take a moment)…", file=sys.stderr)
        # Single yt-dlp call to get all URLs + total (avoids double enumeration).
        all_urls = _yt_dlp_enumerate(source, max_count=0, order=order)
        total = len(all_urls)
        if total == 0:
            print(f"[batch] No videos found at channel URL.", file=sys.stderr)
            return []
        if max_videos > 0 and total > max_videos:
            return _channel_cap_prompt(source, total, max_videos, order, yes)
        # Under cap — reuse the already-enumerated list
        title = _yt_dlp_get_title(source)
        return _confirm_batch(all_urls, f"YouTube channel: \"{title}\"", yes)

    # ── YouTube playlist ───────────────────────────────────────────────────
    if _is_playlist_url(source):
        print(f"[batch] Enumerating playlist (this may take a moment)…", file=sys.stderr)
        urls = _yt_dlp_enumerate(source, max_count=0)  # playlists: no cap
        if not urls:
            print(f"[batch] No videos found in playlist.", file=sys.stderr)
            return []
        title = _yt_dlp_get_title(source)
        return _confirm_batch(urls, f"YouTube playlist: \"{title}\"", yes)

    # ── Single video ───────────────────────────────────────────────────────
    return None  # caller handles single-video flow


# ── 16.1  Core batch runner ───────────────────────────────────────────────────

def _get_python() -> str:
    """Return the Python executable for subprocesses."""
    return sys.executable


def run_single(
    url: str,
    mode: str,
    question: str,
    out_dir: Optional[Path],
) -> dict:
    """Run watch.py + answer.py for one URL. Returns result dict."""
    start = time.monotonic()
    python = _get_python()
    watch_script  = SCRIPT_DIR / "watch.py"
    answer_script = SCRIPT_DIR / "answer.py"

    # Step 1: watch.py → collect frames + transcript into work_dir
    watch_cmd = [python, str(watch_script), url, "--mode", mode]
    if out_dir:
        vid_dir = out_dir / f"batch_{int(time.time())}_{hash(url) & 0xFFFF:04x}"
        vid_dir.mkdir(parents=True, exist_ok=True)
        watch_cmd.extend(["--out-dir", str(vid_dir)])

    try:
        watch_result = subprocess.run(
            watch_cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max per video
        )
    except subprocess.TimeoutExpired:
        return {"url": url, "status": "error", "error": "watch.py timed out (10 min)"}

    if watch_result.returncode != 0:
        err = (watch_result.stderr or watch_result.stdout or "unknown error")[-300:]
        return {"url": url, "status": "error", "error": f"watch.py failed: {err}"}

    # Parse work_dir from watch.py stderr
    work_dir = None
    for line in watch_result.stderr.splitlines():
        if "[watch] working dir:" in line:
            work_dir = line.split("[watch] working dir:")[-1].strip()
            break

    if not work_dir:
        return {"url": url, "status": "error", "error": "Could not find work_dir in watch.py output"}

    # Step 2: answer.py → analyse + persist
    answer_cmd = [
        python, str(answer_script),
        work_dir,
        question,
        "--model", "claude-sonnet-4-6",
        "--mode", mode,
    ]

    try:
        answer_result = subprocess.run(
            answer_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min for analysis
        )
    except subprocess.TimeoutExpired:
        return {"url": url, "status": "error", "error": "answer.py timed out (5 min)",
                "work_dir": work_dir}

    elapsed = time.monotonic() - start

    if answer_result.returncode not in (0, 9):  # 9 = fallback, still ok
        err = (answer_result.stderr or answer_result.stdout or "unknown error")[-300:]
        return {"url": url, "status": "error", "error": f"answer.py failed: {err}",
                "work_dir": work_dir, "elapsed": elapsed}

    # Read wiki_path + title from watch_meta.json (written by persist_all).
    # This is more reliable than scraping answer.py's stderr output.
    wiki_path = None
    title     = url
    duration  = ""
    try:
        watch_meta_path = Path(work_dir) / "watch_meta.json"
        if watch_meta_path.exists():
            watch_meta = json.loads(watch_meta_path.read_text(encoding="utf-8"))
            wiki_path  = watch_meta.get("wiki_path") or None
            title      = watch_meta.get("title") or url
    except Exception:
        pass

    # Fallback: also read meta.json for duration (watch_meta doesn't carry it)
    try:
        meta_path = Path(work_dir) / "meta.json"
        if meta_path.exists():
            meta  = json.loads(meta_path.read_text(encoding="utf-8"))
            if not title or title == url:
                title = meta.get("title") or url
            dur_s = int(meta.get("duration_seconds") or 0)
            if dur_s:
                m, s     = divmod(dur_s, 60)
                duration = f"{m}m{s:02d}s"
    except Exception:
        pass

    return {
        "url": url,
        "status": "ok",
        "title": title,
        "wiki_path": wiki_path or "",
        "duration": duration,
        "work_dir": work_dir,
        "elapsed": elapsed,
    }


# ── 16.2  Quota pre-check ─────────────────────────────────────────────────────

def quota_precheck(n_videos: int, mode: str) -> bool:
    """Warn if Gemini quota is insufficient for the batch. Returns True if ok."""
    try:
        from quota_tracker import get_quota_status, LIMITS
        status = get_quota_status()
        model_key = "pro" if mode == "chart" else "flash"
        limit      = LIMITS.get(model_key, 0)
        used       = status.get(f"{model_key}_used", 0)
        remaining  = limit - used

        if remaining < n_videos:
            print(
                f"[batch] WARNING: {n_videos} video(s) requested but only "
                f"{remaining} {model_key.upper()} calls remain today.",
                file=sys.stderr,
            )
            if mode == "chart":
                flash_used      = status.get("flash_used", 0)
                flash_remaining = LIMITS.get("flash", 0) - flash_used
                print(
                    f"[batch] Tip: use --mode regular to use Flash quota "
                    f"({flash_remaining} remaining), or split across sessions.",
                    file=sys.stderr,
                )
            return False
    except Exception:
        pass  # quota_tracker unavailable — proceed
    return True


# ── 16.7  Checkpoint helpers ──────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Return the set of already-completed URLs from a checkpoint file.

    Returns empty set if the file doesn't exist or is unreadable.
    """
    try:
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return set(state.get("completed", []))
    except Exception:
        return set()


def save_checkpoint(checkpoint_path: Path, completed: set[str]) -> None:
    """Persist the set of completed URLs to disk. Best-effort — never raises."""
    try:
        checkpoint_path.write_text(
            json.dumps({"completed": sorted(completed)}, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[batch] WARNING: could not write checkpoint: {e}", file=sys.stderr)


def delete_checkpoint(checkpoint_path: Path) -> None:
    """Remove checkpoint file after a clean batch run. Best-effort."""
    try:
        checkpoint_path.unlink(missing_ok=True)
    except Exception:
        pass


# ── 16.3  Summary output ──────────────────────────────────────────────────────

def format_summary(results: list[dict], elapsed: float) -> str:
    """Format a markdown table of batch results."""
    n_ok   = sum(1 for r in results if r["status"] == "ok")
    n_err  = sum(1 for r in results if r["status"] == "error")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    m, s   = divmod(int(elapsed), 60)
    wall   = f"{m}m{s:02d}s"

    lines = [
        f"",
        f"## Batch summary — {len(results)} video(s) processed",
        f"",
        f"| # | Title | Status | Wiki page | Duration |",
        f"|---|-------|--------|-----------|----------|",
    ]

    for i, r in enumerate(results, 1):
        if r["status"] == "ok":
            status = "✅ ok"
        elif r["status"] == "skipped":
            status = "⏭ skipped"
        else:
            status = "❌ error"
        title = (r.get("title") or r["url"])[:60]
        wiki  = r.get("wiki_path") or "—"
        dur   = r.get("duration") or "—"
        err   = r.get("error", "")
        if err and r["status"] == "error":
            title = f"{title} ({err[:40]})"
        lines.append(f"| {i} | {title} | {status} | {wiki} | {dur} |")

    footer = f"{n_ok} succeeded, {n_err} failed"
    if n_skip:
        footer += f", {n_skip} skipped (checkpoint)"
    footer += f". Total wall time: {wall}."

    lines.extend(["", footer])
    return "\n".join(lines)


# ── 16.1  Main batch runner ───────────────────────────────────────────────────

def run_batch(
    urls: list[str],
    mode: str = "regular",
    question: str = "Summarize this video.",
    delay: float = 5.0,
    out_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    completed_urls: Optional[set[str]] = None,
) -> list[dict]:
    """Run the batch sequentially. Returns list of result dicts.

    If checkpoint_path is set, completed URLs are persisted after each
    successful video so an interrupted run can be resumed.
    """
    results: list[dict] = []
    total   = len(urls)
    done    = set(completed_urls or set())

    for i, url in enumerate(urls, 1):
        # ── Skip already-completed URLs (checkpoint resume) ───────────────
        if url in done:
            print(f"[batch] ({i}/{total}) Skipping (checkpoint): {url}", file=sys.stderr)
            results.append({"url": url, "status": "skipped", "title": url})
            continue

        print(f"\n[batch] ({i}/{total}) {url}", file=sys.stderr)
        result = run_single(url, mode=mode, question=question, out_dir=out_dir)
        results.append(result)

        status_msg = "ok" if result["status"] == "ok" else f"ERROR: {result.get('error', '')}"
        print(f"[batch] ({i}/{total}) {status_msg}", file=sys.stderr)

        # ── Write checkpoint after each success ───────────────────────────
        if result["status"] == "ok" and checkpoint_path is not None:
            done.add(url)
            save_checkpoint(checkpoint_path, done)

        if i < total and delay > 0:
            print(f"[batch] Waiting {delay}s before next video…", file=sys.stderr)
            time.sleep(delay)

    return results


# ── URL file reader ───────────────────────────────────────────────────────────

def read_url_file(path: Path) -> list[str]:
    """Read URLs from a text file, skipping blank lines and # comments."""
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="batch",
        description="Process multiple videos in a single invocation",
    )
    ap.add_argument("sources", nargs="*",
                    help="Video URL(s) or local path(s)")
    ap.add_argument("--batch",       metavar="FILE",
                    help="Text file with one URL per line")
    ap.add_argument("--mode",        choices=["regular", "chart"], default="regular")
    ap.add_argument("--delay",       type=float, default=5.0,
                    help="Seconds between videos (default 5)")
    ap.add_argument("--max-workers", type=int, default=1,
                    help="Parallel workers (default 1 — sequential; max 3)")
    ap.add_argument("--max-videos",  type=int, default=CHANNEL_CAP_DEFAULT,
                    help=f"Cap for channel enumeration (default {CHANNEL_CAP_DEFAULT}; 0=unlimited)")
    ap.add_argument("--order",       choices=["newest", "oldest", "default"], default="newest",
                    help="Channel video order (default: newest)")
    ap.add_argument("--yes",         action="store_true",
                    help="Skip confirmation prompt")
    ap.add_argument("--question",       default="Summarize this video.",
                    help="Question to answer for each video")
    ap.add_argument("--out-dir",        type=str, default=None,
                    help="Parent directory for per-video work dirs")
    ap.add_argument("--no-checkpoint",  action="store_true",
                    help="Disable checkpoint — re-process all URLs even if previously done")
    args = ap.parse_args()

    # ── Collect URLs ─────────────────────────────────────────────────────
    urls: list[str] = list(args.sources or [])

    if args.batch:
        batch_path = Path(args.batch).expanduser().resolve()
        if not batch_path.exists():
            print(f"[batch] ERROR: batch file not found: {batch_path}", file=sys.stderr)
            return 1
        urls.extend(read_url_file(batch_path))

    if not urls:
        ap.error("Provide at least one URL/path or --batch FILE")

    # ── Auto-detect batch sources (16.5) ──────────────────────────────────
    # If a single source is given and it looks like a channel/playlist/dir,
    # enumerate it into a real URL list.
    if len(urls) == 1:
        detected = detect_and_enumerate(
            urls[0],
            max_videos=args.max_videos,
            order=args.order,
            yes=args.yes,
        )
        if detected is None:
            # Single video — run as single-video batch of 1
            pass
        elif len(detected) == 0:
            # Aborted or nothing found
            return 0
        else:
            urls = detected

    # Multiple URLs: show confirmation unless --yes
    elif not args.yes:
        print(f"\n[batch] {len(urls)} URL(s) queued.")
        _print_video_list(urls)
        answer = input(f"\nProcess all {len(urls)}? [yes / no] -> ").strip().lower()
        if answer not in ("yes", "y"):
            print("[batch] Aborted.")
            return 0

    # ── Quota pre-check (16.2) ─────────────────────────────────────────────
    quota_precheck(len(urls), args.mode)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None

    # ── Checkpoint (16.7) ─────────────────────────────────────────────────
    # batch_state.json lives next to batch_summary.md (cwd, or out_dir if set)
    artifact_dir    = out_dir or Path(".")
    checkpoint_path = None
    completed_urls: set[str] = set()

    if not args.no_checkpoint:
        checkpoint_path = artifact_dir / "batch_state.json"
        completed_urls  = load_checkpoint(checkpoint_path)
        if completed_urls:
            print(
                f"[batch] Checkpoint found — {len(completed_urls)} URL(s) already done; "
                f"resuming from where we left off.",
                file=sys.stderr,
            )

    # ── Run batch (16.1) ──────────────────────────────────────────────────
    batch_start = time.monotonic()
    results = run_batch(
        urls,
        mode=args.mode,
        question=args.question,
        delay=args.delay,
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        completed_urls=completed_urls,
    )
    elapsed = time.monotonic() - batch_start

    # ── Summary (16.3) ─────────────────────────────────────────────────────
    summary = format_summary(results, elapsed)
    print(summary)

    # Write summary to batch_summary.md
    summary_path = artifact_dir / "batch_summary.md"
    try:
        summary_path.write_text(summary + "\n", encoding="utf-8")
        print(f"\n[batch] Summary written to {summary_path.resolve()}", file=sys.stderr)
    except Exception as e:
        print(f"[batch] Could not write summary file: {e}", file=sys.stderr)

    # ── Clean up checkpoint on full success (16.7) ────────────────────────
    n_err = sum(1 for r in results if r["status"] == "error")
    if not n_err and checkpoint_path:
        delete_checkpoint(checkpoint_path)
        print(f"[batch] Checkpoint cleared — all videos complete.", file=sys.stderr)

    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
