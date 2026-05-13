"""Watchdog daemon: monitor Obsidian vault for wiki edits, sync to Supabase.

Runs as a background Windows scheduled task (see register-watcher-task.ps1).
Debounces rapid saves (2s window). Sends a heartbeat to wiki_sync_log every 30 min.

Usage:
    python sync_watcher.py [--vault <path>] [--heartbeat-interval <seconds>]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
_env = Path.home() / ".config" / "watch" / ".env"
if _env.exists():
    load_dotenv(_env)


def _vault_path() -> Path:
    vp = os.environ.get("OBSIDIAN_VAULT_PATH", "")
    if not vp:
        raise RuntimeError("OBSIDIAN_VAULT_PATH not set in ~/.config/watch/.env")
    return Path(vp).expanduser().resolve()


class _WikiChangeHandler:
    """Collect filesystem events, debounce, then call sync_file."""

    DEBOUNCE_SECONDS = 2.0

    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self._pending: dict[Path, float] = {}  # path → process_after timestamp

    # ── watchdog EventHandler protocol ──────────────────────────────────────
    def dispatch(self, event) -> None:
        pass

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        src = Path(event.src_path)
        if src.suffix.lower() != ".md":
            return
        self._pending[src] = time.monotonic() + self.DEBOUNCE_SECONDS

    def on_created(self, event) -> None:
        self.on_modified(event)

    # ── Called from main loop ────────────────────────────────────────────────
    def flush_pending(self) -> None:
        now = time.monotonic()
        ready = [p for p, t in self._pending.items() if t <= now]
        for path in ready:
            del self._pending[path]
            try:
                from sync_wiki import sync_file
                sync_file(path, self.vault_path)
            except Exception as e:
                print(f"[sync-watcher] ERROR syncing {path}: {e}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(prog="sync-watcher")
    ap.add_argument("--vault", type=str, default=None, help="Override vault path from .env")
    ap.add_argument(
        "--heartbeat-interval",
        type=int,
        default=1800,
        help="Seconds between heartbeat rows (default 1800 = 30 min)",
    )
    args = ap.parse_args()

    if os.environ.get("WATCH_IS_WRITER_LAPTOP", "false").lower() != "true":
        print("[sync-watcher] read-only laptop — exiting (sync disabled)", file=sys.stderr)
        return 0

    vault = Path(args.vault).expanduser().resolve() if args.vault else _vault_path()
    if not vault.exists():
        print(f"[sync-watcher] vault path does not exist: {vault}", file=sys.stderr)
        return 1

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("[sync-watcher] watchdog not installed — run: pip install watchdog", file=sys.stderr)
        return 1

    handler = _WikiChangeHandler(vault_path=vault)

    # Wrap our handler so watchdog can call it properly
    class _WatchdogBridge(FileSystemEventHandler):
        def on_modified(self, event):
            handler.on_modified(event)

        def on_created(self, event):
            handler.on_created(event)

    observer = Observer()
    observer.schedule(_WatchdogBridge(), str(vault), recursive=True)
    observer.start()

    print(f"[sync-watcher] watching {vault}", file=sys.stderr)

    last_heartbeat = time.monotonic()
    try:
        while True:
            handler.flush_pending()

            # Heartbeat every N seconds
            if time.monotonic() - last_heartbeat >= args.heartbeat_interval:
                from sync_wiki import write_heartbeat
                write_heartbeat()
                last_heartbeat = time.monotonic()

            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        print("[sync-watcher] stopped", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
