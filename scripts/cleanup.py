"""Retention-mode enforcement: delete work directories older than 48h.

Called from watch.py at end of each invocation (best-effort).
Respects the retention_mode from each video's Supabase row:
  ephemeral  → delete frames + download dir after WORK_DIR_RETENTION_HOURS
  transcript → keep download/captions but delete frames dir
  full       → keep everything (handled by --keep all flag at download time)
"""
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


def _retention_hours() -> float:
    return float(os.environ.get("WORK_DIR_RETENTION_HOURS", "48"))


def _work_dirs() -> list[Path]:
    """Find all watch-* temp directories in the system temp dir."""
    import tempfile
    tmp = Path(tempfile.gettempdir())
    return [d for d in tmp.glob("watch-*") if d.is_dir()]


def cleanup_work_dirs() -> int:
    """Delete old work directories based on retention_mode.

    Returns number of directories cleaned up.
    """
    hours = _retention_hours()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cleaned = 0

    for work_dir in _work_dirs():
        mtime = datetime.fromtimestamp(work_dir.stat().st_mtime, tz=timezone.utc)
        if mtime >= cutoff:
            continue

        # Determine retention mode from watch_meta.json if present
        meta_file = work_dir / "watch_meta.json"
        retention = "ephemeral"
        if meta_file.exists():
            try:
                import json
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                retention = meta.get("retention", "ephemeral")
            except Exception:
                pass

        try:
            if retention == "full":
                # Nothing to delete — kept by design
                continue
            elif retention == "transcript":
                # Delete frames dir only
                frames_dir = work_dir / "frames"
                if frames_dir.exists():
                    shutil.rmtree(frames_dir)
                    print(f"[cleanup] deleted frames in {work_dir.name}", file=sys.stderr)
                # Also delete refined frames
                refined_dir = work_dir / "refined"
                if refined_dir.exists():
                    shutil.rmtree(refined_dir)
                cleaned += 1
            else:
                # ephemeral: delete entire work dir
                shutil.rmtree(work_dir)
                print(f"[cleanup] deleted {work_dir.name}", file=sys.stderr)
                cleaned += 1
        except Exception as e:
            print(f"[cleanup] could not delete {work_dir}: {e}", file=sys.stderr)

    return cleaned


if __name__ == "__main__":
    n = cleanup_work_dirs()
    print(f"[cleanup] cleaned {n} work dir(s)")
