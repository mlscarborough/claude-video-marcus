"""Write watch_invocation and refine_call telemetry rows to Supabase.

Called from watch.py at end of each invocation (writer laptop only).
"""
from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from pathlib import Path


def _get_supabase():
    from dotenv import load_dotenv
    _env = Path.home() / ".config" / "watch" / ".env"
    if _env.exists():
        load_dotenv(_env)

    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def log_invocation(
    mode: str,
    source_type: str,
    duration_seconds: float,
    provider: str,
    n_frames: int = 0,
    n_chunks: int = 0,
) -> None:
    """Write a watch_invocation telemetry row. Best-effort — never raises."""
    if os.environ.get("WATCH_IS_WRITER_LAPTOP", "false").lower() != "true":
        return
    try:
        sb = _get_supabase()
        if sb is None:
            return
        sb.table("watch_telemetry").insert({
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "event_type": "watch_invocation",
            "event_data": {
                "mode": mode,
                "source_type": source_type,
                "duration_seconds": duration_seconds,
                "provider": provider,
                "n_frames": n_frames,
                "n_embedding_chunks": n_chunks,
            },
            "hostname": socket.gethostname(),
        }).execute()
    except Exception:
        pass


def log_refine_call(timestamp: float, window: float, fps: float, n_frames: int) -> None:
    """Write a refine_call telemetry row. Best-effort — never raises."""
    if os.environ.get("WATCH_IS_WRITER_LAPTOP", "false").lower() != "true":
        return
    try:
        sb = _get_supabase()
        if sb is None:
            return
        sb.table("watch_telemetry").insert({
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "event_type": "refine_call",
            "event_data": {
                "timestamp_seconds": timestamp,
                "window_seconds": window,
                "fps": fps,
                "n_frames_extracted": n_frames,
            },
            "hostname": socket.gethostname(),
        }).execute()
    except Exception:
        pass
