"""Gemini free-tier quota tracker.

Reads/writes ~/.config/watch/gemini-usage.json. Also syncs with Supabase
gemini_quota_log on startup (DB is canonical; local JSON is fast cache).

Thresholds:
    70% → warn in stderr
    90% → auto-downgrade Pro → Flash in chart mode
    100% (or last 429 within 1h) → fall back to Claude
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

TRACKER_PATH = Path.home() / ".config" / "watch" / "gemini-usage.json"

LIMITS = {
    "pro": 50,       # gemini-2.5-pro free-tier daily
    "flash": 1500,   # gemini-2.0-flash free-tier daily
    "embed": 1500,   # text-embedding-004 free-tier daily
}

WARN_PCT = 0.70
DEGRADE_PCT = 0.90

_FOUR_TWENTY_NINE_BLOCK_SECONDS = 3600  # treat 429 as exhausted for 1 hour


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _read_local() -> dict:
    if not TRACKER_PATH.exists():
        return _empty_tracker()
    try:
        return json.loads(TRACKER_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _empty_tracker()


def _empty_tracker() -> dict:
    return {
        "date": _today_utc(),
        "pro_used": 0,
        "flash_used": 0,
        "embed_used": 0,
        "last_429_pro": None,
        "last_429_flash": None,
        "last_429_embed": None,
    }


def _write_local(data: dict) -> None:
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _reset_if_new_day(data: dict) -> dict:
    if data.get("date") != _today_utc():
        data = _empty_tracker()
        _write_local(data)
    return data


def get_quota_status() -> dict:
    """Returns current quota status, reconciled with DB if possible."""
    data = _read_local()
    data = _reset_if_new_day(data)

    # Try to reconcile with DB (best-effort; skip on failure)
    try:
        _reconcile_with_db(data)
        data = _read_local()  # re-read after possible update
    except Exception:
        pass

    return data


def _reconcile_with_db(local: dict) -> None:
    """Pull today's counts from gemini_quota_log; trust DB if higher."""
    from dotenv import load_dotenv
    _env = Path.home() / ".config" / "watch" / ".env"
    if _env.exists():
        load_dotenv(_env)

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return

    from supabase import create_client
    sb = create_client(url, key)
    today = _today_utc()
    resp = sb.table("gemini_quota_log").select("provider").gte("call_at", today + "T00:00:00Z").execute()
    if not resp.data:
        return

    db_pro = sum(1 for r in resp.data if "pro" in r["provider"])
    db_flash = sum(1 for r in resp.data if "flash" in r["provider"])
    db_embed = sum(1 for r in resp.data if "embed" in r["provider"])

    changed = False
    if db_pro > local.get("pro_used", 0):
        local["pro_used"] = db_pro
        changed = True
    if db_flash > local.get("flash_used", 0):
        local["flash_used"] = db_flash
        changed = True
    if db_embed > local.get("embed_used", 0):
        local["embed_used"] = db_embed
        changed = True

    if changed:
        _write_local(local)


def increment_usage(provider: str, count: int = 1) -> dict:
    """Increment counter for 'pro', 'flash', or 'embed'. Returns updated status."""
    data = _read_local()
    data = _reset_if_new_day(data)
    key = f"{provider}_used"
    data[key] = data.get(key, 0) + count
    _write_local(data)
    return data


def mark_429(provider: str) -> None:
    """Record that we hit a 429 for this provider."""
    data = _read_local()
    data[f"last_429_{provider}"] = datetime.now(timezone.utc).isoformat()
    _write_local(data)


def _usage_pct(data: dict, provider: str) -> float:
    used = data.get(f"{provider}_used", 0)
    limit = LIMITS.get(provider, 1)
    return used / limit


def _last_429_within_block(data: dict, provider: str) -> bool:
    ts_str = data.get(f"last_429_{provider}")
    if not ts_str:
        return False
    try:
        ts = datetime.fromisoformat(ts_str)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return age < _FOUR_TWENTY_NINE_BLOCK_SECONDS
    except Exception:
        return False


def should_warn(provider: str) -> bool:
    data = _read_local()
    return _usage_pct(data, provider) >= WARN_PCT


def should_degrade(provider: str) -> bool:
    data = _read_local()
    return _usage_pct(data, provider) >= DEGRADE_PCT


def is_exhausted(provider: str) -> bool:
    data = _read_local()
    return _usage_pct(data, provider) >= 1.0 or _last_429_within_block(data, provider)


def log_quota_to_db(provider_name: str, tokens_in: int, tokens_out: int,
                    was_429: bool = False, was_degraded: bool = False) -> None:
    """Write a row to gemini_quota_log. Best-effort, never raises."""
    try:
        import socket
        from dotenv import load_dotenv
        _env = Path.home() / ".config" / "watch" / ".env"
        if _env.exists():
            load_dotenv(_env)

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            return

        from supabase import create_client
        sb = create_client(url, key)
        sb.table("gemini_quota_log").insert({
            "call_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider_name,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "was_429": was_429,
            "was_degraded": was_degraded,
            "hostname": socket.gethostname(),
        }).execute()
    except Exception:
        pass
