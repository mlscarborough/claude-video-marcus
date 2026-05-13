"""Observability commands: --health (dashboard) and --doctor (sanity checks).

Called by watch.py when --health or --doctor flags are passed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
_env = Path.home() / ".config" / "watch" / ".env"
if _env.exists():
    load_dotenv(_env)

# ANSI color helpers
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

_STATUS_COLOR = {"OK": _GREEN, "WARN": _YELLOW, "RED": _RED}


def _col(status: str) -> str:
    return _STATUS_COLOR.get(status, "")


def _line(label: str, value: str, status: str) -> str:
    flag = f"{_col(status)}{status}{_RESET}"
    return f"  {label:<38} {value:<30} [{flag}]"


def _get_supabase():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


# ── --health ──────────────────────────────────────────────────────────────────

def run_health() -> int:
    sb = _get_supabase()

    print(f"\n{_BOLD}=== /watch health dashboard ==={_RESET}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Heartbeat
    if sb:
        try:
            resp = (
                sb.table("wiki_sync_log")
                .select("hostname,synced_at")
                .order("synced_at", desc=True)
                .limit(20)
                .execute()
            )
            laptops: dict[str, datetime] = {}
            for row in resp.data:
                h = row["hostname"] or "unknown"
                ts = datetime.fromisoformat(row["synced_at"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if h not in laptops or ts > laptops[h]:
                    laptops[h] = ts
            now = datetime.now(timezone.utc)
            for host, ts in laptops.items():
                age = now - ts
                age_str = _fmt_duration(age)
                status = "OK" if age < timedelta(hours=2) else "RED"
                print(_line(f"Last heartbeat ({host})", age_str + " ago", status))
        except Exception as e:
            print(_line("Heartbeat check", f"error: {e}", "RED"))
    else:
        print(_line("Heartbeat check", "no DB connection", "RED"))

    # 2. Sync gap
    if sb:
        try:
            last_sync = (
                sb.table("wiki_sync_log")
                .select("synced_at")
                .order("synced_at", desc=True)
                .limit(1)
                .execute()
            )
            last_processed = (
                sb.table("videos")
                .select("processed_at")
                .order("processed_at", desc=True)
                .limit(1)
                .execute()
            )
            if last_sync.data:
                ts = datetime.fromisoformat(last_sync.data[0]["synced_at"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - ts
                status = "OK" if age < timedelta(minutes=30) else ("WARN" if age < timedelta(hours=2) else "RED")
                print(_line("Last wiki sync", _fmt_duration(age) + " ago", status))
            n_videos = len(last_processed.data)
            print(_line("Videos in DB", str(n_videos), "OK"))
        except Exception as e:
            print(_line("Sync gap check", f"error: {e}", "RED"))

    # 3. Gemini quota
    try:
        from quota_tracker import get_quota_status, LIMITS
        status = get_quota_status()
        for key, label in [("pro", "Gemini Pro"), ("flash", "Flash"), ("embed", "Embed")]:
            used = status.get(f"{key}_used", 0)
            limit = LIMITS.get(key, 1)
            pct = used / limit * 100
            pct_str = f"{used}/{limit} ({pct:.0f}%)"
            s = "OK" if pct < 70 else ("WARN" if pct < 90 else "RED")
            print(_line(f"Quota {label}", pct_str, s))
    except Exception as e:
        print(_line("Quota check", f"error: {e}", "WARN"))

    # 4. Audit ratio
    if sb:
        try:
            audit_count = sb.table("audit_log").select("id", count="exact").eq("agent_name", "watch").execute()
            video_count = sb.table("videos").select("id", count="exact").execute()
            a = audit_count.count or 0
            v = video_count.count or 0
            ratio_str = f"{a} audit / {v} videos"
            s = "OK" if v == 0 or a >= v else "WARN"
            print(_line("Audit coverage", ratio_str, s))
        except Exception as e:
            print(_line("Audit ratio", f"error: {e}", "WARN"))

    # 5. Pending videos
    if sb:
        try:
            pend = sb.table("videos").select("id", count="exact").eq("is_pending", True).execute()
            n = pend.count or 0
            print(_line("Pending archives", str(n), "OK" if n == 0 else "WARN"))
        except Exception as e:
            print(_line("Pending archives", f"error: {e}", "WARN"))

    # 6. Watcher task status
    try:
        r = subprocess.run(
            ["schtasks", "/query", "/tn", "watch-sync-watcher", "/fo", "csv"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0 and "Running" in r.stdout:
            print(_line("Sync watcher task", "Running", "OK"))
        elif r.returncode == 0:
            lines = [l for l in r.stdout.splitlines() if "watch-sync-watcher" in l]
            state = lines[0].split(",")[3].strip('"') if lines else "Unknown"
            print(_line("Sync watcher task", state, "WARN"))
        else:
            print(_line("Sync watcher task", "Not registered", "WARN"))
    except Exception:
        print(_line("Sync watcher task", "schtasks check failed", "WARN"))

    print()
    return 0


# ── --doctor ──────────────────────────────────────────────────────────────────

def _check(name: str):
    """Decorator to register a check function."""
    def decorator(fn):
        fn._check_name = name
        return fn
    return decorator


@_check("Tesseract installed")
def check_tesseract_installed():
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        v = pytesseract.get_tesseract_version()
        return True, f"v{v}", ""
    except Exception as e:
        return (
            False,
            f"Tesseract binary not found or pytesseract error: {e}",
            "winget install UB-Mannheim.TesseractOCR  (open new PowerShell after)",
        )


@_check("Python version >= 3.10")
def check_python_version():
    v = sys.version_info
    ok = v >= (3, 10)
    return ok, f"{v.major}.{v.minor}.{v.micro}", "Install Python 3.10+ from python.org"


@_check(".env file exists")
def check_env_file_exists():
    p = Path.home() / ".config" / "watch" / ".env"
    if p.exists():
        return True, str(p), ""
    return (
        False,
        f"Missing: {p}",
        r"New-Item -ItemType File -Path $env:USERPROFILE\.config\watch\.env  (then fill in keys)",
    )


@_check("GEMINI_API_KEY set")
def check_gemini_api_key():
    k = os.environ.get("GEMINI_API_KEY", "")
    if k and k != "<paste from P4>":
        return True, f"...{k[-6:]}", ""
    return (
        False,
        "GEMINI_API_KEY missing or placeholder",
        "Get key from https://ai.google.dev/ and add to ~/.config/watch/.env",
    )


@_check("Gemini API reachable")
def check_gemini_api_reachable():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return False, "No API key to test with", "Set GEMINI_API_KEY first"
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        models = list(genai.list_models())
        return True, f"{len(models)} model(s) visible", ""
    except Exception as e:
        return False, f"API call failed: {e}", "Check key validity at https://ai.google.dev/studio"


@_check("GROQ_API_KEY set")
def check_groq_api_key():
    k = os.environ.get("GROQ_API_KEY", "")
    if k and k.startswith("gsk_"):
        return True, f"...{k[-6:]}", ""
    return (
        False,
        "GROQ_API_KEY missing or invalid",
        "Get key from https://console.groq.com/keys and add to ~/.config/watch/.env",
    )


@_check("SUPABASE_URL set")
def check_supabase_url():
    url = os.environ.get("SUPABASE_URL", "")
    if url.startswith("https://") and ".supabase.co" in url:
        return True, url, ""
    return (
        False,
        f"SUPABASE_URL missing or invalid: '{url}'",
        "Supabase dashboard → Settings → API → Project URL",
    )


@_check("Supabase reachable")
def check_supabase_reachable():
    sb = _get_supabase()
    if sb is None:
        return False, "SUPABASE_URL/KEY not set", "Set Supabase credentials in .env"
    try:
        r = sb.table("videos").select("id").limit(1).execute()
        return True, f"connected (videos table visible)", ""
    except Exception as e:
        return False, f"Connection failed: {e}", "Check SUPABASE_URL and SUPABASE_SERVICE_KEY"


@_check("videos schema up to date")
def check_schema_up_to_date():
    sb = _get_supabase()
    if sb is None:
        return False, "No DB connection", ""
    expected = [
        "videos", "video_frames", "video_transcript_segments",
        "video_analyses", "video_entities", "wiki_embeddings",
        "wiki_sync_log", "gemini_quota_log", "watch_telemetry",
    ]
    try:
        missing = []
        for table in expected:
            try:
                sb.table(table).select("id").limit(1).execute()
            except Exception:
                missing.append(table)
        if not missing:
            return True, f"all {len(expected)} tables present", ""
        return (
            False,
            f"Missing tables: {missing}",
            "Apply supabase/migrations/20260513_create_video_tables.sql via Supabase dashboard",
        )
    except Exception as e:
        return False, f"Schema check failed: {e}", "Check SUPABASE_URL and SUPABASE_SERVICE_KEY"


@_check("Audit endpoint reachable")
def check_audit_endpoint_reachable():
    endpoint = os.environ.get("WATCH_AUDIT_ENDPOINT", "")
    if not endpoint or endpoint == "<filled in Task 9>":
        return (
            False,
            "WATCH_AUDIT_ENDPOINT not set",
            "Add WATCH_AUDIT_ENDPOINT to ~/.config/watch/.env after deploying Edge Function",
        )
    try:
        import requests as _req
        svc_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        r = _req.post(
            endpoint,
            headers={"Authorization": f"Bearer {svc_key}", "Content-Type": "application/json"},
            json={
                "agent_name": "watch",
                "model": "health-check",
                "input_hash": "a" * 64,
                "output_hash": "b" * 16,
            },
            timeout=10,
        )
        if r.status_code == 200:
            return True, f"200 OK (id={r.json().get('id')})", ""
        return (
            False,
            f"Endpoint returned {r.status_code}: {r.text[:100]}",
            "Check Edge Function logs in Supabase dashboard",
        )
    except Exception as e:
        return False, f"Request failed: {e}", "Check network and WATCH_AUDIT_ENDPOINT value"


@_check("Sync watcher task registered")
def check_watcher_registered():
    try:
        r = subprocess.run(
            ["schtasks", "/query", "/tn", "watch-sync-watcher"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            return True, "Task registered", ""
        return (
            False,
            "Task 'watch-sync-watcher' not found in Task Scheduler",
            r"Run as Admin: .\scripts\register-watcher-task.ps1",
        )
    except Exception as e:
        return False, f"schtasks check failed: {e}", "Ensure schtasks.exe is on PATH"


@_check("Sync watcher task running")
def check_watcher_running():
    try:
        r = subprocess.run(
            ["schtasks", "/query", "/tn", "watch-sync-watcher", "/fo", "csv"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            return False, "Task not registered", r"Run register-watcher-task.ps1 as Admin"
        if "Running" in r.stdout:
            return True, "Running", ""
        lines = [l for l in r.stdout.splitlines() if "watch-sync-watcher" in l]
        state = lines[0].split(",")[3].strip('"') if lines else "Unknown"
        return (
            False,
            f"Task state: {state}",
            "schtasks /run /tn watch-sync-watcher",
        )
    except Exception as e:
        return False, f"schtasks check failed: {e}", "Check Task Scheduler manually"


@_check("Obsidian vault exists")
def check_obsidian_vault_exists():
    vp = os.environ.get("OBSIDIAN_VAULT_PATH", "")
    if not vp:
        return (
            False,
            "OBSIDIAN_VAULT_PATH not set",
            r"mkdir C:\Users\marcu\Documents\AI-Agents-Wiki  and add path to .env",
        )
    p = Path(vp).expanduser().resolve()
    if p.exists() and p.is_dir():
        md_count = len(list(p.rglob("*.md")))
        return True, f"{p} ({md_count} .md files)", ""
    return (
        False,
        f"Vault path does not exist: {p}",
        f"mkdir {p}",
    )


CHECKS = [
    check_tesseract_installed,
    check_python_version,
    check_env_file_exists,
    check_gemini_api_key,
    check_gemini_api_reachable,
    check_groq_api_key,
    check_supabase_url,
    check_supabase_reachable,
    check_schema_up_to_date,
    check_audit_endpoint_reachable,
    check_watcher_registered,
    check_watcher_running,
    check_obsidian_vault_exists,
]


def run_doctor() -> int:
    print(f"\n{_BOLD}=== /watch --doctor ==={_RESET}")
    print(f"  Running {len(CHECKS)} sanity checks…\n")

    results = []
    for check in CHECKS:
        name = getattr(check, "_check_name", check.__name__.replace("check_", "").replace("_", " "))
        try:
            passed, diag, fix = check()
            results.append((name, passed, diag, fix))
        except Exception as e:
            results.append((name, False, f"Check crashed: {e}", "Re-run with more verbose logging"))

    for name, passed, diag, _ in results:
        mark = "+" if passed else "-"
        label = "PASS" if passed else "FAIL"
        col = _GREEN if passed else _RED
        print(f"  [{col}{mark}{_RESET}] {name:<40} {col}{label}{_RESET}")

    failed = [(n, d, f) for n, p, d, f in results if not p]
    if not failed:
        print(f"\n{_GREEN}All checks passed. System is healthy.{_RESET}\n")
        return 0

    print(f"\n{_RED}{len(failed)} check(s) failed:{_RESET}\n")
    for name, diag, fix in failed:
        print(f"  {_BOLD}{name}{_RESET}")
        print(f"    Diagnosis: {diag}")
        if fix:
            print(f"    Fix:       {fix}")
        print()

    return 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_duration(td: timedelta) -> str:
    total = int(td.total_seconds())
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    return f"{total // 3600}h {(total % 3600) // 60}m"
