"""HTTP client for the audit-stage2 Edge Function.

Wraps every Gemini/Claude vision call per CLAUDE.md Rule 3.
Never raises — audit failures are logged to stderr and swallowed.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path


def log_audit(model: str, input_data: str, output_data: str,
              tokens_in: int = 0, tokens_out: int = 0,
              confidence: float | None = None,
              flags: list[str] | None = None) -> str | None:
    """POST audit payload to audit-stage2 Edge Function. Returns row ID or None."""
    try:
        import requests
        from dotenv import load_dotenv
        _env = Path.home() / ".config" / "watch" / ".env"
        if _env.exists():
            load_dotenv(_env)

        endpoint = os.environ.get("WATCH_AUDIT_ENDPOINT")
        service_key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not endpoint or not service_key:
            return None

        payload = {
            "agent_name": "watch",
            "session_id": os.environ.get("CLAUDE_SESSION_ID", str(uuid.uuid4())),
            "model": model,
            "input_hash": hashlib.sha256(input_data.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_data.encode()).hexdigest(),
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "confidence": confidence,
            "flags": flags or [],
        }

        r = requests.post(
            endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {service_key}"},
            timeout=5,
        )
        r.raise_for_status()
        return r.json().get("id")
    except Exception as e:
        import sys
        print(f"[watch] audit log failed (non-fatal): {e}", file=sys.stderr)
        return None
