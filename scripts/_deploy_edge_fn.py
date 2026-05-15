"""Deploy audit-stage2 Edge Function via Supabase Management API.

Usage:
    SUPABASE_PAT=sbp_xxx SUPABASE_PROJECT_REF=xxxxxxxx python _deploy_edge_fn.py

Environment variables (read from ~/.config/watch/.env if not set):
    SUPABASE_PAT          — Supabase Personal Access Token (Dashboard → Account → Access Tokens)
    SUPABASE_PROJECT_REF  — Project ref (Settings → General → Reference ID)
"""
import json, os, sys, requests
from pathlib import Path
from dotenv import load_dotenv

_env = Path.home() / ".config" / "watch" / ".env"
if _env.exists():
    load_dotenv(_env)

token = os.environ.get("SUPABASE_PAT", "")
ref   = os.environ.get("SUPABASE_PROJECT_REF", "")
if not token or not ref:
    print("ERROR: set SUPABASE_PAT and SUPABASE_PROJECT_REF (in env or ~/.config/watch/.env)", file=sys.stderr)
    sys.exit(1)

slug    = "audit-stage2"
fn_path = Path(r"C:\Dev\ai-agents\supabase\functions\audit-stage2\index.ts")
body    = fn_path.read_text(encoding="utf-8")

# Create/update function
r = requests.post(
    f"https://api.supabase.com/v1/projects/{ref}/functions",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json={
        "slug": slug,
        "name": "audit-stage2",
        "verify_jwt": False,  # callers use service_role key, not user JWT
        "body": body,
    },
    timeout=60,
)
print(f"Create status: {r.status_code}")
print(f"Response: {r.text[:500]}")

if r.status_code in (200, 201):
    print(f"\nFunction deployed!")
    print(f"Endpoint: https://{ref}.supabase.co/functions/v1/{slug}")
elif r.status_code == 409:
    # Already exists — update it
    print("Function already exists, updating...")
    r2 = requests.patch(
        f"https://api.supabase.com/v1/projects/{ref}/functions/{slug}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"verify_jwt": False, "body": body},
        timeout=60,
    )
    print(f"Update status: {r2.status_code}")
    print(f"Response: {r2.text[:500]}")
    if r2.status_code == 200:
        print(f"\nFunction updated!")
        print(f"Endpoint: https://{ref}.supabase.co/functions/v1/{slug}")
    else:
        print("Update failed!", file=sys.stderr)
        sys.exit(1)
else:
    print("Deployment failed!", file=sys.stderr)
    sys.exit(1)
