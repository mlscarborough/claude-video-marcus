"""Sync Obsidian wiki edits back to Supabase.

CLI:
    python sync_wiki.py [--quick] [--full] [--file <path>]

--quick: only check files modified in the last 10 minutes (used from watch.py)
--full:  scan all watch-managed wiki files against DB (default)
--file:  sync a single specific file
"""
from __future__ import annotations

import argparse
import hashlib
import os
import socket
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
_env = Path.home() / ".config" / "watch" / ".env"
if _env.exists():
    load_dotenv(_env)


def _get_supabase():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
    return create_client(url, key)


def _vault_path() -> Path:
    vp = os.environ.get("OBSIDIAN_VAULT_PATH", "")
    if not vp:
        raise RuntimeError("OBSIDIAN_VAULT_PATH not set in ~/.config/watch/.env")
    return Path(vp).expanduser().resolve()


def _parse_iso(ts_str: str) -> datetime:
    """Parse ISO 8601 string to UTC-aware datetime."""
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def scan_for_changes(vault_path: Path, quick: bool = False) -> list[Path]:
    """Return watch-managed wiki files that have changed since last sync.

    quick=True: only inspect files modified in the last 10 minutes (saves DB round-trips).
    """
    sb = _get_supabase()
    candidates = list(vault_path.rglob("*.md"))

    if quick:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
        candidates = [
            p for p in candidates
            if datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) > cutoff
        ]

    changed = []
    for path in candidates:
        try:
            rel_path = str(path.relative_to(vault_path))
        except ValueError:
            continue

        row = (
            sb.table("videos")
            .select("id,last_wiki_synced_at,wiki_content_hash")
            .eq("wiki_path", rel_path)
            .limit(1)
            .execute()
        )
        if not row.data:
            continue  # not a /watch-managed file

        data = row.data[0]
        last_synced = _parse_iso(data["last_wiki_synced_at"])
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if mtime <= last_synced:
            continue

        # Double-check by hash to avoid spurious mtime-only changes
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if current_hash != data["wiki_content_hash"]:
            changed.append(path)

    return changed


def sync_file(path: Path, vault_path: Path | None = None) -> bool:
    """Sync one wiki file to Supabase. Returns True if changes were persisted."""
    if vault_path is None:
        vault_path = _vault_path()

    sb = _get_supabase()

    try:
        rel_path = str(path.relative_to(vault_path))
    except ValueError:
        rel_path = str(path)

    row = (
        sb.table("videos")
        .select("id,wiki_content")
        .eq("wiki_path", rel_path)
        .limit(1)
        .execute()
    )
    if not row.data:
        print(f"[sync-wiki] {rel_path}: not a watch-managed file, skipping", file=sys.stderr)
        return False

    video_id: str = row.data[0]["id"]
    old_content: str = row.data[0].get("wiki_content") or ""
    new_content: str = path.read_text(encoding="utf-8")
    new_hash = hashlib.sha256(new_content.encode()).hexdigest()

    sb.table("videos").update({
        "wiki_content": new_content,
        "wiki_content_hash": new_hash,
        "last_wiki_synced_at": datetime.now(timezone.utc).isoformat(),
        "wiki_edited_by_user": True,
    }).eq("id", video_id).execute()

    # Smart re-embed (only if content changed >= 20%)
    from persist import write_embeddings
    n_chunks = write_embeddings(video_id, path, old_content=old_content)
    re_embedded = n_chunks > 0

    # Task 20.1: Re-link on re-embed — refresh Related videos section (best-effort)
    re_linked = False
    if re_embedded:
        try:
            from graph import compute_semantic_links, compute_entity_links, inject_links_into_page
            semantic_links = compute_semantic_links(video_id, sb)
            entity_links   = compute_entity_links(video_id, sb)
            inject_links_into_page(path, semantic_links, entity_links)
            re_linked = True
            n_ent_links = sum(len(v) for v in entity_links.values())
            print(
                f"[sync-wiki] re-linked {rel_path}: "
                f"{len(semantic_links)} semantic, {n_ent_links} entity link(s)",
                file=sys.stderr,
            )
        except Exception as _link_err:
            print(f"[sync-wiki] re-link skipped (non-fatal): {_link_err}", file=sys.stderr)

    # wiki_sync_log row
    fields_updated = ["wiki_content", "wiki_content_hash", "last_wiki_synced_at", "wiki_edited_by_user"]
    if re_embedded:
        fields_updated.append("wiki_embeddings")
    if re_linked:
        fields_updated.append("related_videos_relinked")

    sb.table("wiki_sync_log").insert({
        "video_id": video_id,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "change_detected": True,
        "fields_updated": fields_updated,
        "triggered_by": "watcher",
        "hostname": socket.gethostname(),
    }).execute()

    print(
        f"[sync-wiki] synced {rel_path} "
        f"({'re-embedded ' + str(n_chunks) + ' chunks' if re_embedded else 'no re-embed needed'})",
        file=sys.stderr,
    )
    return True


def write_heartbeat() -> None:
    """Write a heartbeat row to wiki_sync_log (no-op if not writer laptop)."""
    if os.environ.get("WATCH_IS_WRITER_LAPTOP", "false").lower() != "true":
        return
    try:
        sb = _get_supabase()
        sb.table("wiki_sync_log").insert({
            "video_id": None,
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "change_detected": False,
            "fields_updated": [],
            "triggered_by": "heartbeat",
            "hostname": socket.gethostname(),
        }).execute()
    except Exception as e:
        print(f"[sync-wiki] heartbeat failed (non-fatal): {e}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(prog="sync-wiki")
    ap.add_argument("--quick", action="store_true", help="Only scan files modified in last 10 min")
    ap.add_argument("--full", action="store_true", help="Scan all watch-managed files (default)")
    ap.add_argument("--file", type=str, help="Sync a single specific file path")
    args = ap.parse_args()

    if os.environ.get("WATCH_IS_WRITER_LAPTOP", "false").lower() != "true":
        print("[sync-wiki] read-only laptop — skipping sync", file=sys.stderr)
        return 0

    vault = _vault_path()

    if args.file:
        path = Path(args.file).expanduser().resolve()
        if not path.exists():
            print(f"[sync-wiki] file not found: {path}", file=sys.stderr)
            return 1
        sync_file(path, vault)
        return 0

    changed = scan_for_changes(vault, quick=args.quick)
    if not changed:
        print("[sync-wiki] no changes detected", file=sys.stderr)
        return 0

    print(f"[sync-wiki] {len(changed)} file(s) to sync", file=sys.stderr)
    errors = 0
    for path in changed:
        try:
            sync_file(path, vault)
        except Exception as e:
            print(f"[sync-wiki] ERROR syncing {path}: {e}", file=sys.stderr)
            errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
