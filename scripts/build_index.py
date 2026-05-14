"""build_index.py — wiki index rebuilder and taxonomy management CLI.

Usage
-----
Taxonomy management (Task 22.7):
  python scripts/build_index.py --review-proposals
  python scripts/build_index.py --approve <slug>
  python scripts/build_index.py --reject  <slug>
  python scripts/build_index.py --list-recent-taxonomy [--days N]
  python scripts/build_index.py --export-taxonomy
  python scripts/build_index.py --retag <slug> --keywords word1,word2,...

Hub page rebuild (Task 19, future):
  python scripts/build_index.py --rebuild-hubs
  python scripts/build_index.py --rebuild-hubs --domain real-estate
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
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
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY not set in ~/.config/watch/.env")
    return create_client(url, key)


# ── 22.7 Taxonomy management commands ─────────────────────────────────────────

def cmd_review_proposals(sb) -> None:
    """List all proposed taxonomy nodes, newest first."""
    rows = (
        sb.table("taxonomy_nodes")
        .select("slug,label,depth,notes,created_at")
        .eq("status", "proposed")
        .order("created_at", desc=True)
        .execute()
        .data
    )
    if not rows:
        print("No pending taxonomy proposals.")
        return

    print(f"\n{'PROPOSED TAXONOMY NODES':=^60}")
    print(f"  {'Slug':<42} {'Depth'}  {'Proposed'}")
    print(f"  {'-'*42} -----  ----------")
    for r in rows:
        dt = (r.get("created_at") or "")[:10]
        print(f"  {r['slug']:<42} {r['depth']:<5}  {dt}")
        notes = r.get("notes") or ""
        if notes:
            print(f"    context: {notes[:80]}")

    print(f"\nTotal: {len(rows)} proposal(s)")
    print(f"\n  Approve: python build_index.py --approve <slug>")
    print(f"  Reject:  python build_index.py --reject <slug>")


def cmd_approve(sb, slug: str) -> None:
    """Approve a proposed taxonomy node."""
    rows = (
        sb.table("taxonomy_nodes")
        .select("slug,status")
        .eq("slug", slug)
        .execute()
        .data
    )
    if not rows:
        print(f"ERROR: '{slug}' not found in taxonomy_nodes.")
        sys.exit(1)
    if rows[0]["status"] == "approved":
        print(f"'{slug}' is already approved — nothing to do.")
        return

    sb.table("taxonomy_nodes").update({"status": "approved"}).eq("slug", slug).execute()

    from taxonomy import refresh_taxonomy
    refresh_taxonomy()

    print(f"Approved: {slug}")
    print(f"  Cache refreshed. Next video processed will include this node.")
    print(f"  Tip: run --retag {slug} --keywords <words> to tag past videos.")


def cmd_reject(sb, slug: str) -> None:
    """Delete a proposed taxonomy node."""
    result = (
        sb.table("taxonomy_nodes")
        .delete()
        .eq("slug", slug)
        .eq("status", "proposed")
        .execute()
    )
    if result.data:
        print(f"Rejected and deleted: {slug}")
    else:
        print(f"Not found as a proposed node: '{slug}' (may already be approved or absent)")


def cmd_list_recent_taxonomy(sb, days: int = 7) -> None:
    """Show taxonomy nodes created in the last N days."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = (
        sb.table("taxonomy_nodes")
        .select("slug,label,depth,status,source,created_at")
        .gte("created_at", since)
        .order("created_at", desc=True)
        .execute()
        .data
    )
    if not rows:
        print(f"No taxonomy nodes added in the last {days} days.")
        return

    print(f"\n{'RECENT TAXONOMY NODES (last ' + str(days) + ' days)':=^60}")
    for r in rows:
        badge  = "+" if r["status"] == "approved" else "?"
        dt     = (r.get("created_at") or "")[:10]
        source = r.get("source", "")
        print(f"  {badge} [{source:<12}] depth={r['depth']}  {r['slug']}  ({dt})")


def cmd_export_taxonomy() -> None:
    """Export current approved DB nodes back to taxonomy.yaml."""
    from seed_taxonomy import export_taxonomy_to_yaml
    export_taxonomy_to_yaml()


def cmd_retag(sb, slug: str, keywords: list[str]) -> None:
    """Retroactively tag past videos whose transcripts mention any keyword.

    Scans transcript_segments for keyword matches. Inserts video_entities rows
    for each matching video at the given slug and all its ancestors. Idempotent.
    """
    from taxonomy import taxonomy_expand, is_valid_slug

    if not is_valid_slug(slug):
        print(f"ERROR: '{slug}' is not an approved taxonomy slug.")
        print(f"  Run --review-proposals or --approve first.")
        sys.exit(1)

    slugs_to_add = taxonomy_expand(slug)
    keyword_patterns = [kw.lower().strip() for kw in keywords if kw.strip()]

    if not keyword_patterns:
        print("ERROR: --keywords requires at least one keyword.")
        sys.exit(1)

    print(f"[retag] scanning transcripts for: {keyword_patterns}")
    print(f"[retag] will tag at levels: {slugs_to_add}")

    tagged_count = 0
    page_size = 500
    offset = 0
    processed_video_ids: set[str] = set()

    while True:
        segs = (
            sb.table("transcript_segments")
            .select("video_id, text")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        if not segs:
            break

        # Group text by video_id
        vid_texts: dict[str, list[str]] = {}
        for s in segs:
            vid_id = s["video_id"]
            if vid_id not in processed_video_ids:
                vid_texts.setdefault(vid_id, []).append((s.get("text") or "").lower())

        for video_id, texts in vid_texts.items():
            processed_video_ids.add(video_id)
            combined = " ".join(texts)
            if not any(kw in combined for kw in keyword_patterns):
                continue

            # Find which slugs are already present for this video
            existing = {
                r["entity_value"]
                for r in (
                    sb.table("video_entities")
                    .select("entity_value")
                    .eq("video_id", video_id)
                    .eq("entity_type", "topic")
                    .execute()
                    .data
                )
            }

            new_rows = [
                {"video_id": video_id, "entity_type": "topic", "entity_value": s}
                for s in slugs_to_add
                if s not in existing
            ]
            if new_rows:
                sb.table("video_entities").insert(new_rows).execute()
                tagged_count += 1
                print(f"  [retag] tagged video {video_id} with {len(new_rows)} new slug(s)")

        offset += page_size

    print(f"[retag] done — {tagged_count} video(s) retroactively tagged with '{slug}'")


# ── Main dispatch ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="build_index.py — wiki index rebuilder and taxonomy management"
    )

    # Taxonomy management
    tax = parser.add_argument_group("Taxonomy management (Task 22.7)")
    tax.add_argument("--review-proposals",      action="store_true",
                     help="List all proposed taxonomy nodes awaiting review")
    tax.add_argument("--approve",               metavar="SLUG",
                     help="Approve a proposed taxonomy node")
    tax.add_argument("--reject",                metavar="SLUG",
                     help="Delete a proposed taxonomy node")
    tax.add_argument("--list-recent-taxonomy",  action="store_true",
                     help="Show taxonomy nodes added in the last N days")
    tax.add_argument("--days",                  type=int, default=7,
                     help="Number of days for --list-recent-taxonomy (default: 7)")
    tax.add_argument("--export-taxonomy",       action="store_true",
                     help="Export DB taxonomy state back to taxonomy.yaml")
    tax.add_argument("--retag",                 metavar="SLUG",
                     help="Retroactively tag past videos matching --keywords")
    tax.add_argument("--keywords",              metavar="WORD1,WORD2",
                     help="Comma-separated keywords for --retag transcript scan")

    args = parser.parse_args()

    # Route to taxonomy commands (no Supabase needed for export)
    if args.export_taxonomy:
        cmd_export_taxonomy()
        return

    # All other commands need Supabase
    sb = _get_supabase()

    if args.review_proposals:
        cmd_review_proposals(sb)
    elif args.approve:
        cmd_approve(sb, args.approve)
    elif args.reject:
        cmd_reject(sb, args.reject)
    elif args.list_recent_taxonomy:
        cmd_list_recent_taxonomy(sb, days=args.days)
    elif args.retag:
        keywords = [k.strip() for k in (args.keywords or "").split(",") if k.strip()]
        cmd_retag(sb, args.retag, keywords)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
