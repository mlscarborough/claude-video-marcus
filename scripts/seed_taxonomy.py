"""Seed and export the /watch topic taxonomy.

Usage
-----
  # Seed taxonomy_nodes DB table from taxonomy.yaml (idempotent)
  python scripts/seed_taxonomy.py [--dry-run]

  # Export current DB state back to taxonomy.yaml (for version-control snapshots)
  python scripts/seed_taxonomy.py --export
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

TAXONOMY_FILE = Path(__file__).parent.parent / "taxonomy.yaml"


def _get_supabase():
    from dotenv import load_dotenv
    _env = Path.home() / ".config" / "watch" / ".env"
    if _env.exists():
        load_dotenv(_env)
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY not set in ~/.config/watch/.env")
    return create_client(url, key)


def seed_taxonomy_from_yaml(dry_run: bool = False) -> int:
    """Idempotently load taxonomy.yaml into taxonomy_nodes.

    Skips rows that already exist (by slug). Returns count of inserted rows.
    Safe to run multiple times — will not overwrite manual edits or AI-proposed
    nodes added after the initial seed.
    """
    sb = _get_supabase()
    tree = yaml.safe_load(TAXONOMY_FILE.read_text(encoding="utf-8"))

    # Collect existing slugs to avoid redundant inserts
    existing_resp = sb.table("taxonomy_nodes").select("slug").execute()
    existing = {r["slug"] for r in (existing_resp.data or [])}

    rows_to_insert: list[dict] = []

    def _walk(node, prefix: str, parent_slug: str | None, domain: str, depth: int):
        if isinstance(node, str):
            label = node
        elif isinstance(node, dict):
            label = node.get("label", prefix.split("/")[-1].replace("-", " ").title())
        else:
            return

        if prefix not in existing:
            rows_to_insert.append({
                "slug":        prefix,
                "label":       label,
                "parent_slug": parent_slug,
                "domain":      domain,
                "depth":       depth,
                "status":      "approved",
                "source":      "seed",
            })

        if isinstance(node, dict):
            for key, child in node.get("children", {}).items():
                _walk(child, f"{prefix}/{key}", prefix, domain, depth + 1)

    for domain_key, domain_node in tree.items():
        if isinstance(domain_node, dict):
            domain_label = domain_node.get("label", domain_key.replace("-", " ").title())
        else:
            domain_label = str(domain_node)

        if domain_key not in existing:
            rows_to_insert.append({
                "slug":        domain_key,
                "label":       domain_label,
                "parent_slug": None,
                "domain":      domain_key,
                "depth":       0,
                "status":      "approved",
                "source":      "seed",
            })

        if isinstance(domain_node, dict):
            for key, child in domain_node.get("children", {}).items():
                _walk(child, f"{domain_key}/{key}", domain_key, domain_key, 1)

    if dry_run:
        print(f"[seed] dry-run: {len(rows_to_insert)} rows would be inserted")
        for r in rows_to_insert[:30]:
            print(f"  {r['slug']}")
        if len(rows_to_insert) > 30:
            print(f"  … ({len(rows_to_insert) - 30} more)")
        return len(rows_to_insert)

    if not rows_to_insert:
        print("[seed] all taxonomy nodes already present — nothing to insert")
        return 0

    # Batch insert in chunks of 100
    inserted = 0
    for i in range(0, len(rows_to_insert), 100):
        chunk = rows_to_insert[i : i + 100]
        sb.table("taxonomy_nodes").insert(chunk).execute()
        inserted += len(chunk)

    print(f"[seed] inserted {inserted} taxonomy nodes from taxonomy.yaml")
    return inserted


def export_taxonomy_to_yaml() -> None:
    """Export current approved taxonomy_nodes DB state back to taxonomy.yaml.

    Used for version-control snapshots after Marcus approves new nodes.
    Rebuilds the nested YAML from the flat DB rows.
    """
    sb = _get_supabase()
    rows = (
        sb.table("taxonomy_nodes")
        .select("slug,label,parent_slug,depth,domain")
        .eq("status", "approved")
        .order("depth")
        .order("slug")
        .execute()
        .data
    )

    if not rows:
        print("[export] no approved nodes found — taxonomy.yaml not updated")
        return

    # Rebuild nested dict from flat rows
    tree: dict = {}
    node_map: dict[str, dict] = {}

    for r in rows:
        slug  = r["slug"]
        label = r["label"]
        depth = r["depth"]
        parts = slug.split("/")

        if depth == 0:
            tree[slug] = {"label": label, "children": {}}
            node_map[slug] = tree[slug]
        else:
            parent_key = r.get("parent_slug") or "/".join(parts[:-1])
            parent_node = node_map.get(parent_key)
            if parent_node is None:
                continue  # orphan — skip
            key = parts[-1]
            if "children" not in parent_node:
                parent_node["children"] = {}
            parent_node["children"][key] = {"label": label, "children": {}}
            node_map[slug] = parent_node["children"][key]

    # Collapse leaf nodes (no children) back to plain strings
    def _collapse(node):
        if not isinstance(node, dict):
            return
        children = node.get("children", {})
        for k, v in list(children.items()):
            _collapse(v)
            if isinstance(v, dict) and not v.get("children"):
                children[k] = v.get("label", k)
        if not children and "children" in node:
            del node["children"]

    for domain_node in tree.values():
        _collapse(domain_node)

    header = (
        "# /watch Topic Taxonomy — controlled vocabulary for multi-level video tagging\n"
        "# Auto-exported from taxonomy_nodes DB. Edit in Supabase; snapshot with:\n"
        "#   python scripts/seed_taxonomy.py --export\n\n"
    )
    TAXONOMY_FILE.write_text(
        header + yaml.dump(tree, default_flow_style=False, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[export] taxonomy.yaml updated ({len(rows)} nodes)")


def main():
    parser = argparse.ArgumentParser(description="Seed or export the /watch topic taxonomy")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview inserts without writing to DB")
    parser.add_argument("--export", action="store_true",
                        help="Export DB → taxonomy.yaml instead of seeding DB ← YAML")
    args = parser.parse_args()

    if args.export:
        export_taxonomy_to_yaml()
    else:
        seed_taxonomy_from_yaml(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
