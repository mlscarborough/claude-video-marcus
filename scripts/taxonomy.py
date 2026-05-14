"""Controlled topic taxonomy for the /watch skill.

This module is the single point of access for taxonomy data. It reads from
the taxonomy_nodes Supabase table (the live source of truth), caches results
for 5 minutes, and falls back to taxonomy.yaml when the DB is unreachable.

Public API (22.1 — read helpers)
---------------------------------
  load_taxonomy()              -> dict[str, str]   {slug: label} all approved nodes
  refresh_taxonomy()                               force-expire the 5-min cache
  is_valid_slug(slug)          -> bool
  taxonomy_label(slug)         -> str              human-readable label
  taxonomy_expand(slug)        -> list[str]        leaf + all valid ancestors
  taxonomy_children(slug)      -> list[str]        direct children of a node
  get_taxonomy_excerpt(...)    -> str              compact text for AI prompt injection
  find_similar_slug(label, domain) -> Optional[str]  SequenceMatcher dedup

Dynamic growth (22.6 — expansion engine)
-----------------------------------------
  expand_or_propose(slug, context) -> str
      Unknown slug? Tries similarity match, then tiered auto-accept/propose.
      Always returns a slug that is safe to tag with.

Tiered acceptance:
  depth >= 3  auto-accept (leaf concept, low collision risk)
  depth == 2  auto-accept + warning logged
  depth == 1  queued as 'proposed'; video tagged with parent
  depth == 0  queued as 'proposed'; video tagged with raw slug
"""
from __future__ import annotations

import logging
import os
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import yaml

TAXONOMY_FILE = Path(__file__).parent.parent / "taxonomy.yaml"
_CACHE_TTL = 300  # 5 minutes — avoids per-video DB round-trips

_taxonomy_cache: dict[str, str] = {}
_cache_loaded_at: float = 0.0

_log = logging.getLogger("watch.taxonomy")


# ── Supabase client ────────────────────────────────────────────────────────────

def _get_supabase():
    from dotenv import load_dotenv
    _env = Path.home() / ".config" / "watch" / ".env"
    if _env.exists():
        load_dotenv(_env)
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL or SUPABASE_SERVICE_KEY not set in ~/.config/watch/.env"
        )
    return create_client(url, key)


# ── YAML fallback loader ───────────────────────────────────────────────────────

def _load_from_yaml() -> dict[str, str]:
    """Walk taxonomy.yaml and return {slug: label} flat map."""
    if not TAXONOMY_FILE.exists():
        _log.warning("[taxonomy] taxonomy.yaml not found — returning empty taxonomy")
        return {}

    tree = yaml.safe_load(TAXONOMY_FILE.read_text(encoding="utf-8"))
    out: dict[str, str] = {}

    def _walk(node, prefix: str):
        if isinstance(node, str):
            out[prefix] = node
            return
        if not isinstance(node, dict):
            return
        label = node.get("label", prefix.split("/")[-1].replace("-", " ").title())
        out[prefix] = label
        for key, child in node.get("children", {}).items():
            _walk(child, f"{prefix}/{key}")

    for domain_key, domain_node in tree.items():
        _walk(domain_node, domain_key)

    return out


# ── Core loader (DB-primary, YAML fallback, 5-min TTL cache) ──────────────────

def load_taxonomy() -> dict[str, str]:
    """Return {slug: label} for all approved taxonomy nodes.

    Reads from taxonomy_nodes Supabase table (5-min TTL cache).
    Falls back to taxonomy.yaml if DB is unreachable.
    """
    global _taxonomy_cache, _cache_loaded_at

    if _taxonomy_cache and (time.monotonic() - _cache_loaded_at) < _CACHE_TTL:
        return _taxonomy_cache

    try:
        sb = _get_supabase()
        rows = (
            sb.table("taxonomy_nodes")
            .select("slug,label")
            .eq("status", "approved")
            .execute()
        )
        if rows.data:
            _taxonomy_cache = {r["slug"]: r["label"] for r in rows.data}
            _cache_loaded_at = time.monotonic()
            return _taxonomy_cache
    except Exception as exc:
        _log.warning("[taxonomy] DB read failed (%s) — falling back to YAML", exc)

    # YAML fallback: read-only laptop, DB unreachable, or first-install before seed
    _taxonomy_cache = _load_from_yaml()
    _cache_loaded_at = time.monotonic()
    return _taxonomy_cache


def refresh_taxonomy() -> None:
    """Force cache expiry. Call after approving or inserting taxonomy nodes."""
    global _cache_loaded_at
    _cache_loaded_at = 0.0


# ── Read helpers ───────────────────────────────────────────────────────────────

def is_valid_slug(slug: str) -> bool:
    """True if slug exists as an approved taxonomy node."""
    return slug in load_taxonomy()


def taxonomy_label(slug: str) -> str:
    """Human-readable label for a slug, or a title-cased fallback."""
    return load_taxonomy().get(slug, slug.split("/")[-1].replace("-", " ").title())


def taxonomy_expand(slug: str) -> list[str]:
    """Expand a leaf slug to [slug + all valid ancestors], shallowest first.

    Example:
        taxonomy_expand("real-estate/multi-family/acquisition/hard-money")
        → ["real-estate",
           "real-estate/multi-family",
           "real-estate/multi-family/acquisition",
           "real-estate/multi-family/acquisition/hard-money"]

    Only returns levels that exist as approved nodes. Unrecognised intermediate
    levels are skipped so the hierarchy stays coherent even as it grows.
    Returns [slug] as a single-element list if nothing matches (graceful fallback).
    """
    if not slug:
        return []
    all_slugs = load_taxonomy()
    parts = slug.split("/")
    result = [
        "/".join(parts[:i])
        for i in range(1, len(parts) + 1)
        if "/".join(parts[:i]) in all_slugs
    ]
    return result if result else [slug]


def taxonomy_children(slug: str) -> list[str]:
    """Return immediate child slugs of a node (direct children only)."""
    target_depth = slug.count("/") + 1
    prefix = slug + "/"
    return sorted(
        s for s in load_taxonomy()
        if s.startswith(prefix) and s.count("/") == target_depth
    )


def get_taxonomy_excerpt(domains: Optional[list[str]] = None,
                         max_lines: int = 80) -> str:
    """Compact indented-tree text for injection into the AI prompt.

    Filters to specific top-level domains if provided (e.g. ["real-estate"]).
    Falls back to all domains if None. Capped at max_lines to protect context budget.

    Example output (partial):
        real-estate (Real Estate)
          multi-family (Multi-Family Real Estate)
            acquisition (Acquisition)
              hard-money (Hard Money Lending)
              underwriting (Underwriting & Analysis)
    """
    all_slugs = load_taxonomy()
    lines: list[str] = []

    for slug in sorted(all_slugs):
        if domains and not any(
            slug == d or slug.startswith(d + "/") for d in domains
        ):
            continue
        depth  = slug.count("/")
        indent = "  " * depth
        short  = slug.split("/")[-1]
        label  = all_slugs[slug]
        lines.append(f"{indent}{short} ({label})")
        if len(lines) >= max_lines:
            lines.append("  … (truncated — see taxonomy_nodes table for full list)")
            break

    return "\n".join(lines)


def find_similar_slug(proposed_label: str, domain: str,
                      threshold: float = 0.80) -> Optional[str]:
    """Return an existing approved slug whose label is ≥ threshold similar.

    Used to prevent fragmentation: "managing tenants" maps to the existing
    "real-estate/multi-family/management/tenant-screening" rather than
    creating a duplicate concept under a different name.

    Compares only within the given domain to avoid cross-domain false matches.
    Uses SequenceMatcher (no embedding call — fast, sufficient for short labels).
    Returns None if no match found above the threshold.
    """
    all_slugs = load_taxonomy()
    best_slug: Optional[str] = None
    best_ratio = 0.0
    proposed_norm = proposed_label.lower().strip()

    for slug, label in all_slugs.items():
        if not (slug == domain or slug.startswith(domain + "/")):
            continue
        ratio = SequenceMatcher(None, proposed_norm, label.lower().strip()).ratio()
        if ratio >= threshold and ratio > best_ratio:
            best_ratio = ratio
            best_slug = slug

    return best_slug


# ── Dynamic expansion engine (22.6) ───────────────────────────────────────────

def expand_or_propose(proposed_slug: str, context: str = "") -> str:
    """Resolve an unknown slug to an approved (or proposed) taxonomy node.

    Returns the slug that should actually be used for tagging — may differ from
    proposed_slug if a similarity match or ancestor fallback was applied.

    Resolution order:
      1. Already valid → return as-is
      2. Similarity match (≥80%) → map to existing node, no write
      3. depth ≥ 3 (concept) → auto-accept, write approved node to DB
      4. depth = 2 (phase)   → auto-accept + log warning
      5. depth 0/1           → write as proposed, tag parent as fallback

    Args:
        proposed_slug: Slug from the AI (may not exist yet)
        context:       Short description/question for logging and DB notes

    Returns:
        A slug safe to tag with (guaranteed to be in approved taxonomy or the
        nearest approved ancestor if a proposal was queued).
    """
    # Fast path — already approved
    if is_valid_slug(proposed_slug):
        return proposed_slug

    # ── Step 1: string-similarity check ──────────────────────────────────────
    leaf_label = proposed_slug.split("/")[-1].replace("-", " ")
    domain     = proposed_slug.split("/")[0]
    similar    = find_similar_slug(leaf_label, domain, threshold=0.80)
    if similar:
        _log.info(
            "[taxonomy] similarity map '%s' -> '%s' (no new node written)",
            proposed_slug, similar,
        )
        return similar

    # ── Step 2: tiered auto-acceptance ────────────────────────────────────────
    parts  = proposed_slug.split("/")
    depth  = len(parts) - 1

    # Recursively ensure parent exists (stops at depth 0/1 which queue as proposed)
    parent_slug: Optional[str] = None
    if depth > 0:
        raw_parent = "/".join(parts[:-1])
        if is_valid_slug(raw_parent):
            parent_slug = raw_parent
        else:
            resolved_parent = expand_or_propose(raw_parent, context)
            parent_slug = resolved_parent if is_valid_slug(resolved_parent) else None

    if depth >= 3:
        _accept_node(
            proposed_slug, parent_slug, domain, depth,
            source="ai-proposed",
            notes=f"auto-accepted depth≥3; context: {context[:120]}",
        )
        _log.info("[taxonomy] auto-accepted new concept (depth≥3): %s", proposed_slug)
        return proposed_slug

    elif depth == 2:
        _accept_node(
            proposed_slug, parent_slug, domain, depth,
            source="ai-proposed",
            notes=f"auto-accepted depth=2; context: {context[:120]}",
        )
        _log.warning(
            "[taxonomy] auto-accepted NEW PHASE NODE (depth=2): %s — "
            "review with: python build_index.py --list-recent-taxonomy",
            proposed_slug,
        )
        return proposed_slug

    else:
        # depth 0 or 1: queue as proposed, return nearest approved ancestor
        _propose_node(proposed_slug, parent_slug, domain, depth,
                      notes=f"context: {context[:120]}")
        _log.warning(
            "[taxonomy] queued as PROPOSED (depth=%d): %s — "
            "approve with: python build_index.py --approve %s",
            depth, proposed_slug, proposed_slug,
        )
        return _nearest_approved_ancestor(proposed_slug)


def _accept_node(slug: str, parent_slug: Optional[str], domain: str,
                 depth: int, source: str, notes: str) -> None:
    """Insert or upsert an approved node and invalidate cache."""
    label = slug.split("/")[-1].replace("-", " ").title()
    try:
        sb = _get_supabase()
        sb.table("taxonomy_nodes").upsert(
            {
                "slug":        slug,
                "label":       label,
                "parent_slug": parent_slug,
                "domain":      domain,
                "depth":       depth,
                "status":      "approved",
                "source":      source,
                "notes":       notes,
            },
            on_conflict="slug",
        ).execute()
        refresh_taxonomy()
    except Exception as exc:
        _log.error("[taxonomy] failed to write node '%s': %s", slug, exc)


def _propose_node(slug: str, parent_slug: Optional[str], domain: str,
                  depth: int, notes: str) -> None:
    """Insert a proposed node (status='proposed') only if it doesn't exist yet."""
    label = slug.split("/")[-1].replace("-", " ").title()
    try:
        sb = _get_supabase()
        existing = (
            sb.table("taxonomy_nodes")
            .select("slug")
            .eq("slug", slug)
            .execute()
            .data
        )
        if not existing:
            sb.table("taxonomy_nodes").insert(
                {
                    "slug":        slug,
                    "label":       label,
                    "parent_slug": parent_slug,
                    "domain":      domain,
                    "depth":       depth,
                    "status":      "proposed",
                    "source":      "ai-proposed",
                    "notes":       notes,
                }
            ).execute()
    except Exception as exc:
        _log.error("[taxonomy] failed to queue proposal '%s': %s", slug, exc)


def _nearest_approved_ancestor(slug: str) -> str:
    """Walk up the slug path until we find an approved node. Returns domain at worst."""
    parts = slug.split("/")
    for i in range(len(parts) - 1, 0, -1):
        candidate = "/".join(parts[:i])
        if is_valid_slug(candidate):
            return candidate
    return parts[0]  # domain key — always return something


# ── CLI (quick sanity check) ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="taxonomy.py sanity checks")
    parser.add_argument("--stats",   action="store_true", help="Print node counts by depth")
    parser.add_argument("--expand",  metavar="SLUG",      help="Expand a slug to ancestors")
    parser.add_argument("--excerpt", metavar="DOMAIN",    help="Print taxonomy excerpt for a domain")
    parser.add_argument("--similar", nargs=2, metavar=("LABEL", "DOMAIN"),
                        help="Find similar slug for a label within a domain")
    args = parser.parse_args()

    tax = load_taxonomy()
    print(f"[taxonomy] loaded {len(tax)} approved nodes")

    if args.stats:
        from collections import Counter
        depths = Counter(slug.count("/") for slug in tax)
        for d in sorted(depths):
            print(f"  depth {d}: {depths[d]} nodes")

    if args.expand:
        print(f"\ntaxonomy_expand('{args.expand}'):")
        for s in taxonomy_expand(args.expand):
            print(f"  {s}  |  {taxonomy_label(s)}")

    if args.excerpt:
        print(f"\nExcerpt for domain '{args.excerpt}':")
        print(get_taxonomy_excerpt([args.excerpt], max_lines=40))

    if args.similar:
        label, domain = args.similar
        result = find_similar_slug(label, domain)
        print(f"\nfind_similar_slug('{label}', '{domain}') = {result}")
