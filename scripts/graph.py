"""Knowledge graph builder for the /watch wiki.

Task 18 — Write-time cross-linking:
  compute_semantic_links(video_id, sb, top_k)   → list[dict]
  compute_entity_links(video_id, sb, max_per)   → dict[str, list[dict]]
  inject_links_into_page(wiki_path, sem, ent)   → None

Task 19 — Hub page generation:
  build_creator_page(creator, vault_path, sb)   → Path
  build_topic_page(topic, vault_path, sb)        → Path
  build_entity_hub_page(entity_type, entity_value, vault_path, sb) → Path
  rebuild_affected_hubs(video_id, vault_path, sb)  → None

Task 19.5 — Full rebuild (called from build_index.py):
  rebuild_all_hubs(vault_path, sb, domain)      → None
"""
from __future__ import annotations

import re
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Task 18.1: Semantic similarity links ──────────────────────────────────────

def compute_semantic_links(
    video_id: str,
    sb,
    top_k: int = 5,
) -> list[dict]:
    """Return top_k most semantically similar videos to video_id.

    Each result: {"video_id", "title", "creator", "wiki_path", "similarity"}.
    Returns [] if the video has no embeddings yet.
    """
    # Fetch this video's embeddings
    rows = sb.table("wiki_embeddings").select("embedding").eq("video_id", video_id).execute()
    if not rows.data:
        return []

    # Average all chunk vectors into a single representative query vector.
    # Supabase returns pgvector values as a string "[f1,f2,...]" — parse to list.
    import json as _json
    def _parse_vec(v):
        if isinstance(v, list):
            return v
        # "[0.1,0.2,...]" → list of floats
        return _json.loads(v.replace("(", "[").replace(")", "]"))

    vectors = [_parse_vec(r["embedding"]) for r in rows.data if r.get("embedding") is not None]
    if not vectors:
        return []
    n = len(vectors)
    dim = len(vectors[0])
    avg_vector = [sum(v[i] for v in vectors) / n for i in range(dim)]

    # Call the find_similar_videos Postgres function (defined in migration)
    result = sb.rpc(
        "find_similar_videos",
        {
            "query_embedding": avg_vector,
            "exclude_video_id": video_id,
            "match_count": top_k,
        }
    ).execute()

    return result.data or []


# ── Task 18.2: Entity co-occurrence links ─────────────────────────────────────

LINKABLE_ENTITY_TYPES = {"ticker", "topic", "company", "person", "model", "technology"}


def compute_entity_links(
    video_id: str,
    sb,
    max_per_entity: int = 3,
    max_entities: int = 15,
) -> dict[str, list[dict]]:
    """Return {entity_value: [video dicts]} for entities shared with other videos.

    Only includes LINKABLE_ENTITY_TYPES. Excludes video_id itself.
    Capped at max_per_entity results per entity, and max_entities total entities.
    """
    # Fetch this video's linkable entities
    my_entities_resp = (
        sb.table("video_entities")
        .select("entity_type,entity_value")
        .eq("video_id", video_id)
        .in_("entity_type", list(LINKABLE_ENTITY_TYPES))
        .execute()
    )
    if not my_entities_resp.data:
        return {}

    # Skip very broad topic ancestors (depth 0-1) to avoid every video linking
    # to every other via "real-estate" or "trading" — link at depth>=2 only
    def _is_specific_enough(etype: str, val: str) -> bool:
        if etype == "topic":
            return val.count("/") >= 2  # depth 2+ (e.g. real-estate/multi-family/acquisition)
        return True

    result: dict[str, list[dict]] = {}
    processed = 0

    for entity in my_entities_resp.data:
        if processed >= max_entities:
            break
        etype = entity["entity_type"]
        val   = entity["entity_value"]
        if not _is_specific_enough(etype, val):
            continue

        matches = (
            sb.table("video_entities")
            .select("video_id, videos(title, creator, wiki_path)")
            .eq("entity_type", etype)
            .eq("entity_value", val)
            .neq("video_id", video_id)
            .limit(max_per_entity)
            .execute()
        )
        if matches.data:
            videos = [
                {
                    "video_id":  r["video_id"],
                    "title":     (r.get("videos") or {}).get("title", "?"),
                    "creator":   (r.get("videos") or {}).get("creator", "?"),
                    "wiki_path": (r.get("videos") or {}).get("wiki_path", ""),
                }
                for r in matches.data
                if r.get("videos")
            ]
            if videos:
                result[val] = videos
                processed += 1

    return result


# ── Task 18.3: Inject links into wiki page ────────────────────────────────────

_SENTINEL_RE = re.compile(
    r"<!-- watch:related-start -->.*?<!-- watch:related-end -->",
    flags=re.DOTALL,
)


def inject_links_into_page(
    wiki_path: Path,
    semantic_links: list[dict],
    entity_links: dict[str, list[dict]],
) -> None:
    """Inject ## Related videos and ## Also covers into wiki_path.

    Idempotent: strips existing injected block before rewriting.
    Updates YAML frontmatter with related_videos list.
    No-ops if both inputs are empty or wiki_path doesn't exist.
    """
    if not wiki_path.exists():
        return
    if not semantic_links and not entity_links:
        return

    content = wiki_path.read_text(encoding="utf-8")

    # ── Update frontmatter related_videos list ────────────────────────────────
    fm_match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", content, re.DOTALL)
    if fm_match:
        fm_text = fm_match.group(1)
        fm_data = yaml.safe_load(fm_text) or {}
        fm_data["related_videos"] = [
            r["wiki_path"] for r in semantic_links if r.get("wiki_path")
        ]
        new_fm = yaml.dump(fm_data, allow_unicode=True, default_flow_style=False)
        content = f"---\n{new_fm}---\n" + content[fm_match.end():]

    # ── Strip any previously injected block (idempotency) ────────────────────
    content = _SENTINEL_RE.sub("", content).rstrip()

    # ── Build ## Related videos ───────────────────────────────────────────────
    related_md = ""
    if semantic_links:
        related_md += "\n\n<!-- watch:related-start -->\n## Related videos\n\n"
        related_md += "_Semantic similarity — most related content in the knowledge base:_\n\n"
        for link in semantic_links:
            wiki_rel = link.get("wiki_path", "")
            title    = link.get("title", "Unknown")
            creator  = link.get("creator", "")
            sim_pct  = int(link.get("similarity", 0) * 100)
            # Obsidian resolves wikilinks by stem — strip folder and extension
            obs_link = Path(wiki_rel).stem if wiki_rel else title
            related_md += f"- [[{obs_link}]] — {creator} ({sim_pct}% match)\n"

    # ── Build ## Also covers ─────────────────────────────────────────────────
    if entity_links:
        if not related_md:
            related_md += "\n\n<!-- watch:related-start -->"
        related_md += "\n## Also covers\n\n"
        related_md += "_Videos sharing specific topics or entities with this one:_\n\n"
        for entity_val, videos in entity_links.items():
            related_md += f"**{entity_val}**\n"
            for v in videos:
                obs_link = Path(v["wiki_path"]).stem if v.get("wiki_path") else v["title"]
                related_md += f"- [[{obs_link}]] — {v['creator']}\n"
            related_md += "\n"

    if related_md and not related_md.rstrip().endswith("<!-- watch:related-end -->"):
        related_md = related_md.rstrip() + "\n<!-- watch:related-end -->"

    wiki_path.write_text(content + related_md, encoding="utf-8")


# ── Task 18 entry point ───────────────────────────────────────────────────────

def build_graph(video_id: str, wiki_path: Path, sb, top_k: int = 5) -> tuple[int, int]:
    """Compute and inject all graph links for a newly persisted video.

    Returns (n_semantic, n_entity) counts for the caller to log.
    Designed to be called best-effort — caller wraps in try/except.
    """
    sem_links    = compute_semantic_links(video_id, sb, top_k=top_k)
    entity_links = compute_entity_links(video_id, sb)
    inject_links_into_page(wiki_path, sem_links, entity_links)
    return len(sem_links), len(entity_links)


# ── Task 19.1: Creator hub pages ──────────────────────────────────────────────

def build_creator_page(creator: str, vault_path: Path, sb) -> Path:
    """Generate or overwrite creators/{creator-slug}.md.

    Lists all processed videos by this creator in reverse-chronological order.
    Shows top topics and total video count.
    """
    from persist import _slugify

    rows = (
        sb.table("videos")
        .select("id,title,wiki_path,processed_at,duration_seconds")
        .eq("creator", creator)
        .order("processed_at", desc=True)
        .execute()
    )

    slug      = _slugify(creator, max_chars=40)
    page_dir  = vault_path / "creators"
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    video_ids = [r["id"] for r in (rows.data or [])]
    topics: list[str] = []
    if video_ids:
        ent_rows = (
            sb.table("video_entities")
            .select("entity_value")
            .in_("video_id", video_ids)
            .eq("entity_type", "topic")
            .execute()
        )
        topics = sorted(set(r["entity_value"] for r in (ent_rows.data or [])))

    table_lines = []
    for r in (rows.data or []):
        date    = (r.get("processed_at") or "")[:10]
        title   = r.get("title") or "Untitled"
        dur_s   = int(r.get("duration_seconds") or 0)
        dur_str = f"{dur_s // 60}:{dur_s % 60:02d}"
        wp      = r.get("wiki_path") or ""
        link    = f"[[{Path(wp).stem}]]" if wp else title
        table_lines.append(f"| {date} | {link} | {dur_str} |")

    n = len(rows.data or [])
    topics_str = ", ".join(f"`{t}`" for t in topics[:12]) if topics else "_none identified_"

    fm_data = {
        "type":         "creator_index",
        "creator":      creator,
        "video_count":  n,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "tags":         ["index", "creator"],
    }
    fm_text = yaml.dump(fm_data, allow_unicode=True, default_flow_style=False)

    body = (
        f"# {creator}\n\n"
        f"*{n} video{'s' if n != 1 else ''} processed.*\n"
        f"*Auto-generated hub page. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        f"**Topics:** {topics_str}\n\n"
        f"## Videos\n\n"
        f"| Date | Title | Duration |\n"
        f"|------|-------|----------|\n"
    ) + "\n".join(table_lines)

    page_path.write_text(f"---\n{fm_text}---\n\n{body}\n", encoding="utf-8")
    return page_path


# ── Task 19.2: Topic hub pages ────────────────────────────────────────────────

def build_topic_page(topic: str, vault_path: Path, sb) -> Path:
    """Generate or overwrite topics/{topic-slug}.md.

    For taxonomy-path topics (e.g. "real-estate/multi-family/acquisition"):
    - Shows child sub-topics as navigation links
    - Aggregates ALL videos in the subtree (LIKE prefix match)
    For leaf topics (no children): lists exact matches only.
    """
    try:
        from taxonomy import taxonomy_label, taxonomy_children
        label    = taxonomy_label(topic)
        children = taxonomy_children(topic)
    except Exception:
        label    = topic.split("/")[-1].replace("-", " ").title()
        children = []

    slug_for_file = topic.replace("/", "-")
    page_dir      = vault_path / "topics"
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path     = page_dir / f"{slug_for_file}.md"

    # Fetch videos tagged at this exact level
    exact_rows = (
        sb.table("video_entities")
        .select("video_id, videos(title, creator, wiki_path, processed_at)")
        .eq("entity_type", "topic")
        .eq("entity_value", topic)
        .execute()
    )

    # Fetch videos in the subtree (any descendant slug)
    prefix_rows = (
        sb.table("video_entities")
        .select("video_id, videos(title, creator, wiki_path, processed_at)")
        .eq("entity_type", "topic")
        .like("entity_value", f"{topic}/%")
        .execute()
    )

    # Merge + dedup by video_id
    seen: set[str] = set()
    all_rows: list[dict] = []
    for r in (exact_rows.data or []) + (prefix_rows.data or []):
        vid = r.get("videos")
        if not vid or r["video_id"] in seen:
            continue
        seen.add(r["video_id"])
        all_rows.append({"video_id": r["video_id"], "videos": vid})

    all_rows.sort(
        key=lambda r: r["videos"].get("processed_at", ""),
        reverse=True,
    )

    fm_data = {
        "type":         "topic_index",
        "topic":        topic,
        "label":        label,
        "video_count":  len(all_rows),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "tags":         ["index", "topic"],
    }
    fm_text = yaml.dump(fm_data, allow_unicode=True, default_flow_style=False)

    lines = [
        f"# {label}",
        f"",
        f"*Auto-generated hub page. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*Topic path: `{topic}`*",
        f"",
    ]

    if children:
        lines += ["## Sub-topics", ""]
        for child in sorted(children):
            try:
                from taxonomy import taxonomy_label as tl
                cl = tl(child)
            except Exception:
                cl = child.split("/")[-1].replace("-", " ").title()
            child_file = child.replace("/", "-")
            lines.append(f"- [[topics/{child_file}|{cl}]]")
        lines.append("")

    if all_rows:
        lines += [
            f"## Videos ({len(all_rows)})",
            "",
            "| Date | Title | Creator |",
            "|------|-------|---------|",
        ]
        for r in all_rows:
            v   = r["videos"]
            dt  = (v.get("processed_at") or "")[:10]
            wp  = v.get("wiki_path", "")
            lnk = f"[[{Path(wp).stem}|{v.get('title','?')}]]" if wp else v.get("title", "?")
            lines.append(f"| {dt} | {lnk} | {v.get('creator','?')} |")
    else:
        lines.append("*No videos tagged with this topic yet.*")

    page_path.write_text(
        f"---\n{fm_text}---\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return page_path


# ── Task 19.3: Generic entity hub pages ───────────────────────────────────────

ENTITY_HUB_CONFIG: dict[str, dict] = {
    # entity_type → {folder, id_field (for frontmatter type key)}
    "ticker":     {"folder": "tickers",      "icon": "📈"},
    "location":   {"folder": "locations",    "icon": "📍"},
    "technology": {"folder": "technologies", "icon": "⚙️"},
    "property":   {"folder": "properties",   "icon": "🏠"},
    "company":    {"folder": "companies",    "icon": "🏢"},
    "person":     {"folder": "people",       "icon": "👤"},
    "model":      {"folder": "models",       "icon": "🤖"},
    "concept":    {"folder": "concepts",     "icon": "💡"},
    "indicator":  {"folder": "indicators",   "icon": "📊"},
    "market":     {"folder": "markets",      "icon": "🌐"},
    "keyword":    {"folder": "keywords",     "icon": "🔑"},
}


def build_entity_hub_page(entity_type: str, entity_value: str,
                           vault_path: Path, sb) -> Path:
    """Generate or overwrite a hub page for any entity type.

    Resolves the output folder from ENTITY_HUB_CONFIG; falls back to
    a generic 'entities/' directory for unmapped types.
    """
    from persist import _slugify

    cfg      = ENTITY_HUB_CONFIG.get(entity_type, {"folder": "entities", "icon": ""})
    folder   = cfg["folder"]
    icon     = cfg.get("icon", "")
    slug     = _slugify(entity_value, max_chars=50)
    page_dir = vault_path / folder
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    rows = (
        sb.table("video_entities")
        .select("video_id, videos(title, creator, wiki_path, processed_at)")
        .eq("entity_type", entity_type)
        .eq("entity_value", entity_value)
        .execute()
    )

    seen: set[str] = set()
    vids: list[dict] = []
    for r in (rows.data or []):
        v = r.get("videos")
        if not v or r["video_id"] in seen:
            continue
        seen.add(r["video_id"])
        vids.append({"video_id": r["video_id"], "videos": v})

    vids.sort(key=lambda r: r["videos"].get("processed_at", ""), reverse=True)

    n = len(vids)
    display_name = f"{icon} {entity_value}".strip()

    fm_data = {
        "type":         f"{entity_type}_index",
        "entity_type":  entity_type,
        "entity_value": entity_value,
        "video_count":  n,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "tags":         ["index", entity_type],
    }
    fm_text = yaml.dump(fm_data, allow_unicode=True, default_flow_style=False)

    lines = [
        f"# {display_name}",
        f"",
        f"*Auto-generated hub page. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*Entity: `{entity_type}` / `{entity_value}`*",
        f"",
    ]

    if vids:
        lines += [
            f"## Videos ({n})",
            "",
            "| Date | Title | Creator |",
            "|------|-------|---------|",
        ]
        for r in vids:
            v   = r["videos"]
            dt  = (v.get("processed_at") or "")[:10]
            wp  = v.get("wiki_path", "")
            lnk = f"[[{Path(wp).stem}|{v.get('title','?')}]]" if wp else v.get("title", "?")
            lines.append(f"| {dt} | {lnk} | {v.get('creator','?')} |")
    else:
        lines.append(f"*No videos tagged with `{entity_value}` yet.*")

    page_path.write_text(
        f"---\n{fm_text}---\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return page_path


# ── Task 19.4: rebuild affected hubs (called after each persist) ──────────────

def rebuild_affected_hubs(video_id: str, vault_path: Path, sb) -> None:
    """Rebuild all hub pages affected by a newly persisted video.

    Fetches the video's creator + all entities, then calls the appropriate
    hub builder for each. Best-effort — errors are logged, not re-raised.
    """
    import sys

    # Creator hub
    vid_row = sb.table("videos").select("creator").eq("id", video_id).execute().data
    if vid_row:
        creator = vid_row[0].get("creator", "")
        if creator:
            try:
                build_creator_page(creator, vault_path, sb)
            except Exception as exc:
                print(f"[graph] creator hub failed ({creator}): {exc}", file=sys.stderr)

    # Entity hubs — one hub page per distinct entity on this video
    entity_rows = (
        sb.table("video_entities")
        .select("entity_type,entity_value")
        .eq("video_id", video_id)
        .execute()
        .data
    ) or []

    for row in entity_rows:
        etype = row["entity_type"]
        eval_ = row["entity_value"]
        try:
            if etype == "topic":
                build_topic_page(eval_, vault_path, sb)
            else:
                build_entity_hub_page(etype, eval_, vault_path, sb)
        except Exception as exc:
            print(f"[graph] hub failed ({etype}/{eval_}): {exc}", file=sys.stderr)


# ── Task 19.5: Full hub rebuild (called from build_index.py --rebuild-hubs) ──

def _paginate(query_fn) -> list[dict]:
    """Fetch all rows by paginating a Supabase query in 1000-row batches.

    query_fn must be a zero-arg callable that returns a fresh (un-executed)
    Supabase query each time — e.g. ``lambda: sb.table("t").select("col")``.
    The Supabase default cap is 1000 rows; without pagination large knowledge
    bases would silently truncate (Task 19.5 bug).
    """
    all_rows: list[dict] = []
    offset    = 0
    page_size = 1000
    while True:
        rows = query_fn().range(offset, offset + page_size - 1).execute().data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


def rebuild_all_hubs(vault_path: Path, sb, domain: Optional[str] = None) -> None:
    """Rebuild every creator, topic, and entity hub page from current DB state.

    Typically called after: approving new taxonomy nodes, bulk imports, or
    schema migrations that changed entity rows.
    Pass domain= to restrict to a single taxonomy domain (e.g. "real-estate").
    """
    import sys

    # ── Creator pages ─────────────────────────────────────────────────────────
    creators = _paginate(lambda: sb.table("videos").select("creator"))
    unique_creators = {r["creator"] for r in creators if r.get("creator")}
    for c in sorted(unique_creators):
        try:
            build_creator_page(c, vault_path, sb)
        except Exception as exc:
            print(f"[graph] creator hub error ({c}): {exc}", file=sys.stderr)
    print(f"[graph] rebuilt {len(unique_creators)} creator hub(s)")

    # ── Topic pages ───────────────────────────────────────────────────────────
    if domain:
        topic_rows = _paginate(
            lambda: (
                sb.table("video_entities")
                .select("entity_value")
                .eq("entity_type", "topic")
                .like("entity_value", f"{domain}%")
            )
        )
    else:
        topic_rows = _paginate(
            lambda: sb.table("video_entities").select("entity_value").eq("entity_type", "topic")
        )
    unique_topics = {r["entity_value"] for r in topic_rows}
    for t in sorted(unique_topics):
        try:
            build_topic_page(t, vault_path, sb)
        except Exception as exc:
            print(f"[graph] topic hub error ({t}): {exc}", file=sys.stderr)
    print(f"[graph] rebuilt {len(unique_topics)} topic hub(s)")

    # ── Entity hub pages (all non-topic types) ────────────────────────────────
    non_topic_rows = _paginate(
        lambda: (
            sb.table("video_entities")
            .select("entity_type,entity_value")
            .neq("entity_type", "topic")
        )
    )
    unique_entities = {(r["entity_type"], r["entity_value"]) for r in non_topic_rows}
    for etype, eval_ in sorted(unique_entities):
        try:
            build_entity_hub_page(etype, eval_, vault_path, sb)
        except Exception as exc:
            print(f"[graph] entity hub error ({etype}/{eval_}): {exc}", file=sys.stderr)
    print(f"[graph] rebuilt {len(unique_entities)} entity hub(s)")
