"""Three-stream persistence: wiki markdown, Supabase rows, pgvector embeddings.

Public API:
    persist_all(source_url, video_meta, ai_output, frame_results,
                transcript_segments, work_dir) -> dict

    parse_ai_output(raw_json: str, question: str) -> AIOutput
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

SCRIPT_DIR = Path(__file__).parent.resolve()

IS_WRITER = os.environ.get("WATCH_IS_WRITER_LAPTOP", "false").lower() == "true"


# ── Domain model ──────────────────────────────────────────────────────────────

VALID_ENTITY_TYPES = {
    # General — any domain
    "person",       # individual mentioned by name
    "company",      # business / organisation
    "topic",        # broad subject area ("options trading", "cap rates", "fine-tuning")
    "keyword",      # general term that doesn't fit a narrower type
    "concept",      # abstract idea ("attention mechanism", "cash-on-cash return")
    # Finance / trading
    "ticker",       # stock / options / crypto symbol (AAPL, SPY, BTC)
    "indicator",    # technical indicator (RSI, MACD, Bollinger Bands)
    "price_level",  # specific price / strike / target ($620, support at $580)
    # Real estate
    "property",     # specific property, address, or deal
    "location",     # geographic area — neighbourhood, city, metro market, zip code
    "market",       # RE market condition or sub-market ("Phoenix multifamily", "STR")
    # AI / technology
    "model",        # AI model name (GPT-4, Claude, Gemini, Llama-3)
    "technology",   # framework, tool, library, platform (PyTorch, LangChain, Supabase)
}


@dataclass
class Entity:
    entity_type: str
    entity_value: str
    mention_count: int = 1
    first_mentioned_at_seconds: Optional[float] = None
    sentiment: Optional[str] = None
    relevance: Optional[float] = None


@dataclass
class AIOutput:
    question: str
    answer: str
    sentiment_overall: Optional[str] = None
    confidence: float = 0.5
    entities: list[Entity] = field(default_factory=list)


# ── AI output parsing ─────────────────────────────────────────────────────────

def parse_ai_output(raw: str, question: str) -> AIOutput:
    """Parse Gemini JSON response (or plain text) into AIOutput.

    Gemini is asked to return JSON with {answer, sentiment_overall, confidence, entities}.
    If it returns plain prose, wraps it as-is with no entities.

    Topic entity handling (22.2b + 22.4):
    - Valid taxonomy slugs are expanded to all ancestor levels (multi-level tagging)
    - Unknown slugs are passed through expand_or_propose() which either:
        (a) maps to a similar existing node (similarity match)
        (b) auto-accepts as a new approved node (depth >= 2)
        (c) queues as proposed and falls back to nearest ancestor (depth 0/1)
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Plain-prose fallback
        return AIOutput(question=question, answer=raw)

    # Lazy imports — taxonomy is only needed on writer laptop when DB is available
    try:
        from taxonomy import is_valid_slug, taxonomy_expand, expand_or_propose
        _taxonomy_available = True
    except Exception:
        _taxonomy_available = False

    raw_entities = []
    for e in data.get("entities", []):
        etype = e.get("type", e.get("entity_type", "keyword")).lower()
        if etype not in VALID_ENTITY_TYPES:
            etype = "keyword"
        raw_entities.append(Entity(
            entity_type=etype,
            entity_value=str(e.get("value", e.get("entity_value", ""))).strip(),
            mention_count=int(e.get("mention_count", 1)),
            first_mentioned_at_seconds=e.get("first_mentioned_at_seconds"),
            sentiment=e.get("sentiment"),
            relevance=float(e.get("relevance", 0.5)) if e.get("relevance") is not None else None,
        ))

    # Expand topic slugs to all ancestor levels (multi-level tagging)
    entities: list[Entity] = []
    seen_topic_slugs: set[str] = set()

    for e in raw_entities:
        if e.entity_type != "topic" or not _taxonomy_available:
            entities.append(e)
            continue

        slug = e.entity_value

        # Skip blank or obviously non-slug values
        if not slug or len(slug) < 2:
            continue

        # Resolve unknown slugs through the expansion engine
        if not is_valid_slug(slug):
            slug = expand_or_propose(slug, context=question[:120])

        # Expand to all ancestor levels — each becomes its own entity row
        expanded_slugs = taxonomy_expand(slug)
        for s in expanded_slugs:
            if s in seen_topic_slugs:
                continue
            seen_topic_slugs.add(s)
            entities.append(Entity(
                entity_type="topic",
                entity_value=s,
                mention_count=e.mention_count,
                first_mentioned_at_seconds=e.first_mentioned_at_seconds,
                sentiment=e.sentiment,
                # Only preserve relevance on the most-specific (original) slug
                relevance=e.relevance if s == slug else None,
            ))

    return AIOutput(
        question=question,
        answer=data.get("answer", raw),
        sentiment_overall=data.get("sentiment_overall"),
        confidence=float(data.get("confidence", 0.5)),
        entities=entities,
    )


# ── Wiki markdown writer ──────────────────────────────────────────────────────

def _slugify(text: str, max_chars: int = 60) -> str:
    """Convert arbitrary text to a URL-safe slug, capped at max_chars."""
    text = (text or "unknown").lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:max_chars].strip("-")


# ── Duplicate-detection enums ─────────────────────────────────────────────────

class DuplicateResult(Enum):
    NEW      = "new"      # No meaningful match found
    LIKELY   = "likely"   # Partial match — needs human confirmation
    DEFINITE = "definite" # Clearly the same video


class DuplicateResolution(Enum):
    SKIP   = "skip"   # Keep existing entry; discard incoming
    UPDATE = "update" # Overwrite existing wiki + DB row with new analysis
    NEW    = "new"    # Create a separate entry regardless


# ── Fingerprint computation ───────────────────────────────────────────────────

def compute_transcript_fingerprint(transcript_segments: list[dict]) -> Optional[str]:
    """SHA-256 of the first 500 characters of combined transcript text.

    Returns None if no transcript is available.
    500 chars is long enough to be unique, short enough to be stable
    even if the end of the transcript differs (ads, outros, etc.).
    """
    if not transcript_segments:
        return None
    combined = " ".join(
        seg.get("text", "") for seg in transcript_segments
    ).lower().strip()
    if not combined:
        return None
    sample = combined[:500]
    return hashlib.sha256(sample.encode("utf-8")).hexdigest()


def compute_content_fingerprint(creator: str, duration_seconds: float, title: str) -> str:
    """SHA-256 of '{creator_norm}|{duration_rounded}|{title_norm}'.

    Cheap to compute (no transcript needed). Used for the fast DB pre-query.

    Normalisation rules:
    - creator:   lowercase, strip whitespace
    - duration:  rounded to nearest integer (handles float precision noise)
    - title:     lowercase, strip punctuation and whitespace
    """
    creator_norm  = (creator or "").lower().strip()
    duration_norm = str(int(round(duration_seconds or 0)))
    title_norm    = re.sub(r"[^\w\s]", "", (title or "").lower()).strip()
    raw           = f"{creator_norm}|{duration_norm}|{title_norm}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── Duplicate detection query ─────────────────────────────────────────────────

def find_potential_duplicate(
    sb,
    source_url: str,
    creator: str,
    duration_seconds: float,
    title: str,
    content_fingerprint: str,
    transcript_fingerprint: Optional[str],
) -> tuple[DuplicateResult, Optional[dict], str]:
    """Query videos table for potential duplicates.

    Returns (DuplicateResult, existing_row_or_None, match_explanation).

    Scoring system:
      Exact URL match        → 100 (bypass scoring, return DEFINITE immediately)
      content_fingerprint    →  70
      transcript_fingerprint →  50
      Same creator (norm)    →  30
      Duration within ±3s    →  40
      Title similarity ≥80%  →  20

    Thresholds:
      ≥90  → DEFINITE   (update or skip without prompting)
      50–89 → LIKELY    (show resolution prompt)
      <50   → NEW       (proceed normally)
    """
    # Step 0: Exact URL match — definite, no scoring needed
    exact = (
        sb.table("videos")
        .select("id,source_url,title,creator,duration_seconds,wiki_path,processed_at")
        .eq("source_url", source_url)
        .limit(1)
        .execute()
    )
    if exact.data:
        return DuplicateResult.DEFINITE, exact.data[0], "exact URL match"

    # Step 1: Fingerprint-based candidate fetch (two separate queries — avoids OR on nullable cols)
    candidates: list[dict] = []

    cf_result = (
        sb.table("videos")
        .select("id,source_url,title,creator,duration_seconds,wiki_path,processed_at,"
                "content_fingerprint,transcript_fingerprint")
        .eq("content_fingerprint", content_fingerprint)
        .execute()
    )
    candidates.extend(cf_result.data)

    if transcript_fingerprint:
        tf_result = (
            sb.table("videos")
            .select("id,source_url,title,creator,duration_seconds,wiki_path,processed_at,"
                    "content_fingerprint,transcript_fingerprint")
            .eq("transcript_fingerprint", transcript_fingerprint)
            .execute()
        )
        existing_ids = {r["id"] for r in candidates}
        candidates.extend(r for r in tf_result.data if r["id"] not in existing_ids)

    if not candidates:
        return DuplicateResult.NEW, None, ""

    # Step 2: Score each candidate; surface the highest-scoring match
    title_norm_in = re.sub(r"[^\w\s]", "", (title or "").lower()).strip()
    creator_norm_in = (creator or "").lower().strip()

    best_score = 0
    best_match: Optional[dict] = None
    best_signals: list[str] = []

    for row in candidates:
        score = 0
        signals: list[str] = []

        if row.get("content_fingerprint") == content_fingerprint:
            score += 70
            signals.append("content fingerprint (+70)")

        if transcript_fingerprint and row.get("transcript_fingerprint") == transcript_fingerprint:
            score += 50
            signals.append("transcript fingerprint (+50)")

        row_creator = (row.get("creator") or "").lower().strip()
        if row_creator and row_creator == creator_norm_in:
            score += 30
            signals.append("same creator (+30)")

        row_dur = row.get("duration_seconds") or 0
        dur_diff = abs(row_dur - (duration_seconds or 0))
        if dur_diff <= 3:
            score += 40
            signals.append(f"duration ±{dur_diff:.0f}s (+40)")

        row_title_norm = re.sub(r"[^\w\s]", "", (row.get("title") or "").lower()).strip()
        similarity = difflib.SequenceMatcher(None, title_norm_in, row_title_norm).ratio()
        if similarity >= 0.80:
            score += 20
            signals.append(f"title {int(similarity * 100)}% match (+20)")

        if score > best_score:
            best_score = score
            best_match = row
            best_signals = signals

    explanation = ", ".join(best_signals) + f" = {best_score}"

    if best_score >= 90:
        return DuplicateResult.DEFINITE, best_match, explanation
    elif best_score >= 50:
        return DuplicateResult.LIKELY, best_match, explanation
    else:
        return DuplicateResult.NEW, None, ""


# ── Duplicate resolution ──────────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    """Convert 1231.0 → '20:31'."""
    total = int(seconds or 0)
    return f"{total // 60}:{total % 60:02d}"


def prompt_duplicate_resolution(
    incoming: dict,
    existing: dict,
    result: DuplicateResult,
    match_explanation: str,
) -> DuplicateResolution:
    """Interactive resolution prompt. Only called in non-batch mode."""
    import sys

    duration_str    = _format_duration(existing.get("duration_seconds", 0))
    processed_str   = (existing.get("processed_at") or "unknown")[:10]

    print(f"\n[watch] ⚠️  This video may already be in your knowledge base.\n", file=sys.stderr)
    print(f"  Existing: \"{existing.get('title')}\" — {existing.get('creator')}", file=sys.stderr)
    print(f"            Processed: {processed_str}, Duration: {duration_str}", file=sys.stderr)
    print(f"            Wiki: {existing.get('wiki_path', 'unknown')}", file=sys.stderr)
    print(f"            Match: {match_explanation}", file=sys.stderr)
    print(f"\n  Incoming: \"{incoming.get('title')}\" — {incoming.get('creator')}", file=sys.stderr)
    print(f"            Duration: {_format_duration(incoming.get('duration_seconds', 0))}\n",
          file=sys.stderr)
    print("  1. Skip   — keep existing entry", file=sys.stderr)
    print("  2. Update — replace wiki + DB row with new analysis", file=sys.stderr)
    print("  3. New    — create a separate entry\n", file=sys.stderr)

    while True:
        try:
            choice = input("[watch] → ").strip()
        except EOFError:
            # Non-interactive context (piped stdin) — default to skip
            print("[watch] Non-interactive; defaulting to Skip.", file=sys.stderr)
            return DuplicateResolution.SKIP
        if choice == "1":
            return DuplicateResolution.SKIP
        if choice == "2":
            return DuplicateResolution.UPDATE
        if choice == "3":
            return DuplicateResolution.NEW
        print("[watch] Please enter 1, 2, or 3.", file=sys.stderr)


def auto_resolve(result: DuplicateResult) -> DuplicateResolution:
    """Non-interactive resolution for batch mode or --yes flag.

    Rules:
    - DEFINITE → SKIP silently (clearly the same video; re-processing wastes quota)
    - LIKELY   → SKIP with a warning (user can re-run interactively to confirm)
    We never auto-UPDATE (would discard prior user edits to the wiki page).
    We never auto-NEW in batch (would create duplicates on every re-run).
    """
    return DuplicateResolution.SKIP  # Both DEFINITE and LIKELY auto-skip


# ── Wiki markdown writer ──────────────────────────────────────────────────────

def _format_entities(entities: list[Entity]) -> str:
    if not entities:
        return "_None identified._"
    lines = []
    for e in entities:
        parts = [f"**{e.entity_value}** ({e.entity_type})"]
        if e.mention_count > 1:
            parts.append(f"×{e.mention_count}")
        if e.sentiment:
            parts.append(e.sentiment)
        lines.append("- " + " — ".join(parts))
    return "\n".join(lines)


def _format_crossrefs(entities: list[Entity]) -> str:
    """Generate Obsidian wikilinks for entities that warrant their own hub pages.

    Linkable types span all domains — any entity type that the nodal wiki
    will build a hub page for gets a wikilink here.
    """
    LINKABLE = {
        "ticker",       # finance
        "company",      # general
        "property",     # real estate
        "location",     # real estate
        "model",        # AI
        "technology",   # AI / tech
    }
    linked = [e for e in entities if e.entity_type in LINKABLE]
    if not linked:
        return "_No cross-references._"
    return "\n".join(f"- [[{e.entity_value}]]" for e in linked)


def write_wiki_page(
    ai_output: AIOutput,
    video_meta: dict,
    vault_path: Path,
    video_id: Optional[str] = None,
) -> Path:
    """Write Obsidian markdown page. Returns absolute path."""
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Richer slug: {date}-{creator-slug}-{title-slug}-{duration-code}.md
    # Each component limits collisions along a different axis.
    creator_slug  = _slugify(video_meta.get("creator") or "unknown", max_chars=20)
    title_slug    = _slugify(video_meta.get("title") or video_meta.get("source_url", "video"),
                             max_chars=40)
    duration_secs = int(video_meta.get("duration_seconds") or 0)
    duration_code = str(duration_secs).zfill(4)   # e.g. "1231"; zfill is min width, not max

    filename  = f"{date_str}-{creator_slug}-{title_slug}-{duration_code}.md"
    page_path = vault_path / "videos" / filename
    page_path.parent.mkdir(parents=True, exist_ok=True)

    # Collision fallback: if the target file already exists AND belongs to a different URL,
    # append the last 6 hex chars of this URL's SHA-256 as a disambiguator.
    source_url_for_slug = video_meta.get("source_url", "")
    if page_path.exists() and source_url_for_slug:
        existing_text = page_path.read_text(encoding="utf-8")
        if f"source_url: {source_url_for_slug}" not in existing_text:
            url_suffix = hashlib.sha256(source_url_for_slug.encode()).hexdigest()[-6:]
            filename   = f"{date_str}-{creator_slug}-{title_slug}-{duration_code}-{url_suffix}.md"
            page_path  = vault_path / "videos" / filename

    # Group entities by type for structured frontmatter.
    # Keys are present for all known groups; empty lists are fine in YAML (grep-friendly).
    def _ents(etype: str) -> list[str]:
        return [e.entity_value for e in ai_output.entities if e.entity_type == etype]

    frontmatter_data = {
        "type": "video_analysis",
        "source_url": video_meta.get("source_url", ""),
        "title": video_meta.get("title", ""),
        "creator": video_meta.get("creator", ""),
        "duration_seconds": video_meta.get("duration_seconds", 0),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "mode": video_meta.get("mode", "regular"),
        "vision_provider": video_meta.get("vision_provider", ""),
        "sentiment_overall": ai_output.sentiment_overall,
        "confidence": ai_output.confidence,
        # Finance
        "tickers": _ents("ticker"),
        "indicators": _ents("indicator"),
        # Real estate
        "locations": _ents("location"),
        "properties": _ents("property"),
        "markets": _ents("market"),
        # AI / technology
        "models": _ents("model"),
        "technologies": _ents("technology"),
        # General
        "topics": _ents("topic"),
        "tags": ["video", "auto-generated"],
    }
    if video_id:
        frontmatter_data["video_id"] = video_id

    fm_text = yaml.dump(frontmatter_data, allow_unicode=True, default_flow_style=False)

    creator = video_meta.get("creator", "")
    source_url = video_meta.get("source_url", "")
    title = video_meta.get("title", "Untitled")

    body = f"""# {title}

**Source:** [{creator}]({source_url})
**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Question

> {ai_output.question}

## Analysis

{ai_output.answer}

## Key entities

{_format_entities(ai_output.entities)}

## Cross-references

{_format_crossrefs(ai_output.entities)}
"""

    content = f"---\n{fm_text}---\n\n{body}"
    page_path.write_text(content, encoding="utf-8")
    return page_path


# ── Supabase row writes ───────────────────────────────────────────────────────

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


def write_supabase_rows(
    source_url: str,
    video_meta: dict,
    ai_output: AIOutput,
    frame_results: list[dict],
    transcript_segments: list[dict],
    wiki_path: Path,
    vault_path: Path,
    video_id: Optional[str] = None,
) -> str:
    """Insert all rows. Returns video_id (uuid str). Uses service_role key (Rule 4).

    Pass video_id to use a pre-generated UUID (avoids a second wiki-page rewrite).
    """
    sb = _get_supabase()

    wiki_content = wiki_path.read_text(encoding="utf-8")
    wiki_hash = hashlib.sha256(wiki_content.encode()).hexdigest()

    try:
        wiki_rel = str(wiki_path.relative_to(vault_path))
    except ValueError:
        wiki_rel = str(wiki_path)

    # Infer source_type from URL
    source_type = "other"
    url_lower = source_url.lower()
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        source_type = "youtube"
    elif "vimeo.com" in url_lower:
        source_type = "vimeo"
    elif "tiktok.com" in url_lower:
        source_type = "tiktok"
    elif "twitter.com" in url_lower or "x.com" in url_lower:
        source_type = "twitter"
    elif not url_lower.startswith("http"):
        source_type = "local"

    row_data: dict = {
        "source_url": source_url,
        "source_type": source_type,
        "title": video_meta.get("title"),
        "creator": video_meta.get("creator"),
        "duration_seconds": video_meta.get("duration_seconds"),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "mode": video_meta.get("mode", "regular"),
        "vision_provider": video_meta.get("vision_provider"),
        "model_used": video_meta.get("model_used"),
        "tokens_in": video_meta.get("tokens_in"),
        "tokens_out": video_meta.get("tokens_out"),
        "retention_mode": video_meta.get("retention", "ephemeral"),
        "archive_path": video_meta.get("archive_path"),
        "is_pending": False,
        "wiki_path": wiki_rel,
        "wiki_content": wiki_content,
        "wiki_content_hash": wiki_hash,
        "last_wiki_synced_at": datetime.now(timezone.utc).isoformat(),
        "wiki_edited_by_user": False,
        "created_on_machine": socket.gethostname(),
        "transcript_fingerprint": video_meta.get("transcript_fingerprint"),
        "content_fingerprint":    video_meta.get("content_fingerprint"),
    }
    if video_id is not None:
        row_data["id"] = video_id

    # Upsert: update existing row if source_url already present; insert otherwise.
    # Prevents duplicate `videos` rows on re-runs of the same URL.
    existing = (
        sb.table("videos")
        .select("id")
        .eq("source_url", source_url)
        .limit(1)
        .execute()
    )
    if existing.data:
        existing_id = existing.data[0]["id"]
        update_data = {k: v for k, v in row_data.items() if k != "id"}
        sb.table("videos").update(update_data).eq("id", existing_id).execute()
        video_id = existing_id
        is_new_video = False
    else:
        video_row = sb.table("videos").insert(row_data).execute()
        video_id = video_row.data[0]["id"]
        is_new_video = True

    # Frames (batch insert, cap at 500 rows to stay under PostgREST limits)
    # Skip on re-runs — frames don't change between runs of the same video.
    if frame_results and is_new_video:
        frame_rows = [
            {
                "video_id": video_id,
                "frame_index": i,
                "timestamp_seconds": f.get("timestamp_seconds"),
                "ocr_text": f.get("text"),
                "ocr_confidence_avg": f.get("confidence_avg"),
                "ocr_relevance_score": f.get("relevance_score"),
                "included_in_prompt": bool(f.get("included_in_prompt", False)),
                "frame_path": f.get("path"),
            }
            for i, f in enumerate(frame_results)
        ]
        for batch_start in range(0, len(frame_rows), 500):
            sb.table("video_frames").insert(
                frame_rows[batch_start:batch_start + 500]
            ).execute()

    # Transcript segments — skip on re-runs
    if transcript_segments and is_new_video:
        seg_source_map = {
            "captions": "captions",
            "whisper-groq": "whisper-groq",
            "whisper (groq)": "whisper-groq",
            "whisper (openai)": "whisper-openai",
        }
        seg_rows = []
        for seg in transcript_segments:
            raw_src = str(seg.get("source", "captions")).lower()
            src = seg_source_map.get(raw_src, "captions")
            seg_rows.append({
                "video_id": video_id,
                "start_seconds": seg.get("start") or seg.get("start_seconds", 0),
                "end_seconds": seg.get("end") or seg.get("end_seconds", 0),
                "text": seg.get("text", ""),
                "source": src,
            })
        for batch_start in range(0, len(seg_rows), 500):
            sb.table("video_transcript_segments").insert(
                seg_rows[batch_start:batch_start + 500]
            ).execute()

    # Entities — skip on re-runs
    if ai_output.entities and is_new_video:
        entity_rows = [
            {
                "video_id": video_id,
                "entity_type": e.entity_type,
                "entity_value": e.entity_value,
                "mention_count": e.mention_count,
                "first_mentioned_at_seconds": e.first_mentioned_at_seconds,
                "sentiment": e.sentiment,
                "relevance": e.relevance,
            }
            for e in ai_output.entities
        ]
        sb.table("video_entities").insert(entity_rows).execute()

    # Analysis row
    sb.table("video_analyses").insert({
        "video_id": video_id,
        "question": ai_output.question,
        "answer": ai_output.answer,
        "model": video_meta.get("model_used") or video_meta.get("vision_provider") or "unknown",
        "tokens_in": video_meta.get("tokens_in"),
        "tokens_out": video_meta.get("tokens_out"),
        "confidence": ai_output.confidence,
        "sentiment": ai_output.sentiment_overall,
        "flags": [],
        "audit_log_id": video_meta.get("audit_log_id"),
    }).execute()

    return video_id


# ── pgvector embeddings ───────────────────────────────────────────────────────

def _chunk_text(text: str, target_chars: int = 2000, overlap_chars: int = 200) -> list[str]:
    """Split at paragraph boundaries targeting ~2000 chars (~500 tokens). Returns non-empty chunks."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current_len + len(para) > target_chars and current:
            chunk = "\n\n".join(current)
            chunks.append(chunk)
            # Carry over last paragraph for overlap
            overlap_paras = []
            carry = 0
            for p in reversed(current):
                if carry < overlap_chars:
                    overlap_paras.insert(0, p)
                    carry += len(p)
                else:
                    break
            current = overlap_paras
            current_len = sum(len(p) for p in current)
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in chunks if c.strip()]


def should_reembed(old_content: str, new_content: str, threshold: float = 0.2) -> bool:
    """True if content changed >= threshold (default 20%)."""
    ratio = difflib.SequenceMatcher(None, old_content, new_content).ratio()
    return (1.0 - ratio) >= threshold


def _embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Call Gemini text-embedding-004. Returns list of 768-dim vectors."""
    import google.generativeai as genai
    from dotenv import load_dotenv
    _env = Path.home() / ".config" / "watch" / ".env"
    if _env.exists():
        load_dotenv(_env)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    embeddings = []
    for chunk in chunks:
        resp = genai.embed_content(
            model="models/gemini-embedding-001",
            content=chunk,
            task_type="retrieval_document",
            output_dimensionality=768,
        )
        embeddings.append(resp["embedding"])

    try:
        from quota_tracker import increment_usage
        increment_usage("embed", len(chunks))
    except Exception:
        pass

    return embeddings


def write_embeddings(video_id: str, wiki_path: Path, old_content: Optional[str] = None) -> int:
    """Chunk wiki body, embed, insert into wiki_embeddings. Returns chunk count (0 if skipped).

    Skips re-embed if old_content is supplied and diff < 20%.
    Clears existing embeddings before inserting when re-embedding.
    """
    content = wiki_path.read_text(encoding="utf-8")

    # Strip YAML frontmatter for embedding (embed the body, not the metadata)
    body = content
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            body = content[end + 3:].strip()

    if old_content is not None:
        old_body = old_content
        if old_content.startswith("---"):
            end = old_content.find("---", 3)
            if end != -1:
                old_body = old_content[end + 3:].strip()
        if not should_reembed(old_body, body):
            return 0

    chunks = _chunk_text(body)
    if not chunks:
        return 0

    embeddings = _embed_chunks(chunks)

    sb = _get_supabase()

    # Clear existing embeddings for this video before re-inserting
    if old_content is not None:
        sb.table("wiki_embeddings").delete().eq("video_id", video_id).execute()

    rows = [
        {
            "video_id": video_id,
            "chunk_index": i,
            "chunk_text": chunk,
            "embedding": emb,
            "metadata": {"source": "wiki", "char_count": len(chunk)},
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    sb.table("wiki_embeddings").insert(rows).execute()
    return len(rows)


# ── Main entry point ──────────────────────────────────────────────────────────

def persist_all(
    source_url: str,
    video_meta: dict,
    ai_output: AIOutput,
    frame_results: list[dict],
    transcript_segments: list[dict],
    work_dir: Path,
    is_batch: bool = False,
    yes_flag: bool = False,
) -> dict:
    """Write wiki page and (on writer laptop) Supabase rows + embeddings.

    Returns {"video_id": str | None, "wiki_path": Path}.
    On duplicate skip: returns {"video_id": str, "wiki_path": None, "skipped": True}.

    is_batch: when True, uses non-interactive auto_resolve() instead of prompting
    yes_flag: when True, same effect as is_batch for resolution purposes
    """
    import sys
    from dotenv import load_dotenv
    _env = Path.home() / ".config" / "watch" / ".env"
    if _env.exists():
        load_dotenv(_env)

    vault_path_str = os.environ.get("OBSIDIAN_VAULT_PATH", "")
    if not vault_path_str:
        raise RuntimeError("OBSIDIAN_VAULT_PATH not set in ~/.config/watch/.env")
    vault_path = Path(vault_path_str).expanduser().resolve()

    is_writer = os.environ.get("WATCH_IS_WRITER_LAPTOP", "false").lower() == "true"

    # ── Duplicate detection (writer laptop only — needs Supabase access) ──────
    if is_writer:
        try:
            sb = _get_supabase()

            tf = compute_transcript_fingerprint(transcript_segments)
            cf = compute_content_fingerprint(
                video_meta.get("creator", ""),
                video_meta.get("duration_seconds", 0),
                video_meta.get("title", ""),
            )

            dup_result, existing_row, explanation = find_potential_duplicate(
                sb,
                source_url=source_url,
                creator=video_meta.get("creator", ""),
                duration_seconds=video_meta.get("duration_seconds", 0),
                title=video_meta.get("title", ""),
                content_fingerprint=cf,
                transcript_fingerprint=tf,
            )

            if dup_result != DuplicateResult.NEW:
                if is_batch or yes_flag:
                    resolution = auto_resolve(dup_result)
                    if dup_result == DuplicateResult.LIKELY:
                        print(f"[watch] ⚠️  Likely duplicate (skipped in batch): {source_url}",
                              file=sys.stderr)
                        print(f"         Match: {explanation}", file=sys.stderr)
                    else:
                        print(f"[watch] Definite duplicate — skipping: {source_url}", file=sys.stderr)
                else:
                    resolution = prompt_duplicate_resolution(
                        video_meta, existing_row, dup_result, explanation
                    )

                if resolution == DuplicateResolution.SKIP:
                    return {
                        "video_id": existing_row["id"],
                        "wiki_path": None,
                        "skipped": True,
                        "existing_wiki": existing_row.get("wiki_path"),
                    }

                if resolution == DuplicateResolution.UPDATE:
                    # Reuse the existing video_id so write_supabase_rows() updates the right row
                    video_meta["_existing_video_id"] = existing_row["id"]

                # DuplicateResolution.NEW falls through to the normal write path

            # Attach fingerprints to video_meta so write_supabase_rows() can store them
            video_meta["transcript_fingerprint"] = tf
            video_meta["content_fingerprint"] = cf

        except Exception as _dup_err:
            # Duplicate detection is best-effort — never blocks the write path
            print(f"[watch] duplicate check skipped (non-fatal): {_dup_err}", file=sys.stderr)

    # Pre-generate video_id so wiki page is written once (with ID in frontmatter).
    # Eliminates the double-write that caused a sync-watcher race condition.
    # If UPDATE resolution was chosen, use the existing video_id instead.
    import uuid as _uuid
    if is_writer and video_meta.get("_existing_video_id"):
        pre_video_id = video_meta["_existing_video_id"]
    else:
        pre_video_id = str(_uuid.uuid4()) if is_writer else None

    # Step 1: Write wiki page (always — both laptops, single write)
    wiki_path = write_wiki_page(ai_output, video_meta, vault_path, video_id=pre_video_id)

    # Save watch_meta.json in work_dir for refine.py and other scripts
    meta_file = work_dir / "watch_meta.json"
    meta_file.write_text(
        json.dumps({
            "source_url": source_url,
            "title": video_meta.get("title"),
            "creator": video_meta.get("creator"),
            "mode": video_meta.get("mode"),
            "wiki_path": str(wiki_path),
        }, indent=2),
        encoding="utf-8",
    )

    if not is_writer:
        print("[watch] Read-only laptop — wiki written locally; Supabase writes skipped.", file=sys.stderr)
        return {"video_id": None, "wiki_path": wiki_path}

    # Step 2: Supabase row writes (writer laptop only, uses pre-generated video_id)
    video_id = write_supabase_rows(
        source_url=source_url,
        video_meta=video_meta,
        ai_output=ai_output,
        frame_results=frame_results,
        transcript_segments=transcript_segments,
        wiki_path=wiki_path,
        vault_path=vault_path,
        video_id=pre_video_id,
    )

    # Step 3: pgvector embeddings (best-effort — failure does not block wiki/DB writes)
    n_chunks = 0
    try:
        n_chunks = write_embeddings(video_id, wiki_path)
    except Exception as _emb_err:
        print(f"[watch] embeddings skipped (non-fatal): {_emb_err}", file=sys.stderr)

    print(f"[watch] persisted: video_id={video_id}, {n_chunks} embedding chunks.", file=sys.stderr)

    return {"video_id": video_id, "wiki_path": wiki_path}
