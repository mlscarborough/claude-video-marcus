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
from pathlib import Path
from typing import Optional

import yaml

SCRIPT_DIR = Path(__file__).parent.resolve()

IS_WRITER = os.environ.get("WATCH_IS_WRITER_LAPTOP", "false").lower() == "true"


# ── Domain model ──────────────────────────────────────────────────────────────

VALID_ENTITY_TYPES = {
    "ticker", "person", "company", "topic", "keyword", "indicator", "price_level"
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

    entities = []
    for e in data.get("entities", []):
        etype = e.get("type", e.get("entity_type", "keyword")).lower()
        if etype not in VALID_ENTITY_TYPES:
            etype = "keyword"
        entities.append(Entity(
            entity_type=etype,
            entity_value=str(e.get("value", e.get("entity_value", ""))),
            mention_count=int(e.get("mention_count", 1)),
            first_mentioned_at_seconds=e.get("first_mentioned_at_seconds"),
            sentiment=e.get("sentiment"),
            relevance=float(e.get("relevance", 0.5)) if e.get("relevance") is not None else None,
        ))

    return AIOutput(
        question=question,
        answer=data.get("answer", raw),
        sentiment_overall=data.get("sentiment_overall"),
        confidence=float(data.get("confidence", 0.5)),
        entities=entities,
    )


# ── Wiki markdown writer ──────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    text = (text or "video").lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60].strip("-")


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


def _format_xrefs(entities: list[Entity]) -> str:
    tickers = [e.entity_value for e in entities if e.entity_type == "ticker"]
    if not tickers:
        return "_No ticker cross-references._"
    return "\n".join(f"- [[{t}]]" for t in tickers)


def write_wiki_page(
    ai_output: AIOutput,
    video_meta: dict,
    vault_path: Path,
    video_id: Optional[str] = None,
) -> Path:
    """Write Obsidian markdown page. Returns absolute path."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = _slugify(video_meta.get("title") or video_meta.get("source_url", "video"))
    page_path = vault_path / "videos" / f"{date_str}-{slug}.md"
    page_path.parent.mkdir(parents=True, exist_ok=True)

    tickers = [e.entity_value for e in ai_output.entities if e.entity_type == "ticker"]
    indicators = [e.entity_value for e in ai_output.entities if e.entity_type == "indicator"]

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
        "tickers": tickers,
        "indicators": indicators,
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

{_format_xrefs(ai_output.entities)}
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
) -> str:
    """Insert all rows. Returns video_id (uuid str). Uses service_role key (Rule 4)."""
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

    video_row = sb.table("videos").insert({
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
    }).execute()

    video_id = video_row.data[0]["id"]

    # Frames (batch insert, cap at 500 rows to stay under PostgREST limits)
    if frame_results:
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

    # Transcript segments
    if transcript_segments:
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

    # Entities
    if ai_output.entities:
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
            model="models/text-embedding-004",
            content=chunk,
            task_type="retrieval_document",
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
) -> dict:
    """Write wiki page and (on writer laptop) Supabase rows + embeddings.

    Returns {"video_id": str | None, "wiki_path": Path}.
    """
    from dotenv import load_dotenv
    _env = Path.home() / ".config" / "watch" / ".env"
    if _env.exists():
        load_dotenv(_env)

    vault_path_str = os.environ.get("OBSIDIAN_VAULT_PATH", "")
    if not vault_path_str:
        raise RuntimeError("OBSIDIAN_VAULT_PATH not set in ~/.config/watch/.env")
    vault_path = Path(vault_path_str).expanduser().resolve()

    is_writer = os.environ.get("WATCH_IS_WRITER_LAPTOP", "false").lower() == "true"

    # Step 1: Write wiki page (always — both laptops)
    wiki_path = write_wiki_page(ai_output, video_meta, vault_path)

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
        import sys
        print("[watch] Read-only laptop — wiki written locally; Supabase writes skipped.", file=sys.stderr)
        return {"video_id": None, "wiki_path": wiki_path}

    # Step 2: Supabase row writes (writer laptop only)
    video_id = write_supabase_rows(
        source_url=source_url,
        video_meta=video_meta,
        ai_output=ai_output,
        frame_results=frame_results,
        transcript_segments=transcript_segments,
        wiki_path=wiki_path,
        vault_path=vault_path,
    )

    # Update wiki page with video_id now that we have it
    write_wiki_page(ai_output, video_meta, vault_path, video_id=video_id)

    # Step 3: pgvector embeddings
    n_chunks = write_embeddings(video_id, wiki_path)

    import sys
    print(f"[watch] persisted: video_id={video_id}, {n_chunks} embedding chunks.", file=sys.stderr)

    return {"video_id": video_id, "wiki_path": wiki_path}
