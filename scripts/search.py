"""search.py — semantic search over the /watch knowledge base.

Embeds the query with Gemini embedding-001, runs cosine similarity search
via the search_wiki_chunks Postgres function, and returns the top matching
passages with video titles, creators, wiki paths, and similarity scores.

CLI:
    python scripts/search.py "cash-secured put entry rules"
    python scripts/search.py "cap rate compression" --top-k 10 --min-score 0.5
    python scripts/search.py "attention mechanism" --json

Flags:
    --top-k N        Max results to show (default 5)
    --min-score F    Min similarity 0.0–1.0 to include (default 0.40)
    --json           Emit raw JSON instead of formatted table
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
_env = Path.home() / ".config" / "watch" / ".env"
if _env.exists():
    load_dotenv(_env)


# ── Supabase client ───────────────────────────────────────────────────────────

def _get_supabase():
    from supabase import create_client
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY not set in ~/.config/watch/.env")
    return create_client(url, key)


# ── 21.1  Embed query ─────────────────────────────────────────────────────────

def embed_query(query_text: str) -> list[float]:
    """Embed a search query using Gemini embedding API. Returns 768-dim vector.

    Uses task_type=retrieval_query (asymmetric retrieval) — distinct from
    retrieval_document used when indexing wiki chunks.
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in ~/.config/watch/.env")

    genai.configure(api_key=api_key)
    resp = genai.embed_content(
        model="models/gemini-embedding-001",
        content=query_text,
        task_type="retrieval_query",
        output_dimensionality=768,
    )
    return resp["embedding"]


# ── 21.2  Search via pgvector ─────────────────────────────────────────────────

def search_wiki(
    query_vector: list[float],
    top_k: int = 5,
    min_score: float = 0.40,
) -> list[dict]:
    """Run cosine similarity search. Returns top-k results above min_score.

    Calls search_wiki_chunks() Postgres function (Task 21 migration).
    Each result includes: video_id, title, creator, wiki_path,
    chunk_index, chunk_text, similarity.
    """
    sb = _get_supabase()

    # Over-fetch so we have room to apply the min_score filter
    fetch_k = max(top_k * 3, 20)

    rows = (
        sb.rpc(
            "search_wiki_chunks",
            {
                "query_embedding": query_vector,
                "match_count": fetch_k,
            },
        )
        .execute()
        .data
    ) or []

    results = [r for r in rows if (r.get("similarity") or 0) >= min_score]
    return results[:top_k]


# ── 21.3  Combined pipeline ───────────────────────────────────────────────────

def search_knowledge_base(
    query_text: str,
    top_k: int = 5,
    min_score: float = 0.40,
) -> list[dict]:
    """Full pipeline: embed query → pgvector search → return enriched results."""
    query_vector = embed_query(query_text)
    return search_wiki(query_vector, top_k=top_k, min_score=min_score)


# ── 21.3  Result formatting ───────────────────────────────────────────────────

def format_results(query: str, results: list[dict]) -> str:
    """Format search results as a human-readable report."""
    SEP = "=" * 60

    lines = [
        f"\n{SEP}",
        f"  SEARCH RESULTS",
        f'  Query: "{query}"',
        SEP,
        "",
    ]

    if not results:
        lines.append("  No results found.")
        lines.append("")
        lines.append("  Tips:")
        lines.append("    - Try broader terms (e.g. 'options' instead of 'cash-secured put entry')")
        lines.append("    - Lower --min-score (default 0.40) to see weaker matches")
        lines.append("    - Process more videos first with /watch")
        return "\n".join(lines)

    for i, r in enumerate(results, 1):
        score_pct = int((r.get("similarity") or 0) * 100)
        title   = (r.get("title") or "(untitled)").strip()
        creator = (r.get("creator") or "(unknown)").strip()
        wiki    = r.get("wiki_path") or ""
        excerpt = (r.get("chunk_text") or "").replace("\n", " ").strip()

        # Truncate excerpt to 220 chars for readability
        if len(excerpt) > 220:
            excerpt = excerpt[:217] + "..."

        lines.append(f"  {i}. [{score_pct}%]  {title}")
        lines.append(f"       Creator : {creator}")
        if wiki:
            lines.append(f"       Wiki    : {wiki}")
        if excerpt:
            lines.append(f"       Excerpt : {excerpt}")
        lines.append("")

    lines.append(f"  {len(results)} result(s) | use --top-k N for more | --min-score F to adjust threshold")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="search",
        description="Semantic search over the /watch knowledge base",
    )
    ap.add_argument("query", nargs="+", help="Search query — bare words or quoted string")
    ap.add_argument("--top-k",     type=int,   default=5,    metavar="N",
                    help="Max results to return (default 5)")
    ap.add_argument("--min-score", type=float, default=0.40, metavar="F",
                    help="Minimum similarity threshold 0.0–1.0 (default 0.40)")
    ap.add_argument("--json",      action="store_true",
                    help="Emit raw JSON instead of formatted output")
    args = ap.parse_args()

    query = " ".join(args.query)

    try:
        results = search_knowledge_base(
            query,
            top_k=args.top_k,
            min_score=args.min_score,
        )
    except Exception as exc:
        print(f"[search] ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        # Omit raw embedding vector from JSON output
        clean = [{k: v for k, v in r.items() if k != "embedding"} for r in results]
        print(json.dumps(clean, indent=2))
    else:
        print(format_results(query, results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
