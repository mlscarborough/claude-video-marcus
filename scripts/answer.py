"""Vision dispatcher: Gemini → Claude fallback with quota management.

Call via CLI:
    python scripts/answer.py <work_dir> "<question>" [--mode chart] [--provider auto]

Exit codes:
    0 — success; answer printed to stdout
    9 — FallbackToClaude; JSON with frame/transcript paths printed to stdout
    1 — hard error
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
_env = Path.home() / ".config" / "watch" / ".env"
if _env.exists():
    load_dotenv(_env)

from quota_tracker import (
    get_quota_status, increment_usage, is_exhausted,
    log_quota_to_db, mark_429, should_degrade, should_warn,
)


class FallbackToClaude(Exception):
    pass


class GeminiQuotaError(Exception):
    pass


CHART_MODE_PROMPT = """You are analyzing a video {duration}s long that contains financial chart content.

TRANSCRIPT (timestamps in MM:SS format):
{transcript}

OCR-EXTRACTED TEXT FROM CHART FRAMES (high-confidence text only):
{ocr_text}

IMAGES: I have provided {n_frames} frames at {resolution}px.

USER QUESTION: {question}

CRITICAL INSTRUCTION: When you report any numerical value in your answer, you MUST tag it:
- [TEXT_READ] — you read this number from on-screen text (via OCR or directly visible)
- [VERBAL] — the speaker said this number out loud in the transcript
- [ESTIMATED] — you inferred this from chart geometry (not a direct read)

Example: "RSI at 67.3 [VERBAL]. Price at $622.50 [TEXT_READ]. MACD near 0.42 [ESTIMATED]."

Tags are mandatory for every number. Do not give untagged numerical claims."""

REGULAR_MODE_PROMPT = """You are analyzing a {duration}s video.

TRANSCRIPT (timestamps in MM:SS format):
{transcript}

IMAGES: I have provided {n_frames} frames at {resolution}px.

USER QUESTION: {question}

ENTITY EXTRACTION — include in your JSON response:
When identifying topics covered in this video, use slugs from the TOPIC TAXONOMY below.
Pick the MOST SPECIFIC applicable slug(s). List each slug as a separate entity.

RULES FOR TOPIC SLUGS:
- Use the exact slug path shown (e.g. "real-estate/multi-family/management/evictions")
- Do NOT invent new slugs — use the closest match. Ancestor tagging is handled automatically,
  so just provide the most specific slug (you don't need to list parent levels separately).
- If a concept is not in the taxonomy, use the nearest parent slug AND add a note in a
  separate entity with type "keyword" describing the missing concept. Do NOT use "(new: ...)"
  notation — the system handles proposals automatically.

TOPIC TAXONOMY (excerpt — most relevant domains shown):
{taxonomy_excerpt}

Return JSON:
{{
  "answer": "...",
  "sentiment_overall": "bullish|bearish|neutral|mixed|null",
  "confidence": 0.0-1.0,
  "entities": [
    {{"type": "topic",   "value": "real-estate/multi-family/acquisition/hard-money"}},
    {{"type": "ticker",  "value": "NVDA"}},
    {{"type": "person",  "value": "Brandon Turner"}},
    {{"type": "keyword", "value": "wraparound mortgage"}}
  ]
}}"""

CHART_MODE_PROMPT_SUFFIX = """

TOPIC TAXONOMY (for entity extraction — use most specific slug):
{taxonomy_excerpt}

Return JSON with the same schema shown in the regular prompt (answer, sentiment_overall,
confidence, entities). Tags [TEXT_READ]/[VERBAL]/[ESTIMATED] are mandatory on every number."""


def _detect_domains(question: str, transcript: str) -> list[str]:
    """Heuristic: detect which taxonomy domains are likely relevant to inject a focused excerpt."""
    combined = (question + " " + transcript[:500]).lower()
    domains: list[str] = []
    if any(w in combined for w in [
        "real estate", "property", "rental", "multi-family", "multifamily",
        "apartment", "flip", "landlord", "tenant", "lease", "rehab", "cap rate",
        "noi", "dscr", "1031", "hard money", "seller finance",
    ]):
        domains.append("real-estate")
    if any(w in combined for w in [
        "trading", "options", "stock", "ticker", "chart", "put", "call",
        "scalp", "swing", "rsi", "macd", "candlestick", "earnings", "short",
        "long", "theta", "delta", "iron condor", "spreads",
    ]):
        domains.append("trading")
    if any(w in combined for w in [
        "ai", "llm", "model", "gpt", "claude", "gemini", "llama", "agent",
        "rag", "embedding", "fine-tun", "prompt", "inference", "vector",
        "langchain", "openai", "anthropic",
    ]):
        domains.append("artificial-intelligence")
    # Always include general as a catch-all
    domains.append("general")
    return domains or None  # None = include everything


def _build_prompt(transcript: str, ocr_included: list[dict], question: str,
                  mode: str, duration: float, n_frames: int, resolution: int) -> str:
    ocr_text = "\n".join(
        f"[t={i}] {r['text'][:300]}"
        for i, r in enumerate(ocr_included)
        if r.get("text", "").strip()
    ) or "(no high-relevance OCR text)"

    # Build focused taxonomy excerpt (domain-filtered, capped at 60 lines)
    try:
        from taxonomy import get_taxonomy_excerpt
        domains = _detect_domains(question, transcript)
        taxonomy_excerpt = get_taxonomy_excerpt(domains, max_lines=60)
    except Exception:
        taxonomy_excerpt = "(taxonomy unavailable — use general topic descriptions)"

    if mode == "chart":
        base = CHART_MODE_PROMPT.format(
            duration=int(duration),
            transcript=transcript or "(no transcript available)",
            ocr_text=ocr_text,
            question=question,
            n_frames=n_frames,
            resolution=resolution,
        )
        return base + CHART_MODE_PROMPT_SUFFIX.format(taxonomy_excerpt=taxonomy_excerpt)
    else:
        return REGULAR_MODE_PROMPT.format(
            duration=int(duration),
            transcript=transcript or "(no transcript available)",
            question=question,
            n_frames=n_frames,
            resolution=resolution,
            taxonomy_excerpt=taxonomy_excerpt,
        )


def call_gemini_vision(model_name: str, frame_paths: list[Path], transcript: str,
                       ocr_included: list[dict], question: str, mode: str,
                       duration: float, resolution: int) -> dict:
    """Returns {answer, tokens_in, tokens_out, model}. Raises GeminiQuotaError on 429."""
    import google.generativeai as genai
    from PIL import Image

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in ~/.config/watch/.env")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    content = []
    for fp in frame_paths:
        try:
            content.append(Image.open(fp))
        except Exception:
            pass

    prompt = _build_prompt(transcript, ocr_included, question, mode, duration, len(frame_paths), resolution)
    content.append(prompt)

    try:
        response = model.generate_content(content)
        tokens_in = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        tokens_out = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
        return {
            "answer": response.text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "model": model_name,
        }
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
            raise GeminiQuotaError(err_str) from e
        raise


def dispatch(frame_paths: list[Path], transcript: str, ocr_included: list[dict],
             question: str, mode: str, provider_override: str = "auto",
             duration: float = 0, resolution: int = 512) -> dict:
    """Returns {answer, model, tokens_in, tokens_out, provider}. Raises FallbackToClaude if needed."""

    if provider_override == "claude":
        raise FallbackToClaude("Forced via --provider claude")

    # Decide initial model and quota key
    if mode == "chart":
        target_model = "gemini-2.5-pro-preview-06-05"
        quota_key = "pro"
    else:
        target_model = "gemini-2.0-flash"
        quota_key = "flash"

    was_degraded = False

    if is_exhausted(quota_key):
        if provider_override == "gemini":
            raise RuntimeError(f"Gemini {quota_key} quota exhausted and --provider gemini was forced")
        raise FallbackToClaude(f"Gemini {quota_key} quota exhausted")

    if should_degrade(quota_key) and quota_key == "pro":
        target_model = "gemini-2.0-flash"
        quota_key = "flash"
        was_degraded = True
        print("[watch] Quota at 90%; auto-downgrading chart mode to Gemini Flash", file=sys.stderr)

    if should_warn(quota_key):
        status = get_quota_status()
        used = status.get(f"{quota_key}_used", 0)
        from quota_tracker import LIMITS
        limit = LIMITS.get(quota_key, 1)
        print(f"[watch] WARNING: Gemini {quota_key} quota at {used}/{limit} ({used/limit:.0%})", file=sys.stderr)

    try:
        result = call_gemini_vision(
            target_model, frame_paths, transcript, ocr_included,
            question, mode, duration, resolution
        )
        increment_usage(quota_key)
        log_quota_to_db(target_model, result["tokens_in"], result["tokens_out"],
                        was_429=False, was_degraded=was_degraded)
        result["provider"] = "gemini"
        return result
    except GeminiQuotaError as e:
        mark_429(quota_key)
        log_quota_to_db(target_model, 0, 0, was_429=True, was_degraded=was_degraded)
        if provider_override == "gemini":
            raise
        raise FallbackToClaude(f"Hit 429: {e}") from e


def _find_work_artifacts(work_dir: Path) -> dict:
    """Collect paths to frames, transcript, OCR sidecars in a work dir."""
    frame_dir = work_dir / "frames"
    frames = sorted(frame_dir.glob("*.jpg")) if frame_dir.exists() else []
    transcript_path = work_dir / "audio.mp3"  # transcript stored as .vtt in download/
    vtt_files = list((work_dir / "download").glob("*.vtt")) if (work_dir / "download").exists() else []
    return {
        "frames": [str(f) for f in frames],
        "transcript_path": str(vtt_files[0]) if vtt_files else None,
        "ocr_dir": str(frame_dir),
    }


def _do_persist(work_dir: Path, answer_text: str, question: str, mode: str,
                provider: str, model: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
    """Call persist_all() from watch_run.json + answer text. Best-effort — never raises."""
    try:
        run_meta_file = work_dir / "watch_run.json"
        if not run_meta_file.exists():
            print("[answer] watch_run.json not found — skipping persist", file=sys.stderr)
            return
        run_meta = json.loads(run_meta_file.read_text(encoding="utf-8"))
        from persist import persist_all, parse_ai_output
        ai_out = parse_ai_output(answer_text, question)
        video_meta = {
            "source_url": run_meta.get("source_url", ""),
            "title": run_meta.get("title"),
            "creator": run_meta.get("creator"),
            "duration_seconds": run_meta.get("duration_seconds"),
            "mode": run_meta.get("mode", mode),
            "vision_provider": provider,
            "model_used": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "retention": run_meta.get("retention", "ephemeral"),
        }
        persist_all(
            source_url=run_meta.get("source_url", ""),
            video_meta=video_meta,
            ai_output=ai_out,
            frame_results=run_meta.get("frame_results", []),
            transcript_segments=run_meta.get("transcript_segments", []),
            work_dir=work_dir,
        )
    except Exception as e:
        print(f"[answer] persist failed (non-fatal): {e}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(prog="answer")
    ap.add_argument("work_dir", help="Working directory from watch.py")
    ap.add_argument("question", help="The user's question about the video")
    ap.add_argument("--mode", choices=["regular", "chart"], default="regular")
    ap.add_argument("--provider", choices=["gemini", "claude", "auto"], default="auto")
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument(
        "--claude-answer",
        metavar="FILE",
        default=None,
        help="Path to a file containing Claude's answer text. Skips Gemini dispatch and persists directly.",
    )
    ap.add_argument("--model", default="claude-sonnet-4-6", help="Model name for --claude-answer path")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)

    # ── Claude-answer fast path: persist without Gemini dispatch ─────────────
    if args.claude_answer:
        answer_file = Path(args.claude_answer)
        if not answer_file.exists():
            print(f"[answer] --claude-answer file not found: {answer_file}", file=sys.stderr)
            return 1
        answer_text = answer_file.read_text(encoding="utf-8")

        # Audit log for Claude answers (satisfies Rule 3 for this path too)
        try:
            from audit_client import log_audit
            log_audit(model=args.model, input_data=args.question, output_data=answer_text)
        except Exception:
            pass

        _do_persist(work_dir, answer_text, args.question, args.mode,
                    provider="claude", model=args.model)
        return 0

    # ── Normal Gemini dispatch path ───────────────────────────────────────────
    frame_dir = work_dir / "frames"
    frame_paths = sorted(frame_dir.glob("*.jpg")) if frame_dir.exists() else []

    if not frame_paths:
        print(f"[answer] No frames found in {frame_dir}", file=sys.stderr)
        return 1

    # Load transcript text
    transcript = ""
    vtt_files = list((work_dir / "download").glob("*.vtt")) if (work_dir / "download").exists() else []
    if vtt_files:
        try:
            from transcribe import parse_vtt, format_transcript
            transcript = format_transcript(parse_vtt(vtt_files[0]))
        except Exception:
            pass

    # Load OCR results if available
    ocr_included = []
    for fp in frame_paths:
        sidecar = fp.with_suffix(".txt")
        if sidecar.exists():
            text = sidecar.read_text(encoding="utf-8").strip()
            ocr_included.append({"path": str(fp), "text": text, "included_in_prompt": True})

    try:
        result = dispatch(
            frame_paths, transcript, ocr_included, args.question,
            args.mode, args.provider, args.duration, args.resolution
        )

        # Audit log
        try:
            from audit_client import log_audit
            log_audit(
                model=result["model"],
                input_data=args.question,
                output_data=result["answer"],
                tokens_in=result["tokens_in"],
                tokens_out=result["tokens_out"],
            )
        except Exception:
            pass

        print(result["answer"])

        _do_persist(work_dir, result["answer"], args.question, args.mode,
                    provider=result.get("provider", "gemini"),
                    model=result.get("model", ""),
                    tokens_in=result.get("tokens_in", 0),
                    tokens_out=result.get("tokens_out", 0))

        return 0

    except FallbackToClaude as e:
        print(f"FALLBACK_TO_CLAUDE: {e}", file=sys.stderr)
        artifacts = _find_work_artifacts(work_dir)
        print(json.dumps(artifacts))
        return 9

    except Exception as e:
        print(f"[answer] Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
