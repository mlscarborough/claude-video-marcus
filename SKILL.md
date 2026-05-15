---
name: watch
description: Watch a video (URL or local path). Downloads with yt-dlp, extracts auto-scaled frames with ffmpeg, pulls the transcript from captions (or Whisper API fallback), and hands the result to Claude so it can answer questions about what's in the video.
argument-hint: "<video-url-or-path> [question]"
allowed-tools: Bash, Read, AskUserQuestion
homepage: https://github.com/bradautomates/claude-video
repository: https://github.com/mlscarborough/claude-video-marcus
author: bradautomates (fork by mlscarborough)
license: MIT
user-invocable: true
---

# /watch — Claude watches a video (personal-v1)

You don't have a video input; this skill gives you one. A Python script downloads the video, extracts frames as JPEGs, gets a timestamped transcript (native captions first, then Whisper API as fallback), and prints frame paths. You then `Read` each frame path to see the images and combine them with the transcript to answer the user.

**Windows note:** use `python` (not `python3`) throughout this skill. On Windows `python3` is the Microsoft Store stub and will not run the scripts. The venv Python is at `"$env:USERPROFILE\.claude\skills\watch\.venv\Scripts\python.exe"` — use it explicitly.

**Venv Python variable** (set at the start of every /watch session):
```bash
WATCH_PYTHON="$USERPROFILE\.claude\skills\watch\.venv\Scripts\python.exe"
WATCH_DIR="$USERPROFILE\.claude\skills\watch"
```

## Step 0 — Setup preflight (runs every `/watch` invocation, silent on success)

Before every `/watch` run, verify that dependencies and an API key are in place:

```bash
"$WATCH_PYTHON" "$WATCH_DIR/scripts/setup.py" --check
```

This is a <100ms lookup. On exit 0, emit **nothing** — proceed to Step 1 without comment. Do NOT announce "setup is complete" on every turn.

On non-zero exit:

| Exit | Meaning | Action |
|------|---------|--------|
| `2` | Missing binaries (`ffmpeg` / `ffprobe` / `yt-dlp`) | Run installer |
| `3` | No Whisper API key | Run installer to scaffold `.env`, then ask user for a key |
| `4` | Both missing | Run installer, then ask for a key |

## Mode selection logic

Choose the mode based on user intent:

- **`--mode regular`** (default): talking-head content, general questions, summarization
- **`--mode chart`** when:
  - User mentions charts, candlesticks, trading, technical indicators (RSI, MACD, etc.)
  - User asks for pinpoint accuracy ("exact value at X", "what is the RSI reading")
  - User identifies content as financial / technical analysis
  - Video title/URL suggests financial/trading content
- If ambiguous (e.g., "educational finance video"), use `AskUserQuestion` to confirm before running

Chart mode automatically uses: 1920px resolution, up to 150 frames, Gemini 2.5 Pro (best vision quality).

## Step 1 — Parse user input

Separate:
- Video source (URL or file path)
- Optional user question
- Infer mode from question/context (see Mode selection logic above)
- **Detect batch intent** (see Batch processing below)
- **Detect search intent** (see Semantic search below)

### Batch intent detection (Task 16.6)

Recognise these patterns as batch requests and extract the relevant flags before calling the script:

| User says | What to extract | Flags to pass |
|-----------|----------------|---------------|
| "last 10 videos from https://youtube.com/@PatrickKenney" | channel URL, N=10, order=newest | `--max-videos 10 --order newest` |
| "first 5 videos from …" | channel URL, N=5, order=oldest | `--max-videos 5 --order oldest` |
| "all videos from this playlist …" | playlist URL | no cap flag (playlists are processed in full) |
| "watch this whole channel" | channel URL, no N | enumerate total, show cap prompt (channel detection handles this automatically) |
| "watch these 4 URLs: url1 url2 url3 url4" | multiple URLs | pass all as positional args |
| "watch this folder" / local directory path | directory path | pass directory as source (auto-detected) |

**Rules:**
- If user says "last N" or "most recent N" → `--order newest`
- If user says "first N" or "oldest N" → `--order oldest`
- If user says "whole channel" or "all videos" with no quantity → omit `--max-videos` (let channel cap prompt handle it)
- **Never start downloading without confirming.** The script will always show what it found and ask before proceeding. You do NOT need to do your own confirmation step — just run the script and show the user its output.
- If user names a topic filter ("the trading videos from this channel") → note that topic filtering is not yet supported; tell the user you'll process the N most recent and they can filter later

### Search intent detection

Recognise these as search requests (no video download needed):
- "search for …", "find videos about …", "what have I watched about …"
- "what does my knowledge base say about …"
- Any question about previously processed content

## Step 2 — Run the watch script

```bash
"$WATCH_PYTHON" "$WATCH_DIR/scripts/watch.py" "<source>" [--mode regular|chart] [--keep none|transcript|all] [--provider auto|gemini|claude]
```

Additional flags:
- `--start T` / `--end T` — focus on a section (SS, MM:SS, or HH:MM:SS)
- `--max-frames N` — override frame count (mode default applied if omitted)
- `--resolution W` — override frame width (mode default applied if omitted)
- `--fps F` — override auto-fps
- `--out-dir DIR` — keep working files at a specific path
- `--whisper groq|openai` — force Whisper backend
- `--no-whisper` — disable Whisper fallback

**Batch flags** (automatically dispatches to `batch.py`):
- `--batch FILE` — text file, one URL per line (`#` comments ignored)
- `"<url1>" "<url2>"` — multiple positional URLs
- `--max-videos N` — cap channel enumeration (default 20; 0 = unlimited)
- `--order newest|oldest` — channel video ordering (default: newest)
- `--yes` — skip interactive confirmation prompt (non-interactive / scripted use)

### Retention flags

| Flag | Behavior |
|------|---------|
| `--keep none` (default) | Delete local raw files after 48h; DB + wiki + vectors persist indefinitely |
| `--keep transcript` | Also keep transcript file in archive |
| `--keep all` | Keep full archive (video, frames, OCR, transcript) on BackupSSD |

## Step 3 — Read every frame path

The Read tool renders JPEGs directly as images. Read all frames in a single message (parallel tool calls) so you see them together. Frames are in chronological order with `t=MM:SS` timestamps.

## Step 4 — Answer the user

Combine frames + transcript to answer. Cite timestamps.

### Numerical claim tagging (MANDATORY in chart mode)

In chart mode (`--mode chart`), every numerical value you report MUST be tagged with its provenance:

- `[TEXT_READ]` — you read this number from on-screen text (OCR or visible text in the frame)
- `[VERBAL]` — the speaker said this number out loud in the transcript
- `[ESTIMATED]` — you inferred this from chart geometry (line position relative to axes); not a direct read

Example:
> RSI at 67.3 [VERBAL] as bullish momentum. Price at $622.50 [TEXT_READ]. MACD crossover near 0.42 [ESTIMATED] but exact value not displayed.

**Tags are mandatory for every number in chart mode. Do not give untagged numerical claims.**

In regular mode, tagging is optional but use `[ESTIMATED]` when you are inferring rather than reading.

## Step 5 — Frame refinement (on-demand zoom)

When the user asks for more detail at a specific timestamp, invoke the refine script:

```bash
"$WATCH_PYTHON" "$WATCH_DIR/scripts/refine.py" "<work_dir>" --timestamp 2:13 [--window 5] [--resolution max] [--fps 4]
```

**Auto-invoke refine (no need to ask) when:**
- User asks for "exact value at X" or "pinpoint" detail
- User says "look more closely at X" or "zoom into X"
- You answered [ESTIMATED] on a value the user wants verified

**Ask via AskUserQuestion first when:**
- Refine requires re-downloading the video (no local copy) AND it's a long video
- The question is ambiguous and refinement may not help

**Skip refine when:**
- The question is general or qualitative (summary, sentiment, "what happens in this video")
- You already have enough detail from initial frames

Refined frames are returned as paths you can Read. They are session-scoped and not saved to DB.

## Step 6 — Persistence (mandatory after every answer)

After you finish answering the user, **always** persist — regardless of which provider was used.

### If Gemini answered (answer.py exited 0)
Persistence ran automatically inside answer.py. Nothing extra needed.

### If Claude answered directly (answer.py exited 9, or --provider claude)
You must trigger persistence manually. Do this every time, no exceptions:

1. Write your complete answer text to `<work_dir>/claude_answer.md` using the Write tool.
2. Run:
```bash
"$WATCH_PYTHON" "$WATCH_DIR/scripts/answer.py" "<work_dir>" "<question>" --claude-answer "<work_dir>/claude_answer.md" --model claude-sonnet-4-6
```

This writes the Obsidian wiki page, inserts Supabase rows (videos, frames, transcript, entities, analyses), and generates pgvector embeddings.

**The work_dir is printed in the watch.py stderr output as `[watch] working dir: <path>`.**

Persistence is best-effort — if it fails, the error is printed to stderr and does not affect your answer. But do not skip the step.

## Gemini / Claude provider routing

The `answer.py` script handles provider selection automatically (when `--provider auto`):

1. Checks Gemini quota (`~/.config/watch/gemini-usage.json`)
2. If quota available → uses Gemini (Pro for chart, Flash for regular)
3. At 70% quota → warns user in console
4. At 90% quota → auto-downgrades chart-mode from Pro to Flash
5. At 100% or on 429 → exits with code 9 (fallback signal)

**When you see exit code 9 from answer.py:**
The script prints JSON with frame paths and transcript path. Use `Read` to read the frames directly and answer using your vision. This uses your existing Anthropic session — no additional API key needed.

## Semantic search (`/watch --search`)

Query the accumulated knowledge base without processing a new video:

```bash
"$WATCH_PYTHON" "$WATCH_DIR/scripts/watch.py" --search "cash-secured put entry rules"
"$WATCH_PYTHON" "$WATCH_DIR/scripts/watch.py" --search "cap rate compression" --top-k 10 --min-score 0.5
```

Or call `search.py` directly:
```bash
"$WATCH_PYTHON" "$WATCH_DIR/scripts/search.py" "attention mechanism"
```

Flags:
- `--top-k N` — max results (default 5)
- `--min-score F` — minimum similarity threshold 0.0–1.0 (default 0.40)
- `--json` — raw JSON output

**When to use:** When the user asks a question about previously processed content rather than about a new video. The search embeds the query with Gemini (same model used for wiki chunks), runs pgvector cosine similarity, and returns the top matching passages with video titles, creators, wiki paths, and similarity scores.

**When NOT to use:** When the user provides a new video URL or file — use the standard watch+answer pipeline instead.

## Taxonomy management (`build_index.py`)

Manage the controlled topic vocabulary and rebuild hub pages:

```bash
# Review new taxonomy proposals
"$WATCH_PYTHON" "$WATCH_DIR/scripts/build_index.py" --review-proposals

# Approve / reject a proposed node
"$WATCH_PYTHON" "$WATCH_DIR/scripts/build_index.py" --approve real-estate/multi-family/acquisition
"$WATCH_PYTHON" "$WATCH_DIR/scripts/build_index.py" --reject  some-bad-slug

# Retroactively tag past videos
"$WATCH_PYTHON" "$WATCH_DIR/scripts/build_index.py" --retag real-estate/multi-family/cap-rates --keywords "cap rate,capitalization rate"

# Rebuild all hub pages (creators, topics, entities) from current DB state
"$WATCH_PYTHON" "$WATCH_DIR/scripts/build_index.py" --rebuild-hubs
"$WATCH_PYTHON" "$WATCH_DIR/scripts/build_index.py" --rebuild-hubs --domain real-estate

# Show recently added taxonomy nodes
"$WATCH_PYTHON" "$WATCH_DIR/scripts/build_index.py" --list-recent-taxonomy --days 14
```

## System commands

- `/watch --health` — show system health dashboard (quota, sync status, heartbeat)
- `/watch --doctor` — run 14 sanity checks and explain any failures in plain English
- `/watch --search "query"` — semantic search over the accumulated knowledge base

## Recommended limits

- Best accuracy: videos under 10 minutes
- Regular mode defaults: 80 frames at 512px
- Chart mode defaults: 150 frames at 1920px
- For long videos, use `--start`/`--end` to focus on a specific section

## Transcription

1. **Native captions (free, preferred)**: yt-dlp pulls platform subtitles
2. **Whisper fallback**: Groq (preferred, cheaper) → OpenAI (fallback)

Keys live in `~/.config/watch/.env`.

## Failure modes

- **Setup preflight failed** → run installer script
- **No transcript** → proceed frames-only; tell the user
- **Long video warning** → offer to re-run focused on a section
- **Download fails** → tell user plainly (login-required, region-locked, etc.)
- **answer.py exits 9** → fallback to Claude vision (Read the frames yourself)
- **Watcher dead** → tell user to run `/watch --doctor` for diagnosis and fix

## Security & Permissions

**What this skill does:**
- Runs `yt-dlp`, `ffmpeg`/`ffprobe` locally
- Sends extracted audio to Groq/OpenAI Whisper APIs (not the video)
- Sends frames + transcript to Gemini API for vision analysis (when `--provider auto` or `gemini`)
- Writes wiki markdown files to Obsidian vault (writer laptop only)
- Writes structured rows and embeddings to Supabase (writer laptop only, uses service_role key)
- Calls the `audit-stage2` Edge Function for every Gemini/Claude vision call (Rule 3 compliance)
- Reads/creates `~/.config/watch/.env` (mode 0600)

**What this skill does NOT do:**
- Does not upload the video itself to any API
- Does not access platform accounts
- Does not write to Supabase on the read-only laptop (`WATCH_IS_WRITER_LAPTOP=false`)
- Does not skip the audit log for vision calls
