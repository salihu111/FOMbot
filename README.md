# FOM Assistant — Telegram Bot (embedding-based retrieval)

Answers First Officer questions strictly from the Ethiopian Airlines Flight
Operations Manual (FOM Rev.25B, 814 pages) — with page-cited answers and
tap-to-view source page images.

**What changed from the keyword version:** retrieval now uses real semantic
embeddings (`BAAI/bge-small-en-v1.5` via the `fastembed` library) instead of
TF-IDF keyword matching. This means a question phrased very differently from
the manual's wording (e.g. "what if the transponder dies near RVSM airspace"
vs. the manual's "RVSM contingency procedures") still finds the right page,
because it matches on meaning, not exact words.

**Repo size:** ~13 MB (PDF ~11 MB, extracted text ~1.6 MB) — fits GitHub, no
Git LFS needed. The embedding model itself (~130 MB) is *not* stored in the
repo — `fastembed` downloads and caches it automatically the first time the
bot runs.

## How it works
1. `indexer.py` (shared module) extracts text per page from the PDF, strips
   repeated header/footer boilerplate, detects section titles, and embeds
   each page into a 384-dimensional vector.
2. `bot.py` embeds each incoming question the same way and finds the most
   similar pages by cosine similarity. If nothing scores above a confidence
   threshold, it widens the search to neighboring pages instead of just
   replying "not found."
3. The retrieved excerpts are sent to `openai/gpt-oss-120b` (via Groq) with
   a system prompt that forces answers to stay inside the manual and cite
   page numbers like `[p.51]`.
4. Each cited page becomes a tappable inline button. Tapping it renders that
   exact PDF page as an image (via PyMuPDF) and sends it in chat.

The page text (`data/chunks.json`) is already extracted and committed. The
embeddings (`data/embeddings.npy`) are **not** pre-built in this package —
see step 4 for the two ways to generate them.

## 1. Get your API keys
- **Telegram bot token**: message [@BotFather](https://t.me/BotFather) →
  `/newbot` → follow prompts → copy the token.
- **Groq API key**: sign up at [console.groq.com](https://console.groq.com) →
  API Keys → Create Key. (Groq hosts `openai/gpt-oss-120b` for free-tier use.)

## 2. Put the project on GitHub
```bash
cd fom-bot
git init
git add .
git commit -m "FOM assistant bot (embedding retrieval)"
git branch -M main
git remote add origin https://github.com/<your-username>/fom-bot.git
git push -u origin main
```
Since the repo is well under 25MB, this pushes with a plain `git push` — no
Git LFS setup required.

**Do not commit your API keys.** They're read from environment variables
(`TELEGRAM_BOT_TOKEN`, `GROQ_API_KEY`), never hardcoded.

## 3. Deploy on Railway
1. Go to [railway.app](https://railway.app) → New Project → **Deploy from
   GitHub repo** → pick `fom-bot`.
2. Railway auto-detects `requirements.txt` and `Procfile` (worker process,
   since this is a polling bot with no web server).
3. In the project's **Variables** tab, add:
   - `TELEGRAM_BOT_TOKEN` = your BotFather token
   - `GROQ_API_KEY` = your Groq key
4. Deploy.

## 4. Building the embedding index
You have two options — pick one:

**Option A — let the bot build it on first boot (simplest).**
Just deploy as-is. On first startup, `bot.py` notices `data/embeddings.npy`
is missing and automatically extracts + embeds all 814 pages (a few minutes,
plus a one-time ~130MB model download). Watch the Railway logs for
`"No prebuilt index found — building it now..."` then `"Bot starting..."`.
Note: if Railway's filesystem for your plan doesn't persist between
restarts, it will rebuild every time the service restarts — still works,
just adds a few minutes to each restart.

**Option B — pre-build it locally, commit it (faster cold starts).**
```bash
pip install -r requirements.txt
python ingest.py
```
This creates `data/embeddings.npy` (~1.2MB) locally. Commit and push it:
```bash
git add data/embeddings.npy
git commit -m "Add prebuilt embedding index"
git push
```
Now Railway deploys start instantly with no build step.

Either way, once you see `"Bot starting..."` in the logs, open Telegram,
message your bot, send `/start`, then ask a question.

## 5. (Optional) Rebuild after a manual revision
Swap in the new PDF at `data/FOM_REV_25B.pdf` (keep the same filename, or
update `PDF_PATH` in `indexer.py` and `bot.py`), delete the old
`data/chunks.json` and `data/embeddings.npy`, then run `python ingest.py`
again and push.

## Tuning
- `CONFIDENCE_THRESHOLD` in `bot.py` (default `0.45`) — embedding cosine
  scores behave differently from TF-IDF scores; if you notice the bot
  triggering "closest relevant section" fallback too often (or not often
  enough) once it's live, adjust this up or down and redeploy.
- `TOP_K` — how many top-matching pages get sent to the model per question.
- Swapping the embedding model: change `EMBED_MODEL_NAME` in `indexer.py` to
  any model `fastembed` supports (see their model list) — larger models are
  more accurate but slower/heavier to download and run.

## Files
```
fom-bot/
├── bot.py               # Telegram bot: retrieval + LLM + page previews
├── indexer.py            # Shared: PDF extraction + embedding logic
├── ingest.py              # CLI: pre-build the index locally (optional)
├── requirements.txt
├── Procfile                # worker process for Railway
├── .gitignore
└── data/
    ├── FOM_REV_25B.pdf     # compressed manual (~11 MB)
    ├── chunks.json         # extracted per-page text + section titles
    └── embeddings.npy      # (only present if you ran Option B above)
```
