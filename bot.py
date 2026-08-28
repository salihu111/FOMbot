"""
bot.py — Telegram bot that answers First Officer questions strictly from the
Ethiopian Airlines Flight Operations Manual (FOM Rev.25B).

- Retrieval: real semantic embeddings (fastembed / BAAI/bge-small-en-v1.5)
  over per-page chunks — not keyword matching, so paraphrased or
  differently-worded questions still find the right section.
- Generation: Groq-hosted openai/gpt-oss-120b, constrained to the manual only
- "Smarter fallback": if no strong match is found, widens the search to
  neighboring pages and asks the model to surface the closest relevant
  section instead of a flat "not found"
- Source-page previews: each cited page becomes a tappable inline button
  that renders and sends the real manual page as an image
"""

import logging
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from groq import Groq
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from indexer import embed_query, load_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---- config -----------------------------------------------------------
TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "openai/gpt-oss-120b"
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.45   # cosine similarity below this triggers fallback widening
PAGE_CACHE_DIR = Path("/tmp/fom_page_cache")
PDF_PATH = Path("data/FOM_REV_25B.pdf")

PAGE_CACHE_DIR.mkdir(exist_ok=True)
#groq_client = Groq(api_key=GROQ_API_KEY)
groq_client = Groq(
    api_key=GROQ_API_KEY,
    timeout=60.0,
    max_retries=2,
)

SYSTEM_PROMPT = """You are a flight operations reference assistant for Ethiopian \
Airlines First Officers, answering strictly from the excerpts of the Flight \
Operations Manual (FOM Rev.25B) provided below each question.

Rules:
- Use ONLY the provided excerpts. Never use outside knowledge, never guess, \
never invent regulations or numbers.
- Be brief: a few sentences or a short list, not a full essay.
- Always cite the page(s) you used, in the exact format [p.NN].
- If the excerpts do not directly answer the question, say so plainly in one \
short sentence, then briefly point to the closest relevant information that \
IS in the excerpts (still citing pages). Do not just say "not found" and stop.
- If truly nothing in the excerpts is relevant, say the manual excerpts \
retrieved don't cover this and suggest the crew member rephrase or check the \
relevant chapter directly.
"""

# ---- load index at startup (auto-builds if data/embeddings.npy is missing) --
log.info("Loading index (this embeds all chunks on first run if not prebuilt)...")
CHUNKS, MATRIX = load_index()
PAGE_TO_CHUNK = {c["page"]: c for c in CHUNKS}
log.info(f"Loaded {len(CHUNKS)} page-chunks, embeddings shape {MATRIX.shape}.")


def retrieve(question: str):
    """Return (chunks_used, was_fallback)."""
    qv = embed_query(question)
    sims = MATRIX @ qv  # cosine similarity (rows already L2-normalized)
    ranked = np.argsort(sims)[::-1][:TOP_K]
    top_score = float(sims[ranked[0]]) if len(ranked) else 0.0

    fallback = top_score < CONFIDENCE_THRESHOLD
    picked_pages = {CHUNKS[i]["page"] for i in ranked}

    if fallback:
        # widen: pull in neighboring pages of each top hit, in case the
        # relevant passage spans a page break or uses very different wording
        for i in list(ranked):
            p = CHUNKS[i]["page"]
            picked_pages.update({p - 1, p, p + 1})

    used = [PAGE_TO_CHUNK[p] for p in sorted(picked_pages) if p in PAGE_TO_CHUNK]
    return used, fallback


def ask_llm(question: str, chunks: list, fallback: bool) -> str:
    context = "\n\n".join(
        f"--- Page {c['page']} | {c['section']} ---\n{c['text']}" for c in chunks
    )
    note = (
        "\n\n(Note: no high-confidence match was found for this question — "
        "the excerpts below are the closest related pages found.)"
        if fallback else ""
    )
    user_msg = f"Manual excerpts:{note}\n\n{context}\n\nQuestion: {question}"

    completion = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return completion.choices[0].message.content.strip()


def extract_cited_pages(answer: str, fallback_chunks: list) -> list:
    pages = {int(m) for m in re.findall(r"\[p\.(\d+)\]", answer)}
    if not pages:
        # model didn't cite in the exact format; fall back to whichever
        # chunks were actually used for retrieval
        pages = {c["page"] for c in fallback_chunks[:3]}
    return sorted(pages)[:6]  # cap buttons at 6


def render_page_image(page_num: int) -> Path:
    cached = PAGE_CACHE_DIR / f"page_{page_num}.png"
    if cached.exists():
        return cached
    doc = fitz.open(PDF_PATH)
    page = doc[page_num - 1]  # 0-indexed
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # ~144dpi, crisp on phone
    pix.save(cached)
    doc.close()
    return cached


# ---- telegram handlers --------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "FOM Assistant ready ✈️\n"
        "Ask me anything from the Flight Operations Manual (Rev.25B) and "
        "I'll answer from it directly, with page citations you can tap to view."
    )


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    chunks, fallback = retrieve(question)
    answer = ask_llm(question, chunks, fallback)
    cited_pages = extract_cited_pages(answer, chunks)

    keyboard = None
    if cited_pages:
        buttons = [
            InlineKeyboardButton(f"📄 Page {p}", callback_data=f"page:{p}")
            for p in cited_pages
        ]
        rows = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
        keyboard = InlineKeyboardMarkup(rows)

    await update.message.reply_text(answer, reply_markup=keyboard)


async def handle_page_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    page_num = int(query.data.split(":")[1])
    try:
        img_path = render_page_image(page_num)
        with open(img_path, "rb") as f:
            await context.bot.send_photo(
                chat_id=query.message.chat_id,
                photo=f,
                caption=f"FOM Rev.25B — Page {page_num}",
            )
    except Exception as e:
        log.exception("page render failed")
        await context.bot.send_message(query.message.chat_id, f"Couldn't render page {page_num}: {e}")


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_page_button, pattern=r"^page:\d+$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    log.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
