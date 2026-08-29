"""
indexer.py — shared logic used by both ingest.py (manual CLI build) and
bot.py (auto-build on first run if the index is missing).

Extraction: PyMuPDF, per page, boilerplate stripped, section titles detected.
Embedding: fastembed (BAAI/bge-small-en-v1.5) — a small ONNX model, no torch,
           runs on CPU, ~130MB model downloaded automatically on first use
           and cached locally. Real semantic embeddings, not TF-IDF.
"""

import json
import re
from pathlib import Path

import numpy as np

PDF_PATH = Path("data/FOM_REV_25B.pdf")
CHUNKS_PATH = Path("data/chunks.json")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

BOILERPLATE_PATTERNS = [
    re.compile(r"^ETHIOPIAN AIRLINES", re.I),
    re.compile(r"^FLIGHT OPERATIONS\b", re.I),
    re.compile(r"^\d{1,2}-[A-Z]{3}-\d{4}$"),
    re.compile(r"^MANUAL$", re.I),
    re.compile(r"^UNCONTROLLED WHEN COPIED", re.I),
    re.compile(r"^REV\.?\s*(NO\.?)?\s*\S+", re.I),
]
SECTION_RE = re.compile(r"^(SECTION|CHAPTER|APPENDIX)\s+[\d.A-Z]+", re.I)


def clean_page(raw_text: str):
    """Strip repeated header/footer lines; return (clean_text, section_title)."""
    lines = raw_text.split("\n")
    out, section = [], None
    seen_section_line = False
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if any(p.search(s) for p in BOILERPLATE_PATTERNS):
            continue
        if SECTION_RE.match(s):
            if section is None:
                section = s
            if seen_section_line:
                continue
            seen_section_line = True
        out.append(s)
    return "\n".join(out), section


def extract_chunks() -> list:
    """Extract per-page text chunks from the PDF. Returns list of dicts."""
    import fitz  # PyMuPDF

    doc = fitz.open(PDF_PATH)
    chunks = []
    last_section = None
    for i, page in enumerate(doc):
        raw = page.get_text()
        text, section = clean_page(raw)
        if section:
            last_section = section
        if len(text) < 20:
            continue  # skip near-blank pages (dividers, etc.)
        chunks.append({
            "page": i + 1,  # 1-indexed, matches printed manual page
            "section": section or last_section or "Unknown section",
            "text": text,
        })
    doc.close()
    return chunks


def embed_texts(texts: list, batch_size: int = 16) -> np.ndarray:
    """Embed texts in small batches to keep RAM usage under 200MB on Railway."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    all_vectors = []
    
    # Process pages in small chunks of 16 so RAM doesn't spike
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = list(model.embed(batch))
        all_vectors.extend(vectors)
        
    matrix = np.array(all_vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms



def embed_query(query: str) -> np.ndarray:
    """Embed a single query string into a (384,) L2-normalized float32 vector."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    vec = list(model.embed([query]))[0].astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_index(save: bool = True):
    """Full build: extract chunks, embed them, optionally save to data/."""
    print("Extracting text from PDF...")
    chunks = extract_chunks()
    print(f"  {len(chunks)} page-chunks extracted")

    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL_NAME} "
          f"(downloads the model on first run, ~130MB)...")
    corpus = [f"{c['section']}. {c['text']}" for c in chunks]
    matrix = embed_texts(corpus)
    print(f"  embeddings shape: {matrix.shape}")

    if save:
        CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=1))
        np.save(EMBEDDINGS_PATH, matrix)
        print(f"  saved -> {CHUNKS_PATH}, {EMBEDDINGS_PATH}")

    return chunks, matrix


def load_index():
    """Load a previously built index, or build it if missing (auto-build)."""
    if CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists():
        chunks = json.loads(CHUNKS_PATH.read_text())
        matrix = np.load(EMBEDDINGS_PATH)
        return chunks, matrix
    print("No prebuilt index found — building it now (one-time, may take a "
          "few minutes on first deploy)...")
    return build_index(save=True)
