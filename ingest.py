"""
ingest.py — one-time build step (optional but recommended).

Pre-builds the embedding index locally so your Railway deploy starts fast
instead of building it on first boot. Run this whenever the manual PDF at
data/FOM_REV_25B.pdf is replaced with a new revision.

    pip install -r requirements.txt
    python ingest.py

Outputs into data/:
    chunks.json      -> [{page, section, text}, ...]
    embeddings.npy   -> (N, 384) float32 matrix, one row per chunk
"""

from indexer import build_index

if __name__ == "__main__":
    build_index(save=True)
    print("Done.")
