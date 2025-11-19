import os
import numpy as np
from typing import Tuple
from sentence_transformers import SentenceTransformer

from vector_store import PersistentVectorStore, sha256
from chunker import chunk_text

import markdown
from bs4 import BeautifulSoup
import PyPDF2

# Config
DOCS_DIR = "./documents"
DATA_DIR = "./data"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 16

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Load embedder (SentenceTransformer wrapper)
embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
dim = embedder.get_sentence_embedding_dimension()
store = PersistentVectorStore(dim, data_dir=DATA_DIR)


# -----------------------
# Text loaders
# -----------------------
def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def load_pdf(path: str) -> str:
    out = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                out.append(txt)
    return "\n".join(out).strip()


def load_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    soup = BeautifulSoup(raw, "html.parser")
    return soup.get_text(separator="\n").strip()


def load_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    html = markdown.markdown(raw)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n").strip()


def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return load_txt(path)
    if ext == ".pdf":
        return load_pdf(path)
    if ext in (".html", ".htm"):
        return load_html(path)
    if ext == ".md":
        return load_markdown(path)
    return ""


# -----------------------
# Process and index
# -----------------------
def process_file(path: str):
    text = load_text(path)
    if not text:
        return [], []

    # pass the embedder's tokenizer so tokenization matches the embedder
    chunks = chunk_text(text, tokenizer=embedder.tokenizer)

    new_chunks = []
    new_texts = []

    for i, chunk in enumerate(chunks):
        cid = sha256(chunk)
        if store.has_id(cid):
            continue
        meta = {
            "id": cid,
            "text": chunk,
            "source": os.path.basename(path),
            "chunk_index": i
        }
        new_chunks.append(meta)
        new_texts.append(chunk)

    return new_chunks, new_texts


def main():
    all_new_meta = []
    all_new_texts = []

    files = sorted(os.listdir(DOCS_DIR))
    for fn in files:
        path = os.path.join(DOCS_DIR, fn)
        if not os.path.isfile(path):
            continue
        new_meta, new_texts = process_file(path)
        if new_meta:
            print(f"[+] {fn}: {len(new_meta)} new chunks")
        else:
            print(f"[-] {fn}: no new chunks")
        all_new_meta.extend(new_meta)
        all_new_texts.extend(new_texts)

    if not all_new_texts:
        print("No new chunks to index.")
        return

    # embed in batches
    embeddings_batches = []
    for i in range(0, len(all_new_texts), BATCH_SIZE):
        batch_texts = all_new_texts[i:i+BATCH_SIZE]
        vecs = embedder.encode(batch_texts, convert_to_numpy=True, show_progress_bar=True)
        embeddings_batches.append(vecs)

    embeddings = np.vstack(embeddings_batches)
    store.add(embeddings, all_new_meta)
    print(f"Indexed {len(all_new_meta)} new chunks.")


if __name__ == "__main__":
    main()
