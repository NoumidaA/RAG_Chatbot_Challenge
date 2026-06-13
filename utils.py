import os
import re
import io
import json
import fitz
import faiss
import pytesseract
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        native_text = page.get_text("text").strip()
        text = native_text
        ocr_used = False
        if not native_text:
            pix = page.get_pixmap(dpi=180)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            ocr_used = True
        text = clean_text(text)
        if text:
            pages.append({
                "filename": os.path.basename(pdf_path),
                "page": page_num,
                "text": text,
                "ocr_used": ocr_used
            })
    doc.close()
    return pages

def chunk_text(text, chunk_size=900, overlap=200):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks

def build_chunks_from_pdf(pdf_path):
    pages = extract_pdf_pages(pdf_path)
    chunks = []
    for p in pages:
        for i, ch in enumerate(chunk_text(p["text"])):
            chunks.append({
                "id": f'{p["filename"]}_p{p["page"]}_c{i}',
                "filename": p["filename"],
                "page": p["page"],
                "chunk_id": i,
                "text": ch,
                "ocr_used": p["ocr_used"]
            })
    return chunks

def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_texts(embedder, texts):
    emb = embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return np.array(emb, dtype=np.float32)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def save_db(index, chunks, db_dir="db"):
    os.makedirs(db_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(db_dir, "index.faiss"))
    with open(os.path.join(db_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def load_db(db_dir="db"):
    index = faiss.read_index(os.path.join(db_dir, "index.faiss"))
    with open(os.path.join(db_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def retrieve(query, embedder, index, chunks, top_k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype=np.float32)
    scores, idxs = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        item = chunks[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results
