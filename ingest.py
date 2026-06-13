import os
import glob
from tqdm import tqdm
from utils import build_chunks_from_pdf, load_embedder, embed_texts, build_faiss_index, save_db

PDF_DIR = "data/pdfs"
DB_DIR = "db"

def main():
    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_files:
        raise RuntimeError("No PDFs found in data/pdfs")
    all_chunks = []
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        all_chunks.extend(build_chunks_from_pdf(pdf))
    embedder = load_embedder()
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(embedder, texts)
    index = build_faiss_index(embeddings)
    save_db(index, all_chunks, DB_DIR)
    print(f"Saved {len(all_chunks)} chunks to {DB_DIR}")

if __name__ == "__main__":
    main()
