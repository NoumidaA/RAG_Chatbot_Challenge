---
title: PDF RAG Chatbot
sdk: gradio
app_file: app.py
emoji: 📚
colorFrom: blue
colorTo: green
---

# PDF RAG Chatbot

Open-source RAG chatbot for large PDF corpora.

## Features
- Native PDF text extraction
- OCR for scanned pages
- Chunking with metadata
- FAISS vector search
- Hugging Face generation model
- Source citations by file and page

## Local run

```bash
pip install -r requirements.txt
python ingest.py
python app.py
```

## Folder structure

- `data/pdfs/`: place input PDFs here
- `db/`: saved FAISS index and chunk metadata

## Notes
If `db/index.faiss` is missing, run `python ingest.py` first.
