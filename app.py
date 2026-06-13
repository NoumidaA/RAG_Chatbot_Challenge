import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import load_embedder, load_db, retrieve

DB_DIR = "db"
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

def db_ready():
    return os.path.exists(os.path.join(DB_DIR, "index.faiss")) and os.path.exists(os.path.join(DB_DIR, "chunks.json"))

embedder = load_embedder()
index = None
chunks = None

if db_ready():
    index, chunks = load_db(DB_DIR)

tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, device_map="auto", torch_dtype="auto")
gen = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=250, do_sample=False, temperature=0.1)

def make_context(results):
    return "\n\n".join([f"[Source: {r['filename']} | Page {r['page']}]\n{r['text']}" for r in results])

def answer_question(question, history):
    global index, chunks
    if index is None or chunks is None:
        return "Vector database not found. Please run ingestion first."
    results = retrieve(question, embedder, index, chunks, top_k=5)
    if not results:
        return "No relevant chunks were found."
    context = make_context(results)
    prompt = f"""You are a helpful RAG assistant.
Answer only from the provided context.
Cite sources using [filename | page X].

Context:
{context}

Question: {question}

Answer:"""
    output = gen(prompt)[0]["generated_text"]
    answer = output.split("Answer:")[-1].strip()
    citations = "\n".join([f"- {r['filename']} | page {r['page']}" for r in results])
    return answer + "\n\nSources:\n" + citations

demo = gr.ChatInterface(fn=answer_question, title="PDF RAG Chatbot", description="Ask questions about your PDF corpus.")

if __name__ == "__main__":
    demo.launch()
