import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# === CONFIG ===
PDF_DIR = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_DIM = 384  # for MiniLM
OLLAMA_MODEL = "tinyllama"
OLLAMA_URL = "http://localhost:11434/api/generate"

# === STEP 1: Extract Text from PDFs ===
def extract_text_from_pdfs(pdf_dir):
    texts = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_dir, file))
            text = ""
            for page in doc:
                text += page.get_text()
            texts.append(text)
    return texts

# === STEP 2: Chunk Text ===
def chunk_texts(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

# === STEP 3: Embed Chunks ===
def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings, model

# === STEP 4: Store in FAISS ===
def store_embeddings_faiss(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# === STEP 5: Retrieve Relevant Chunks ===
def retrieve_faiss(query, embed_model, index, chunks):
    query_embedding = embed_model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), k=5)
    return "\n".join([chunks[i] for i in I[0]])

# === STEP 6: Generate Answer with Ollama (Streaming) ===
import json

def generate_answer(context, query):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt},
            stream=True
        )
        answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    answer += data.get("response", "")
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
        return answer.strip() if answer else "⚠️ Ollama returned no usable output."
    except Exception as e:
        return f"❌ Failed to connect to Ollama: {e}"

# === MAIN ===
if __name__ == "__main__":
    print("🔍 Extracting text...")
    texts = extract_text_from_pdfs(PDF_DIR)

    print("✂️ Chunking...")
    chunks = chunk_texts(texts)

    print("🧬 Embedding...")
    embeddings, embed_model = embed_chunks(chunks)

    print("📦 Storing in FAISS...")
    index = store_embeddings_faiss(embeddings)

    print("💬 Ready for queries.")
    query = input("Enter your question: ")
    context = retrieve_faiss(query, embed_model, index, chunks)

    print("🦙 Generating answer...")
    answer = generate_answer(context, query)
    print("\nAnswer:\n", answer)