import os
import re
import logging
import numpy as np
import pdfplumber
from llama_cpp import Llama

# --------------------------
# CONFIGURATION
# --------------------------

# Folder containing PDF files
#PDF_FOLDER = "./pdfs"

# Chunk size (number of words per chunk); adjust as needed.
CHUNK_WORD_SIZE = 300

# Number of top similar chunks to retrieve
TOP_K = 3

# Paths to your GGUF models (update these paths as needed)
# EMBED_MODEL_PATH = "all-MiniLM-L6-v2.Q4_K_M.gguf"  # embedding model
# CHAT_MODEL_PATH = "Llama-3.2-3B-Instruct-uncensored.Q4_K_M.gguf"  # chat model
# Selene-1-Mini-Llama-3.1-8B-Q4_0.gguf
# mistral-7b-instruct-v0.3.Q4_0.gguf

SIJAINTI = "H:/tekoaly/"
MODAL_SIJAINTI = "H:/tekoaly/Embedding/"
MODALMALLI = f"{MODAL_SIJAINTI}all-MiniLM-L6-v2.Q4_K_M.gguf"
GGUFMALLI = f"{SIJAINTI}mistral-7b-instruct-v0.3.Q4_0.gguf"
PDF_SIJAINTI = "G:/code/pdftekoaly_gguf/data/"
# --------------------------
# HELPER FUNCTIONS
# --------------------------

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def split_text_into_chunks(text, chunk_size=CHUNK_WORD_SIZE):
    """
    Splits text into chunks of roughly `chunk_size` words.
    Returns a list of text chunks.
    """
    words = re.split(r'\s+', text)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two numpy arrays."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# --------------------------
# MAIN LOGIC
# --------------------------

def main():
    # Load the embedding model via llama-cpp-python.
    print("Loading embedding model...")
    embed_model = Llama(
        model_path=MODALMALLI,
        n_ctx=2048,
        seed=0,
        n_gpu_layers=-1,
        embedding=True,   # Enable embedding mode.
    )

    # Build a list of document chunks and their embeddings.
    docs = []  # Each item will be a dict with 'text' and 'embedding'
    print("Processing PDF files for embeddings...")

    # Iterate through all PDF files in the specified folder.
    for filename in os.listdir(PDF_SIJAINTI):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_SIJAINTI, filename)
            print(f"Extracting text from {filename}...")
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                print(f"No text extracted from {filename}. Skipping.")
                continue
            chunks = split_text_into_chunks(text, CHUNK_WORD_SIZE)
            print(f"Splitting {filename} into {len(chunks)} chunks.")
            for chunk in chunks:
                # Get embedding for the chunk.
                embedding = embed_model.embed(chunk)
                # Convert embedding (list) to numpy array for later similarity calculation.
                embedding_np = np.array(embedding, dtype=np.float32)
                docs.append({"text": chunk, "embedding": embedding_np})
    
    if not docs:
        print("No document chunks were processed. Exiting.")
        return

    print(f"Processed {len(docs)} document chunks.")

    # Load the chat model.
    print("Loading chat model...")
    chat_model = Llama(
        model_path=GGUFMALLI,
        n_ctx=8192,
        seed=0,
        n_gpu_layers=-1
        # Note: we use the model in text generation mode, so no `embedding=True` here.
    )

    # Chat loop.
    print("\nRAG Chat is ready. Type your questions (or 'quit' to exit).")
    while True:
        user_query = input("\nUser: ")
        if user_query.strip().lower() in {"quit", "exit"}:
            break

        # Get embedding for the user query.
        query_embedding = np.array(embed_model.embed(user_query), dtype=np.float32)

        # Compute similarity with each document chunk.
        similarities = []
        for doc in docs:
            sim = cosine_similarity(query_embedding, doc["embedding"])
            similarities.append(sim)
        
        # Get indices of the top K most similar chunks.
        top_indices = np.argsort(similarities)[-TOP_K:][::-1]

        # Concatenate the top chunks as context.
        context_chunks = [docs[i]["text"] for i in top_indices]
        context = "\n\n".join(context_chunks)

        # Prepare the prompt for the chat model.
        # Here we instruct the model to only use the provided context.
        prompt = (
            "You are an assistant that answers questions strictly based on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_query}\n\n"
            "Answer (use only the context above):"
        )

        # Generate the answer.
        print("Generating answer...")
        output = chat_model(
            prompt,
            max_tokens=8192,
            temperature=0.1,
            top_p=0.95,
            #n_gpu_layers=-1
            #stop=["\n"]
        )
        answer = output.get("choices", [{}])[0].get("text", "").strip()
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()