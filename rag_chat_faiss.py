import os
import re
import numpy as np
import pdfplumber
import faiss
from llama_cpp import Llama

# --------------------------
# CONFIGURATION
# --------------------------

# Folder containing PDF files
PDF_FOLDER = "./pdfs"

# Chunk size (number of words per chunk)
CHUNK_WORD_SIZE = 300

# Number of top similar chunks to retrieve
TOP_K = 3

# Paths to your GGUF models (update these paths as needed)
EMBED_MODEL_PATH = "all-MiniLM-L6-v2.Q4_K_M.gguf"  # embedding model
CHAT_MODEL_PATH = "Llama-3.2-3B-Instruct-uncensored.Q4_K_M.gguf"  # chat model

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

def split_text_into_chunks(text, chunk_size=300):
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

def l2_normalize(vecs):
    """L2-normalize a 2D numpy array (each row is a vector)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    return vecs / norms

# --------------------------
# MAIN LOGIC
# --------------------------

def main():
    # Load the embedding model via llama-cpp-python.
    print("Loading embedding model...")
    embed_model = Llama(
        model_path=EMBED_MODEL_PATH,
        n_ctx=512,
        seed=0,
        embedding=True,   # Enable embedding mode.
    )

    # Build a list of document chunks and store their embeddings.
    doc_texts = []  # To hold text chunks
    doc_embeddings = []  # To hold embeddings (as numpy arrays)

    print("Processing PDF files for embeddings...")

    # Iterate through all PDF files in the specified folder.
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
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
                embedding_np = np.array(embedding, dtype=np.float32)
                doc_texts.append(chunk)
                doc_embeddings.append(embedding_np)

    if not doc_texts:
        print("No document chunks were processed. Exiting.")
        return

    # Convert list of embeddings into a numpy array.
    embeddings_np = np.vstack(doc_embeddings)  # shape: (n_chunks, embedding_dim)
    print(f"Processed {embeddings_np.shape[0]} document chunks with dimension {embeddings_np.shape[1]}.")

    # Normalize embeddings to use cosine similarity (cosine similarity = inner product on normalized vectors)
    embeddings_np = l2_normalize(embeddings_np)

    # Build FAISS index using inner product (IP)
    embedding_dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_np)
    print("FAISS index built and embeddings added.")

    # Load the chat model.
    print("Loading chat model...")
    chat_model = Llama(
        model_path=CHAT_MODEL_PATH,
        n_ctx=512,
        seed=0,
    )

    # Chat loop.
    print("\nRAG Chat is ready. Type your questions (or 'quit' to exit).")
    while True:
        user_query = input("\nUser: ")
        if user_query.strip().lower() in {"quit", "exit"}:
            break

        # Get embedding for the user query.
        query_embedding = np.array(embed_model.embed(user_query), dtype=np.float32)
        # Normalize query embedding.
        query_embedding /= np.linalg.norm(query_embedding) if np.linalg.norm(query_embedding) != 0 else 1
        query_embedding = query_embedding.reshape(1, -1)

        # Search in FAISS index for top K similar chunks.
        distances, indices = index.search(query_embedding, TOP_K)
        # indices is a 2D array; we use the first (and only) row.
        top_indices = indices[0]
        print(f"Top indices from FAISS: {top_indices}")

        # Concatenate the top chunks as context.
        context_chunks = [doc_texts[i] for i in top_indices if i < len(doc_texts)]
        context = "\n\n".join(context_chunks)

        # Prepare the prompt for the chat model.
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
            max_tokens=256,
            temperature=0.3,
            top_p=0.95,
            stop=["\n"]
        )
        answer = output.get("choices", [{}])[0].get("text", "").strip()
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
