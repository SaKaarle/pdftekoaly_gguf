
# Initialize the model

import os
import pdfplumber
import numpy as np
from llama_cpp import Llama

# ----------------------------
# 1. PDF Loading and Text Chunking
# ----------------------------

def load_pdfs(PDF_SIJAINTI):
    """
    Loads all PDF files in a folder and extracts their text.
    Returns a list of tuples: (filename, full_text)
    """
    pdf_texts = []
    for filename in os.listdir(PDF_SIJAINTI):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(PDF_SIJAINTI, filename)
            print(f"Processing {full_path}...")
            try:
                with pdfplumber.open(full_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                pdf_texts.append((filename, text))
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return pdf_texts

def chunk_text(text, max_length=500):
    """
    Splits the text into chunks of roughly max_length characters.
    (A real application might want to use a smarter tokenizer.)
    """
    # Split on double newlines as a basic paragraph separator.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= max_length:
            chunks.append(para)
        else:
            # Split long paragraphs into subchunks
            for i in range(0, len(para), max_length):
                chunk = para[i:i+max_length].strip()
                if chunk:
                    chunks.append(chunk)
    return chunks

# ----------------------------
# 2. Embedding Functions and Vector Store Construction
# ----------------------------

def get_embedding(embedding_model, text):
    """
    Uses the embedding model to compute the embedding for a given text.
    This version handles the case when the model returns a list directly.
    """
    result = embedding_model.embed(text)
    
    # Check if result is a dict containing the embedding or a list
    if isinstance(result, dict) and 'embedding' in result:
        embedding_vector = result['embedding']
    elif isinstance(result, list):
        embedding_vector = result
    else:
        raise ValueError("Unexpected embedding result format.")

    # Convert the embedding vector to a NumPy array.
    return np.array(embedding_vector)


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

def build_vector_store(embedding_model, pdf_texts):
    """
    For each PDF file’s text, break it into chunks and build a vector store.
    Each entry in the store is a dict containing the text chunk, its embedding, and the source filename.
    """
    vector_store = []
    for filename, full_text in pdf_texts:
        chunks = chunk_text(full_text)
        for chunk in chunks:
            try:
                emb = get_embedding(embedding_model, chunk)
                vector_store.append({
                    'text': chunk,
                    'embedding': emb,
                    'source': filename
                })
            except Exception as e:
                print(f"Error embedding chunk from {filename}: {e}")
    return vector_store

def retrieve_relevant_chunks(vector_store, embedding_model, query, top_k=3):
    """
    Given a query, embed it and compute cosine similarity with all stored embeddings.
    Returns the top_k matching chunks.
    """
    query_emb = get_embedding(embedding_model, query)
    scored = []
    for item in vector_store:
        score = cosine_similarity(query_emb, item['embedding'])
        scored.append((score, item))
    # Sort descending by similarity score.
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [item for score, item in scored[:top_k]]
    return top_chunks

# ----------------------------
# 3. Chat Loop with Retrieval-Augmented Generation (RAG)
# ----------------------------

def chat_loop(embedding_model, vector_store, chat_model):
    """
    An interactive chat loop that retrieves context and streams responses.
    """
    print("\nEnter your queries below. Type 'exit' or 'quit' to end the session.\n")
    while True:
        query = input("User: ").strip()
        if query.lower() in ['exit', 'quit']:
            break

        # Retrieve top relevant chunks from the vector store.
        relevant_chunks = retrieve_relevant_chunks(vector_store, embedding_model, query, top_k=3)
        # Combine the chunks into one context string.
        context = "\n\n".join([f"Source: {item['source']}\nText: {item['text']}" for item in relevant_chunks])
        
        # Construct a prompt with the context.
        prompt = (
            "You are an assistant that uses the following background information to answer questions. "
            "Please provide a direct and concise answer to the question, and do not repeat or list previous examples.\n\n"
            f"Background Information:\n{context}\n\n"
            f"User Question: {query}\n"
            "Answer:"
        )

        try:
            answer = ""
            # Iterate over the streamed responses.
            for response in chat_model(prompt=prompt, max_tokens=256, stream=True):
                token = response['choices'][0]['text']
                # Filter out unwanted debug logs.
                if "llama_perf_context_print:" in token:
                    token = token.replace("llama_perf_context_print:", "")
                    if not token.strip():
                        continue

                print(token, end="")  # Print without flushing.
                answer += token

            print()  # Newline after streaming.
        except Exception as e:
            answer = f"Error generating response: {e}"
            print(answer)

# ----------------------------
# 4. Main Routine: Load Models, Build Store, and Start Chat
# ----------------------------


SIJAINTI = "H:/tekoaly/"
MODAL_SIJAINTI = "H:/tekoaly/Embedding/"
MODALMALLI = f"{MODAL_SIJAINTI}all-MiniLM-L6-v2.Q4_K_M.gguf"
GGUFMALLI = f"{SIJAINTI}Llama-3.2-3B-Instruct-uncensored.Q4_K_M.gguf"
PDF_SIJAINTI = "G:/code/tekoaly_gguf/data/"

def main():
    # Folder containing PDF files.
    #pdf_folder = "./pdfs"  # Adjust this path as needed.
    
    print("Loading PDFs...")
    pdf_texts = load_pdfs(PDF_SIJAINTI)
    if not pdf_texts:
        print("No PDFs found or failed to load PDFs.")
        return

    print("\nLoading the embedding model...")
    # Load the embedding GGUF model.
    # The parameter `embedding=True` is assumed; adjust if your API differs.
    embedding_model = Llama(
        model_path=MODALMALLI,
        n_ctx=512,
        n_threads=4,
        embedding=True  # This flag is hypothetical; check your llama-cpp-python version.
    )

    print("\nBuilding vector store from PDFs...")
    vector_store = build_vector_store(embedding_model, pdf_texts)
    if not vector_store:
        print("Failed to build the vector store.")
        return

    print("\nLoading the chat model...")
    chat_model = Llama(
        model_path=GGUFMALLI,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False  # Adjust based on your system's GPU capabilities.
    )

    print("\nAll set! Starting the chat session.")
    chat_loop(embedding_model, vector_store, chat_model)

if __name__ == "__main__":
    main()
