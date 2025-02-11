import os
import re
import logging
import numpy as np
import pdfplumber
import json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def read_pdf_text(pdf_path):
    """
    Read text from a PDF file.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text.strip()

def chunk_text(text):
    """
    Split the input text into chunks.
    Returns a list of chunked texts.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = []
    current_word_count = 0
    print(sentences)

    for sentence in sentences: 
        if not sentence.strip():
            continue
        current_chunk.append(sentence)
        current_word_count += len(sentence.split())
        if current_word_count >= CHUNK_WORD_SIZE:
            yield ' '.join(current_chunk)
            print(current_chunk)
            current_chunk = []
            current_word_count = 0
        # Yield the last chunk if it's not empty
    if current_chunk: 
        yield ' '.join(current_chunk)


def process_embed_model(model_path):
    """
    Load and process an embedding model.
    Returns the loaded model.
    """
    pdf_files = [os.path.join(PDF_SIJAINTI, f) for f in os.listdir(PDF_SIJAINTI) if f.endswith('.pdf')]
    llm = None
    try:
        llm = Llama(model_path=model_path,
                    n_ctx=2048,
                    embedding=True,
                    n_gpu_layers=-1,
                    )
        for pdf_path in pdf_files:
            text = read_pdf_text(pdf_path)  # Read text from PDF
            chunks = chunk_text(text)

            if not text.strip():
                logging.warning("No Text Extracted from %s. Skipping.",pdf_path)
                continue
            embeddings = []
            for chunk in chunks:

                response = llm.create_embedding([chunk])
                embeddings.append(response)
                #logging.info(f"Embedding chunk {i+1}/{len(chunks)} for PDF: {pdf_path}")

            output_file = os.path.join(PDF_SIJAINTI,
                                       os.path.basename(pdf_path).replace(".pdf","_embedding.json"))
            with open(output_file, "w",encoding='utf-8') as f:
                json.dump(response,f,indent=4)
                #JSON import and json output for clearer message.
                logging.info("Saved embeddings for %s to %s", pdf_path,output_file)
    finally:
        if llm is not None and hasattr(llm, 'close') and callable(getattr(llm,'close')):
            llm.close()
        else:
            logging.warning("Model couldn't be closed.")

if __name__ == "__main__":
    process_embed_model(MODALMALLI)

# def split_text_into_chunks(text, chunk_size=CHUNK_WORD_SIZE):
#     """
#     Splits text into chunks of roughly `chunk_size` words.
#     Returns a list of text chunks.
#     """
#     words = re.split(r'\s+', text)
#     chunks = []
#     for i in range(0, len(words), chunk_size):
#         chunk = " ".join(words[i:i+chunk_size]).strip()
#         if chunk:
#             chunks.append(chunk)
#     return chunks

# def cosine_similarity(vec1, vec2):
#     """Compute cosine similarity between two numpy arrays."""
#     if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
#         return 0.0
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# # --------------------------
# # MAIN LOGIC
# # --------------------------

# def main():
    

# if __name__ == "__main__":
#     process_document_with_llama(MODALMALLI, MODAL_SIJAINTI)