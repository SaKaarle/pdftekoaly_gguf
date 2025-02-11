'''
This script is modal GGUF example script
'''
import os
import sys
import logging
import json
import pdfplumber
from llama_cpp import Llama

# all-MiniLM-L6-v2.Q4_K_M.gguf
# nomic-embed-text-v1.5.Q4_K_M.gguf
# Llama-3.2-3B-Instruct-uncensored.Q4_K_M.gguf
SIJAINTI = "H:/tekoaly/"
MODAL_SIJAINTI = "H:/tekoaly/Embedding/"
MODALMALLI = "all-MiniLM-L6-v2.Q4_K_M.gguf"
GGUFMALLI = f"{SIJAINTI}Llama-3.2-3B-Instruct-uncensored.Q4_K_M.gguf"
PDF_SIJAINTI = "G:/code/pdftekoaly_gguf/data/"


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def read_pdf_text(pdf_path):
    """
    Read PDF and extract text.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as p:
            for page in p.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        logging.error("Error reading PDF %s: %s", pdf_path, str(e))
    return text

def split_text(text, chunk_size=100):
    """
    Split text into chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def process_document_with_llama(pdf_path, model_path, output_path="embeddings/"):
    """
    Process document using Llama model and save embeddings.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Initialize Llama
        llm = Llama(model_path=model_path, n_ctx=2048, embedding=True)
        text = read_pdf_text(pdf_path)
        if not text.strip():
            logging.warning("No text extracted from %s. Skipping.")
            return

        # Split text if needed (e.g., based on token size, not implemented here)
        response = llm.create_embedding([text])

        # Save the embeddings to a file
        output_file = os.path.join(output_path, os.path.basename(pdf_path).replace(".pdf", "_embedding.txt"))
        with open(output_file, "w",encoding='utf-8') as f:
            f.write(str(response))
        logging.info("Saved embeddings for %s to %s",pdf_path,output_file)

        # Save the embeddings to a JSON file
        output_file_json = os.path.join(output_path, os.path.basename(pdf_path).replace(".pdf", "_embedding.json"))
        with open(output_file_json, "w",encoding='utf-8') as f_json:
            json.dump(response, f_json, indent=4)
        logging.info("Saved embeddings for %s to %s",pdf_path,output_file_json)

    except Exception as e:
        logging.error("Error processing %s with Llama: %s",pdf_path, str(e))

def get_pdf_files(directory):
    """
    Get all PDF files in a specified directory.
    """
    try:
        pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
        return pdf_files
    except Exception as e:
        logging.error("Error accessing directory %s: %s", directory,str(e))
        return []

def start_chatting(output_path="embeddings/", chat_model_path=GGUFMALLI):
    """
    Start chatting using embeddings from all PDFs.
    """
    try:
        chat_llm = Llama(model_path=chat_model_path, n_ctx=2048, embedding=True)
        
        while True:
            prompt = input("Type your prompt below\n")
            if not prompt:
                continue
            for filename in os.listdir(output_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(output_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        response = chat_llm.create_embedding([text])
                        print(f"From {filename}: {response}")
    except Exception as e:
        logging.error("Error in start_chatting(): %s", str(e))


def main():
    '''
    Main().
    '''

    model_path = os.path.join(MODAL_SIJAINTI, MODALMALLI)  # Update with your model path
    output_path = "embeddings/"  # Define the directory to save embeddings

    # Get all PDF files in the specified directory
    pdf_files = get_pdf_files(PDF_SIJAINTI)
    if not pdf_files:
        logging.warning("No PDF files found. Exiting.")
        return

    for pdf_path in pdf_files:
        logging.info("Processing %s",pdf_path)
        process_document_with_llama(pdf_path, model_path, output_path)
    # START THE CHAT MODE
    start_chatting()

if __name__ == "__main__":
    main()
