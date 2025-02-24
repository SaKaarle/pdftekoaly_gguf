'''
Paras mahdollinen RAG joka voisi toimia jos halutaan testailla datan käsittelyä
Ainoastaan purkaa PDF tiedoston onnistuneesti.
Mutta Alumiiniprofiilien datat voisivat olla CSV / Excel / SQL taulukoissa varmemmin tallessa.
Sitten varmemmin hakea funktio-hauilla tietoja ja verrata esimerkki tilaukseen.
Koodiin voisi lisätä mahdollisen tiktoken tai tokeni laskijan debugaamiseen.
Tämän hetkinen configuraatio vie 6GB VRAMia.

Voitaisiin lisätä mahdollisesti dynaaminen Chunkkaus ja overlappaus.

Numpy versio FYI: 

`pip install numpy==1.25.*`


'''
import os
import glob
import json
import re
import numpy as np
import pdfplumber
from llama_cpp import Llama

# KANSIOT JA SIJAINNIT:

PDF_SIJAINTI = "G:/code/pdftekoaly_gguf/pdf_data/"  # Folder containing PDF files
# "G:/code/pdftekoaly_gguf/data/" 

SIJAINTI = "H:/tekoaly/"
MODAL_SIJAINTI = "H:/tekoaly/Embedding/"
MODALMALLI = f"{MODAL_SIJAINTI}all-MiniLM-L6-v2.Q4_K_M.gguf"  # Embedding model
GGUFMALLI = f"{SIJAINTI}mistral-7b-instruct-v0.3.Q4_0.gguf"     # Main generation model

# MUUTTUVAT:
# esimerkiksi

CHUNK_SIZE = 300
OVERLAP = 100

# FUNKTIOT:

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        print(f"[DEBUG] Extracted text from {pdf_path} ({len(text)} characters).")
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}")
        # Handle error for avoiding returning None
        text = ""
    return text


def chunk_text(text):
    """
    Improved chunking method that splits text into sentences first,
    then groups sentences into chunks that have approximately chunk_size words,
    while preserving sentence boundaries. It also maintains an overlap between chunks
    by including the last few sentences
    from the previous chunk until reaching the overlap word count.
    
    Parameters:
        text (str): The full text to be chunked.
        chunk_size (int): Approximate target word count for each chunk.
        overlap (int): Target word count to overlap between consecutive chunks.
        
    Returns:
        List[str]: A list of text chunks.
    """
    # Split text into sentences using a regex that looks for sentence-ending punctuation.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        # If adding this sentence would exceed our chunk size and the current chunk is not empty,
        # then finish the current chunk.
        if current_count + sentence_word_count > CHUNK_SIZE and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Prepare the next chunk by retaining an overlap of sentences from the current chunk.
            overlap_chunk = []
            overlap_count = 0
            # Iterate backward over the current chunk to gather sentences for overlap.
            for s in reversed(current_chunk):
                s_word_count = len(s.split())
                if overlap_count + s_word_count <= OVERLAP:
                    overlap_chunk.insert(0, s)  # Insert at the beginning to maintain order.
                    overlap_count += s_word_count
                else:
                    break
            current_chunk = overlap_chunk.copy()
            current_count = overlap_count

        # Add the current sentence to the chunk.
        current_chunk.append(sentence)
        current_count += sentence_word_count

    # Append any remaining sentences as the last chunk.
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"[DEBUG] Text split into {len(chunks)} chunks using improved method.")
    return chunks

def get_embedding(text, embed_model):
    """
    Get an embedding for the given text using llama-cpp-python.
    Uses the .embed() method to obtain embeddings.
    """
    try:
        embedding = embed_model.embed(text)
        # If the result is a dict, extract the embedding
        if isinstance(embedding, dict) and "embedding" in embedding:
            embedding = embedding["embedding"]
        # Convert numpy array embeddings to a list for JSON serialization
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        return embedding
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# PDF PROCESSING

def process_pdf_file(pdf_path, embed_model):
    """
    Process a single PDF:
      - Extract text (and save to .txt for debugging)
      - Split text into chunks
      - Generate an embedding for each chunk
      - Save the embeddings (with the corresponding text) to a JSON file.
    """
    print(f"\n[INFO] Processing PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)

    # Save raw text for debugging
    txt_path = os.path.splitext(pdf_path)[0] + ".txt"
    try:
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(text)
        print(f"[DEBUG] Saved extracted text to {txt_path}")
    except Exception as e:
        print(f"[ERROR] Could not save text file {txt_path}: {e}")

    # Chunk the text
    chunks = chunk_text(text)

    embedded_data = []
    for idx, chunk in enumerate(chunks):
        print(f"[INFO] Embedding chunk {idx + 1}/{len(chunks)}...")
        embedding = get_embedding(chunk, embed_model)
        if embedding is not None:
            embedded_data.append({
                "chunk": chunk,
                "embedding": embedding
            })

    # Save embeddings for this PDF
    embedding_file = os.path.splitext(pdf_path)[0] + "_embeddingsNew.json"
    try:
        with open(embedding_file, "w", encoding="utf-8") as f_emb:
            json.dump(embedded_data, f_emb, indent=4)
        print(f"[DEBUG] Saved embeddings to {embedding_file}")
    except Exception as e:
        print(f"[ERROR] Could not save embeddings to {embedding_file}: {e}")
    return embedded_data

def process_all_pdfs(pdf_folder, embed_model):
    """
    Process all PDF files in the specified folder.
    Returns a list of all chunk embeddings across PDFs.
    """

    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    print(f"[INFO] Found {len(pdf_files)} PDF file(s) in {pdf_folder}.")
    all_embeddings = []
    for pdf in pdf_files:
        embeddings = process_pdf_file(pdf, embed_model)
        all_embeddings.extend(embeddings)
    print(f"[INFO] Processed all PDFs; total chunks embedded: {len(all_embeddings)}")
    return all_embeddings

# RETRIEVAL CHUNKS FUNC

def retrieve_relevant_chunks(query, embed_model, all_embeddings, top_k=3):
    """
    Compute the query's embedding and return the top_k chunks
    from all_embeddings with the highest cosine similarity.
    Includes detailed debugging to show the similarity scores.
    """
    print("\n[INFO] Computing query embedding...")
    query_embedding = get_embedding(query, embed_model)
    if query_embedding is None:
        return []

    scored = []
    print("[DEBUG] Comparing query embedding with stored embeddings:")
    # Compare each stored embedding with the query embedding.
    for idx, item in enumerate(all_embeddings):
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append((score, item))
        # Show the similarity score for each chunk (showing first 100 chars of text for context)
        snippet = item["chunk"][:100].replace("\n", " ")
        print(f"  [DEBUG] Chunk {idx}: score = {score:.4f} | text snippet: {snippet}...")

    # Sort the scored list by similarity score (highest first).
    # En ole ennemmin käyttänyt lambdaa. Pitää tarkistaa miten tämä toimii konepellin alla.

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [item for score, item in scored[:top_k]]

    print(f"[DEBUG] Selected top {top_k} chunks:")
    for rank, (score, item) in enumerate(scored[:top_k], start=1):
        snippet = item["chunk"][:100].replace("\n", " ")
        print(f"  [DEBUG] Rank {rank}: score = {score:.4f} | text snippet: {snippet}...")
    return top_chunks

def answer_query(query, main_model, embed_model, all_embeddings):
    """
    Retrieve the most relevant chunks and build a prompt
    for the main language model (GGUFMALLI) to answer the question.
    """
    relevant_chunks = retrieve_relevant_chunks(query, embed_model, all_embeddings)
    if not relevant_chunks:
        return "No relevant context found."

    # Build a context prompt from the retrieved chunks.
    context = "\n---\n".join(chunk["chunk"] for chunk in relevant_chunks)
    #Answer and must provide and give information prof.nr of set dimension and top 3 results for the possible aluminium profile and it's profile number. Dimensional details and notations of aluminium profiles that is asked by the user.
    prompt = (
        "Answer the question based solely on the following context.\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {query}\n"
    )
    print("\n[DEBUG] Prompt for main model constructed:")
    print(prompt)
    try:
        # Generate answer using the main model.
        response = main_model(prompt, max_tokens=4096,temperature=0.6)
        answer = response["choices"][0]["text"].strip()
    except Exception as e:
        answer = f"[ERROR] Main model generation failed: {e}"
    return answer


def main():
    '''
    Main
    '''
    # STEP 1: Process PDFs and create embeddings
    # Load the embedding model (all-miniLM) using llama-cpp-python.
    # Adjust n_gpu_layers and other parameters as needed for your setup.

    print(f"[INFO] Loading embedding model from: {MODALMALLI}")
    embed_model = Llama(
        model_path=MODALMALLI,
        n_gpu_layers=-1,  # adjust this value for your setup
        verbose=False,
        embedding=True # Loads Embedded model.
    )

    # Process all PDFs in the designated folder.
    all_embeddings = process_all_pdfs(PDF_SIJAINTI, embed_model)

    # Close (or delete) the embedding model instance used for PDF processing.

    # del embed_model
    # print("[INFO] Finished embedding PDFs and released the embedding model.")

    # STEP 2: Q&A using Retrieval-Augmented Generation (RAG)
    # Load the main generation model (GGUFMALLI)

    print(f"\n[INFO] Loading main model from: {GGUFMALLI}")
    main_model = Llama(
        model_path=GGUFMALLI,
        n_gpu_layers=-1,  # adjust as needed for your setup
        verbose=False,
        n_ctx=8192 # depends of the chunked and overlapped size and numbers.
    )

    # # For computing query embeddings during Q&A, reinitialize the embedding model.
    # print(f"[INFO] Reloading embedding model for query embeddings from: {MODALMALLI}")
    # query_embed_model = Llama(
    #     model_path=MODALMALLI,
    #     n_gpu_layers=-1,
    #     verbose=True,
    #     embedding=True,
    # )

    # Q&A loop
    print("\n[INFO] Entering Q&A loop. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.strip().lower() == "exit":
            break
        try:
            answer = answer_query(query, main_model, embed_model, all_embeddings)
            print(f"\n{answer}")
        except Exception as e:
            print(f"[ERROR] An error occurred during the Q&A process: {e}")


    # Cleanup (if needed)
    del main_model
    del embed_model
    print("[INFO] Q&A session ended.")

if __name__ == "__main__":
    main()
