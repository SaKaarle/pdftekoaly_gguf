'''
Paras mahdollinen RAG joka voisi toimia jos halutaan testailla datan käsittelyä
Ainoastaan purkaa PDF tiedoston onnistuneesti.
Mutta Alumiiniprofiilien datat voisivat olla CSV / Excel / SQL taulukoissa varmemmin tallessa.
Sitten varmemmin hakea funktio-hauilla tietoja ja verrata esimerkki tilaukseen.
Koodiin voisi lisätä mahdollisen tiktoken tai tokeni laskijan debugaamiseen.
Tämän hetkinen configuraatio vie 6GB VRAMia.

Voitaisiin lisätä mahdollisesti dynaaminen Chunkkaus ja overlappaus.

Conda conffeja:

Muista PATH / Enviroment Variables...

Powershell: Poistamalla rajoitteita `set-executionpolicy remotesigned` että pystytään avaamaan PowerShellillä/Visual Studio Codella
esim. `conda activate gguf` automaattisesti. Pystytään helpommin hallitsemaan VENVejä ilman, että tarvitsee vaihdella CMD ja
PS välillä. Myös helpompi laittaa `$env: ...` komentoja ja asentaa Vulkan tai Cuda versio Llama.cpp.pythonista.

`conda config --set auto_activate_base false` ottaa pois automaattisen aktivoinnin kun avataan esim Powershell tietokoneella.
  
Numy versio FYI: 
`pip install numpy==1.25.*`
  
Windows Powershell terminaaliin:
p  
`conda activate tekoalyllama` tai oma virtualenv
  
AMD tai Intel tai muu GPU: `$env:CMAKE_ARGS="-DGGML_VULKAN=on"` Windowsilla aktivoidaan Vulkan ajurien asentaminen
  
Nvidia GPU: Asennettuna Cuda Toolkit: https://developer.nvidia.com/cuda-downloads ja sen jälkeen: `$env:CMAKE_ARGS="-DGGML_CUDA=on"`

```
PS C:\\Users\\Saku-Laptop> nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:42:46_Pacific_Standard_Time_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```
  
`pip install llama-cpp-python  --no-cache-dir --verbose` ja itse asennus Llama-cpp-pythonille.` --force ` jos pitää overwritettää asennus.

'''

import os
import glob
import json
import re
import numpy as np
import pdfplumber
from llama_cpp import Llama

# KANSIOT JA SIJAINNIT:

#Läppäri:
# D:/RAG/
# D:/tekoalymallit/

# PDF_SIJAINTI = "D:/RAG/"

# SIJAINTI = "D:/tekoalymallit/"
# MODAL_SIJAINTI = "D:/tekoalymallit/Embedding/"
# MODALMALLI = f"{MODAL_SIJAINTI}all-MiniLM-L6-v2.Q4_K_M.gguf"  # Embedding model
# GGUFMALLI = f"{SIJAINTI}gemma-3-1b-it-Q4_K_M.gguf"

#PC:

PDF_SIJAINTI = "G:/code/pdftekoaly_gguf/pdf_data/"  # Folder containing PDF files
# "G:/code/pdftekoaly_gguf/data/"
# "G:/code/pdftekoaly_gguf/pdf_data/"

SIJAINTI = "H:/tekoaly/"
MODAL_SIJAINTI = "H:/tekoaly/Embedding/"
MODALMALLI = f"{MODAL_SIJAINTI}all-MiniLM-L6-v2.Q4_K_M.gguf"  # Embedding model
GGUFMALLI = f"{SIJAINTI}gemma3-4b-it-abliterated.Q4_K_M.gguf"     # Main generation model

# Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf
# Phi-4-mini-instruct-Q4_K_M.gguf
# Phi-3.5-mini-instruct_Uncensored-Q4_K_M.gguf
# gemma3-4b-it-abliterated.Q4_K_M.gguf
# gemma-3-1b-it-Q4_K_M.gguf

# MUUTTUVAT:
# esimerkiksi

CHUNK_SIZE = 600
OVERLAP = 150

# FUNKTIOT:

################### EXTRACT TEXT FROM PDF ###################

def extract_text_from_pdf(pdf_path):

    """
    Extract text from a PDF using pdfplumber.
    """

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

################### CHUNK TEXT ###################

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

################### GET EMBEDDING ###################

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

################### COSINE SIMILARITY ###################

def cosine_similarity(vec1, vec2):

    """
    Compute cosine similarity between two vectors.
    """

    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

################### PDF PROCESSING ###################

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
    
    # KORJAA TÄMÄ. LISÄÄ SE ALEMMAS ETTÄ SAADAAN EMBEDDING JA TEKSTI .TXT TIEDOSTOON.
    txt_path = os.path.splitext(pdf_path)[0] + ".txt"
    try:
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(text)
        print(f"[DEBUG] Saved extracted text to {txt_path}")
    except Exception as e:
        print(f"[ERROR] Could not save text file {txt_path}: {e}")

    # Chunk the text
    chunks = chunk_text(text)

    # POISTA " "chunk":chunk,"
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

################### PROCESS ALL PDFS ###################

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

################### RETRIEVAL CHUNKS FUNC ###################

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

################### ANSWER QUERY ###################

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
    # Answer the question based solely on the following context. Answer and must provide and give information prof.nr of set dimension and top 3 results for the possible aluminium profile and it's profile number. Dimensional details and notations of aluminium profiles that is asked by the user.
    prompt_text = "Analyze the catalog and search the tables for all rows that contain the ordered product dimensions, \
        in any order. If you find multiple possible five digits profile numbers (prof.nr), list them all in order. \
        Do not filter any out, unless they are completely the wrong item according to the name. \
        Dimensions can appear in any order! (e.g., 25x40mm = 40x25mm). \
        Do not exclude options based on value order differences. The small and big number can be exchanged. Do not favor any profile number over another - \
        if multiple matches exist, report all of them. Do not make assumptions based on the product name - find matches using dimensions only. \
        If no match is found, report 'Not found.' Do not make guesses. \
        Reporting format: Ordered product | Suggested profile number(s) | Table | Page [Product name + dimensions] | [List of profile numbers]|[list of table titles]|[list of pages]|[Seokselle / For alloy]"
    prompt_text2 = "You are helpful and smart assistant. Answer users questions based solely on the context."
    prompt = (
        f"You are helpful and smart assistant. Answer users questions based solely on the context. The context: '{context}' \n"
    )
    # {prompt_text2} 
    print("\n\n",query)
    #question = query
    print("\n[DEBUG] Prompt for main model constructed:")
    print(prompt)
    system_message = {
        "role":"system",
        "content": f"{prompt_text2}"}
    user_message = {
        "role": "user",
        "content": f"Question:{query}, context: {context}"
    }
    try:
        # Generate answer using the main model.
        response = main_model.create_chat_completion(
            messages=[system_message,
                      user_message],
                      max_tokens=1024,
                      temperature=0.1,
                      repeat_penalty=1,
                      stream=False

        )
        # response = main_model(prompt,
        #                       max_tokens=512,
        #                       temperature=0.6,
        #                       repeat_penalty=1,
        #                       stream=False,
        #                       )
        #answer = response["choices"][0]["text"].strip()
        answer = response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        answer = f"[ERROR] Main model generation failed: {e}"
    return answer


def main():
    '''
    Main code that launches
    '''
    # STEP 1: Process PDFs and create embeddings
    # Load the embedding model (all-miniLM) using llama-cpp-python.
    # Adjust n_gpu_layers and other parameters as needed for your setup.

    print(f"[INFO] Loading embedding model from: {MODALMALLI}")
    embed_model = Llama(
        model_path=MODALMALLI,
        n_gpu_layers=-1,  # adjust this value for your setup
        verbose=False,
        embedding=True, # Loads Embedded model.
        n_ctx=512
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
        n_ctx=8192 # 32768 16384 8192 4096 2048 depends of the chunked and overlapped size and numbers.
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
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"[ERROR] An error occurred during the Q&A process: {e}")


    # Cleanup (if needed)
    del main_model
    del embed_model
    print("[INFO] Q&A session ended.")

if __name__ == "__main__":
    main()
