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
  
AMD/ROCM: `$env:CMAKE_ARGS="-DGGML_VULKAN=on"` tai `$env:CMAKE_ARGS="-DGGML_HIPBLAS=on"`
NVIDIA: `$env:CMAKE_ARGS="-DGGML_CUDA=on"`
CPU: ´$env:CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"´ tai pelkästään `pip install llama-cpp-python`
  
  
  
`pip install llama-cpp-python  --no-cache-dir --verbose` ja itse asennus Llama-cpp-pythonille.` --force ` jos pitää overwritettää asennus.
  
`pip show llama-cpp-python`: Testattu ja toimii versiolla 0.3.9

uv venv: `CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install llama-cpp-python==0.3.9 --verbose --reinstall --no-cache-dir --verbose` 
```
DEBUG uv 0.7.4
DEBUG Searching for default Python interpreter in virtual environments
DEBUG Found `cpython-3.13.3-linux-x86_64-gnu` at `/home/saku/VSC/pdftekoaly_gguf/pdftekoaly_gguf/.venv/bin/python3` 
```   

'''

import os
import glob
# import json 
import re
import numpy as np
import pdfplumber
from llama_cpp import Llama
from tqdm import tqdm
import time
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import faiss

# KANSIOT JA SIJAINNIT:

# Gemma3 4B it abliterated Q4_K_M + Nomic Embed Text V2 = 4387 MB

FILE_LOCATION = 4
# 1: PC, 2: LINUX, 3: Laptop

match (int(FILE_LOCATION)):
    case 1:
        # PC:
        PDF_SIJAINTI =  "G:/Dokumentit/Satunnainen/kuulokkeet/"
        EMBEDTALLENNUS = "./embeddings/"
        SIJAINTI = "F:/Tekoaly/GGUF/"
        MODAL_SIJAINTI = "F:/Tekoaly/Embedding/"
        MODALMALLI = f"{MODAL_SIJAINTI}nomic-embed-text-v2-moe.Q8_0.gguf"  # Embedding model
        GGUFMALLI = f"{SIJAINTI}gemma3-4b-it-abliterated.Q4_K_M.gguf"     # Main generation model
    case 2:
        # LINUX:
        # /media/hdd2tb/Tekoaly/GGUFlmstudio/second-state/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf
        # /media/hdd2tb/Tekoaly/GGUF/gemma3-4b-it-abliterated.Q4_K_M.gguf
        # /media/hdd2tb/Tekoaly/GGUFlmstudio/mradermacher/gemma3-4b-it-abliterated-GGUF/gemma3-4b-it-abliterated.Q4_K_M.gguf
        PDF_SIJAINTI = "/media/hdd4tb/Dokumentit/Satunnainen/kuulokkeet/"
        EMBEDTALLENNUS = "./embeddings/"
        SIJAINTI = "/media/hdd2tb/Tekoaly/GGUFlmstudio/mradermacher/gemma3-4b-it-abliterated-GGUF/"
        MODAL_SIJAINTI = "/media/hdd2tb/Tekoaly/Embedding/"
        MODALMALLI = f"{MODAL_SIJAINTI}nomic-embed-text-v2-moe.Q8_0.gguf"  # Embedding model
        GGUFMALLI = f"{SIJAINTI}gemma3-4b-it-abliterated.Q4_K_M.gguf"
    case 3:
        # Laptop:
        # C:\Users\Saku-Laptop\.lmstudio\models\
        PDF_SIJAINTI = "C:/Users/Saku-Laptop/Documents/PDF/"
        EMBEDTALLENNUS = "./embeddings/"
        SIJAINTI = "C:/Users/Saku-Laptop/.lmstudio/models/bartowski/mlabonne_gemma-3-4b-it-abliterated-GGUF/"
        MODAL_SIJAINTI = "C:/Users/Saku-Laptop/.lmstudio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/"
        MODALMALLI = f"{MODAL_SIJAINTI}nomic-embed-text-v2-moe.Q8_0.gguf"  # Embedding model
        GGUFMALLI = f"{SIJAINTI}mlabonne_gemma-3-4b-it-abliterated-IQ3_M.gguf"
        
    case 4:
        # Linux Laptop:
        # C:\Users\Saku-Laptop\.lmstudio\models\
        PDF_SIJAINTI = "/media/winos/Users/Saku-Laptop/Documents/PDF/"
        EMBEDTALLENNUS = "./embeddings/"
        SIJAINTI = "/media/winos/Users/Saku-Laptop/.lmstudio/models/bartowski/mlabonne_gemma-3-4b-it-abliterated-GGUF/"
        MODAL_SIJAINTI = "/media/winos/Users/Saku-Laptop/.lmstudio/models/nomic-ai/nomic-embed-text-v2-moe-GGUF/"
        MODALMALLI = f"{MODAL_SIJAINTI}nomic-embed-text-v2-moe.Q8_0.gguf"  # Embedding model
        GGUFMALLI = f"{SIJAINTI}mlabonne_gemma-3-4b-it-abliterated-IQ3_M.gguf"
    case _:
        print(f"[ERROR]: Invalid file location. Check 'FILE_LOCATION' value. Value is: {FILE_LOCATION} \n")   
# end match

# Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf
# Phi-4-mini-instruct-Q4_K_M.gguf
# Phi-3.5-mini-instruct_Uncensored-Q4_K_M.gguf
# gemma3-4b-it-abliterated.Q4_K_M.gguf
# gemma-3-1b-it-Q4_K_M.gguf
# Qwen_Qwen3-1.7B-IQ4_NL.gguf

# nomic-embed-text-v1.5.Q8_0.gguf
# nomic-embed-text-v1.5.Q4_K_M.gguf
# all-MiniLM-L6-v2.Q4_K_M.gguf
# nomic-embed-text-v2-moe.Q8_0.gguf

# MUUTTUVAT:
# esimerkiksi

CHUNK_SIZE = 350
OVERLAP = 150

# TOKENien määrä jota vaihtaa
# 32768 16384 8192 4096 2048 depends of the chunked and overlapped size and numbers.
MODALMAXTOKEN = 512 # 2048 tai 512
MAXVIESTITOKEN = 8192 # Riippuu paljon RAG tokeneita se ottaa vastaan.
VASTAUSTOKEN = 1024 #vastauksen pituus määrittyy tällä.


# FUNKTIOT:

async def main():

    '''
    Main code that launches

    Embedding, main model and Q&A logic.

    1. Ladataan muistiin malli Multimodal / Embedding malli. Käy monet muut mallit.
    2. Kysytään käyttäjältä, halutaanko prosessoida EMBEDDING tiedostot?
    3. Prosessoidaan Embedding mallilla PDF tiedostojen tekstit.
    4. Q&A loop

    '''

    # STEP 1: Process PDFs and create embeddings
    # Load the embedding model (all-miniLM) using llama-cpp-python.
    # Adjust n_gpu_layers and other parameters as needed for your setup.

    # EMBEDDING MALLI JA SEN SÄÄTÖ:
    print(f"[INFO] Loading embedding model from: {MODALMALLI}")
    embed_model = Llama(
        model_path=MODALMALLI,
        n_gpu_layers=-1,  # adjust this value for your setup
        verbose=False,
        embedding=True, # Loads Embedded model.
        n_ctx=MODALMAXTOKEN
    )
    # ASK the user if they want to process PDF files or use existing vector maps from JSON file:
    use_existing_npy = input("Do you want to use existing '.npy' files with embeddings? (y/n): ")
    if use_existing_npy.lower() == 'y':
        # Load embeddings from the JSON file
        all_embeddings = load_embeddings_from_npy(EMBEDTALLENNUS)
        print(f"[INFO] Loaded {len(all_embeddings)} embeddings from NPY files.")
        # Check if there are any embeddings loaded
        if all_embeddings:
            print(f"[DEBUG] First embedding: {all_embeddings[0]}")
    else:
        start_time = time.time()
        all_embeddings = await process_all_pdfs_multithreaded(PDF_SIJAINTI, embed_model)
        end_time = time.time()
        print(f"[INFO] Total processing time: {end_time - start_time:.2f} seconds")

    # STEP 2: Q&A using Retrieval-Augmented Generation (RAG)
    # Load the main generation model (GGUFMALLI)
    # JOS KONTEKSTI TAI RAG ON LIIAN SUURI, SÄÄDÄ "n_ctx" ARVOJA

    print(f"\n[INFO] Loading main model from: {GGUFMALLI}")
    main_model = Llama(
        model_path=GGUFMALLI,
        n_gpu_layers=-1,  # adjust as needed for your setup
        verbose=False,
        n_ctx=MAXVIESTITOKEN # 32768 16384 8192 4096 2048 depends of the chunked and overlapped size and numbers.
    )

    # Q&A loop
    print("\n[INFO] Entering Q&A loop. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.strip().lower() == "exit":
            break
        try:
            answer = answer_query(query, main_model, embed_model, all_embeddings)
            print(f"\nQuestion: '{query}' \n")
            print(f"\nAnswer:\n{answer}")
            
        except FileNotFoundError as file_e:
            print(f"[ERROR] File not found: {file_e}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")
            
    # Cleanup (if needed)
    del main_model
    del embed_model
    print("[INFO] Q&A session ended.")

################### EXTRACT TEXT FROM PDF ###################

async def extract_text_from_pdf(pdf_path):

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

async def process_pdf_file(pdf_path, embed_model):

    """
    Process a single PDF:
      - Extract text (and save to .txt for debugging)
      - Split text into chunks
      - Generate an embedding for each chunk
      - Save the embeddings to a .npy file.
      - Save the embeddings to a .txt file for debugging.
    """

    print(f"\n[INFO] Processing PDF: {pdf_path}")
    text = await extract_text_from_pdf(pdf_path)

    # Save raw text for debugging
    txt_path = os.path.splitext(pdf_path)[0] + "_raw.txt"
    try:
        async with aiofiles.open(txt_path, "w", encoding="utf-8") as f_txt:
            await f_txt.write(text)
        print(f"[DEBUG] Saved raw text to {txt_path}")
    except Exception as e:
        print(f"[ERROR] Could not save raw text to {txt_path}: {e}")

    # Chunk the text
    chunks = chunk_text(text)

    # Testiä, kuinka ne erotetaan toisistaan, embedding ja chunkit. "embedded_data_debug" on txt tiedostoa varten.
    embedded_data = []
    
    for idx, chunk in enumerate(chunks):
        print(f"[INFO] Embedding chunk {idx + 1}/{len(chunks)}...")
        embedding = get_embedding(chunk, embed_model)
        if embedding is not None:
            embedded_data.append({"chunk": chunk, "embedding": embedding})
        else:
            print(f"[WARNING] Embedding failed for chunk {idx + 1}/{len(chunks)}")

    # Save embeddings for this PDF to .npy
    embedding_file_npy = os.path.splitext(os.path.basename(pdf_path))[0] + "_embeddings.npy"
    save_embeddings_to_npy(embedded_data, embedding_file_npy, EMBEDTALLENNUS)

    # Save embeddings to a .txt file for debugging
    embedding_file_txt = os.path.splitext(os.path.basename(pdf_path))[0] + "_embeddings.txt"
    save_embeddings_to_txt(embedded_data, embedding_file_txt, EMBEDTALLENNUS)

    return embedded_data

################### SAVE TO EMBEDDING VECTORMAP TO NPY ###################

def save_embeddings_to_npy(embeddings, file_path, embed_directory):
    """
    Save embeddings to a .npy file.
    """
    try:
        full_file_path = os.path.join(embed_directory,file_path)
        with open(full_file_path, "wb") as embed_f:
            np.save(embed_f, embeddings)
            print(f"[INFO] Saved embeddings to {embed_f}")
    except Exception as e:
        print(f"[ERROR] Could not save embeddings to {embed_f}: {e}")

################### SAVE TO EMBEDDING VECTORMAP TO TXT ###################

def save_embeddings_to_txt(embeddings, file_path, embed_directory):
    """
    Save embeddings to a .txt file for debugging.
    """
    try:
        full_file_path = os.path.join(embed_directory,file_path)
        with open(full_file_path, "w", encoding="utf-8") as txt_f:
            for embedding in embeddings:
                txt_f.write(str(f"{embedding}\n"))
                print(f"[INFO] Saved embeddings to {embed_directory} for debugging")
    except Exception as e:
        print(f"[ERROR] Could not save embeddings to {embed_directory}: {e}")

################### PROCESS ALL PDFS ###################

async def process_all_pdfs_multithreaded(pdf_folder, embed_model):
    """
    Process all PDF files in the specified folder using multithreading.
    Returns a list of all chunk embeddings across PDFs.
    """
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    print(f"[INFO] Found {len(pdf_files)} PDF file(s) in {pdf_folder}.")
    all_embeddings = []

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as thread_pool:
        tasks = [
            loop.run_in_executor(thread_pool, asyncio.run, process_pdf_file(pdf, embed_model))
            for pdf in pdf_files
        ]
        for future in tqdm(asyncio.as_completed(tasks), total=len(pdf_files), desc="Processing PDFs"):
            try:
                embeddings = await future
                all_embeddings.extend(embeddings)
            except Exception as e:
                print(f"[ERROR] Failed to process PDF: {e}")

    print(f"[INFO] Processed all PDFs; total chunks embedded: {len(all_embeddings)}")
    return all_embeddings

################### RETRIEVAL CHUNKS FUNC ###################

def retrieve_relevant_chunks(query, embed_model, all_embeddings, top_k=3):

    """
    Compute the query's embedding and return the top_k chunks
    from all_embeddings with the highest cosine similarity.
    Includes detailed debugging to show the similarity scores.
    
    Uses FAISS and not Cosine_Similarity
    """

    print("\n[INFO] Computing query embedding...")
    query_embedding = get_embedding(query, embed_model)
    if query_embedding is None:
        return []

    # Cosine_Similarity:

    # scored = []
    # print("[DEBUG] Comparing query embedding with stored embeddings:")
    # # Compare each stored embedding with the query embedding.
    # for idx, item in enumerate(all_embeddings):
    #     score = cosine_similarity(query_embedding, item["embedding"])
    #     scored.append((score, item))
    #     # Show the similarity score for each chunk (showing first 100 chars of text for context)
    #     snippet = item["chunk"][:100].replace("\n", " ")
    #     print(f"  [DEBUG] Chunk {idx}: score = {score:.4f} | text snippet: {snippet}...")

    # # Sort the scored list by similarity score (highest first).
    # scored.sort(key=lambda x: x[0], reverse=True)
    # top_chunks = [item for score, item in scored[:top_k]]

    # print(f"[DEBUG] Selected top {top_k} chunks:")
    # for rank, (score, item) in enumerate(scored[:top_k], start=1):
    #     snippet = item["chunk"][:100].replace("\n", " ")
    #     print(f"  [DEBUG] Rank {rank}: score = {score:.4f} | text snippet: {snippet}...")
    # return top_chunks

    # FAISS index

    # Normalize the query embedding
    query_embedding = np.array(query_embedding).astype('float32')
    query_embedding /= np.linalg.norm(query_embedding)

    # FAISS index
    dimension = len(query_embedding)
    index = faiss.IndexFlatIP(dimension)  # Use Inner Product for cosine similarity

    embeddings = np.array([item["embedding"] for item in all_embeddings]).astype('float32')

    # Normalize the embeddings before indexing
    faiss.normalize_L2(embeddings)

    index.add(embeddings)
    
    # Search for the top_k most similar embeddings
    distances, indices = index.search(np.array([query_embedding]), top_k)
    top_chunks = [all_embeddings[idx] for idx in indices[0]]

    print(f"[DEBUG] Selected top {top_k} chunks:")
    for rank, (distance, item) in enumerate(zip(distances[0], top_chunks), start=1):
        snippet = item["chunk"][:100].replace("\n", " ")
        print(f"  [DEBUG] Rank {rank}: distance = {distance:.4f}, text snippet: {snippet}...")

    return top_chunks

################### NPY FILE USAGE ###################

def load_embeddings_from_npy(folder):
    """
    Load embeddings from .npy files in the specified folder.
    Returns a list of dictionaries, each containing the 'chunk' and 'embedding' keys.
    """
    npy_files = glob.glob(os.path.join(folder, "*.npy"))
    print(f"[INFO] Found {len(npy_files)} .npy file(s) in {folder}.")
    all_embeddings = []
    for npy_file in npy_files:
        try:
            embeddings = np.load(npy_file, allow_pickle=True)
            for embedding in embeddings:
                all_embeddings.append({"chunk": embedding["chunk"], "embedding": embedding["embedding"]})
            print(f"[INFO] Loaded embeddings from {npy_file}")
        except Exception as e:
            print(f"[ERROR] Could not load embeddings from {npy_file}: {e}")
    return all_embeddings

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
    prompt_text = "Answer the question based solely on the following context. Analyze the catalog and search the tables for all rows that contain the ordered product dimensions, \
        in any order. If you find multiple possible five digits profile numbers (prof.nr), list them all in order. \
        Do not filter any out, unless they are completely the wrong item according to the name. \
        Dimensions can appear in any order! (e.g., 25x40mm = 40x25mm). \
        Do not exclude options based on value order differences. The small and big number can be exchanged. Do not favor any profile number over another - \
        if multiple matches exist, report all of them. Do not make assumptions based on the product name - find matches using dimensions only. \
        If no match is found, report 'Not found.' Do not make guesses. \
        Reporting format: Ordered product | Suggested profile number(s) | Table | Page [Product name + dimensions] | [List of profile numbers] | [list of table titles] | [list of pages] | [Seokselle / For alloy]"
    prompt_text2 = "You are helpful and smart assistant. Answer users questions based solely on the context." #Answer in Finnish. Kuinka korjaan HD681 kuulokkeet? Mitä tarvitsen?
    prompt = (
        f"You are helpful and smart assistant. Answer users questions based solely on the context. The context: '{context}' \n"
    )

    system_message = {
        "role":"system",
        "content": f"{prompt_text2}"}
    user_message = {
        "role": "user",
        "content": f"Question:{query}, context: {context}"
    }
    #print(user_message)
    try:
        # Generate answer using the main model.
        response = main_model.create_chat_completion(
            messages=[system_message,
                      user_message],
                      max_tokens=VASTAUSTOKEN,
                      temperature=0.1,
                      repeat_penalty=1,
                      stream=False

        )

        answer = response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        answer = f"[ERROR] Main model generation failed: {e}"
    return answer

if __name__ == "__main__":
    asyncio.run(main())
