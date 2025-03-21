# PDF RAG Embedded koodi
  

- Esimerkki Python skripti sovelluksista, joilla yritetään kerätä dataa PDF-tiedostoista, käyttämällä eri methodeja, kuten llama-cpp-python rakennettua "embedded" toimitoa.
- Testattu myös CHUNKien optimointia.
  
  
Toistaiseksi on vain testattu yhdellä PDF tiedostolla embedding ominaisuutta ja myös testattu embed mallia: `all-MiniLM-L6-v2.Q4_K_M.gguf`. Tämän jälkeen lisätää tarkempi promptaus ja "chät" interaktio käyttäjän kanssa tekstillä ja testataan eri malleja. Tässä esimerkissä on testattu Mistral 7B v0.3 ja Llama3.1 Uncensored -GGUF malleja.
  
```
GGUF mallit:

[gemma3-4b-it-abliterated.Q4_K_M.gguf](https://huggingface.co/mlabonne/gemma-3-4b-it-abliterated-GGUF)
[Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf](https://huggingface.co/bartowski/Dolphin3.0-Llama3.2-3B-GGUF)

[nomic-embed-text-v1.5.Q8_0.gguf](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF)
[nomic-embed-text-v1.5.Q4_K_M.gguf](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF)
```
  
https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
  
https://huggingface.co/mlabonne/gemma-3-4b-it-abliterated-GGUF 
https://huggingface.co/bartowski/Dolphin3.0-Llama3.2-3B-GGUF 
    
### Testattu:

llama-cpp-python versiot: 0.3.6 - 0.3.8
nvcc versio: 12.8
NumPy versio: 1.25.* AMD / Vulkan API
NumPy versio: 2.2.4 NVidia


Seuraavassa prosessissa testataan kuinka voitaisiin eka prosessoida pdfplumberilla PDF tiedostot, etsiä point-of-intrest ja myös mahdollisesti testata OpenCV:tä tai muita OCR (kuvantunnistus) sovelluksia.
  
  
## Huomioitavaa asennuksessa:

### Conda /miniconda/ anaconda asetuksia ja säätöjä:

Muista PATH / Enviroment Variables...

Powershell: Poistamalla rajoitteita `set-executionpolicy remotesigned` että pystytään avaamaan PowerShellillä/Visual Studio Codella
esim. `conda activate gguf` automaattisesti. Pystytään helpommin hallitsemaan VENVejä ilman, että tarvitsee vaihdella CMD ja
PS välillä. Myös helpompi laittaa `$env: ...` komentoja ja asentaa Vulkan tai Cuda versio Llama.cpp.pythonista.

`conda config --set auto_activate_base false` ottaa pois automaattisen aktivoinnin kun avataan esim Powershell tietokoneella.
  
### Numpy version asennus: 
`pip install numpy==1.25.*` tai `pip install numpy==2.2.4` tai `pip install numpy`
  
### Windows Powershell terminaaliin:
  
Aktivoi venv `conda activate tekoalyllama` korvaamalla `tekoalyllama` omalla virtuaaliympäristöllä.
  
AMD tai Intel tai muu GPU: `$env:CMAKE_ARGS="-DGGML_VULKAN=on"` Windowsilla aktivoidaan Vulkan ajurien asentaminen
  
Nvidia GPU: Asennettuna Cuda Toolkit: https://developer.nvidia.com/cuda-downloads ja sen jälkeen: `$env:CMAKE_ARGS="-DGGML_CUDA=on"`

Seuraavaksi Llama-cpp-pythonin asennus:

`pip install llama-cpp-python --no-cache-dir --verbose`

Tarvittaessa käytä `--force` jos pitää overwritettää edellinen asennus.
  
  
## Mitä muutettavaa?
  
```
# Build a context prompt from the retrieved chunks.
context = "\n---\n".join(chunk["chunk"] for chunk in relevant_chunks)
    prompt = (
        "Answer the question based solely on the following context. List information of prof.nr. and other dimensional details.\n"
        "Context:\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
```
  
- Muutettava promptia ja testata erillaisia promptaus kikkoja datan laadun varmistamiseen.
  
- Tokenien määrä on suuri jopa yhdellä PDF:llä. Dynaaminen CHUNK ja OVERLAP arvot?
## Mitä lisättävää?


- Koodi `pdf_file_process_gguf_v4.py` tarvitsee "chättäys" lisäyksen, että voidaan testata embedding toimivuutta. Tarvitaan myös mahdollisesti vertailua tai mitä muissa koodeissa on testailtuja Vektori storeja esim:
  

```
def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

def build_vector_store(embedding_model, pdf_texts):
    """
    For each PDF file’s text, break it into chunks and build a vector store.
    Each entry in the store is a dict containing the text chunk, its embedding, and the source filename.
    """
```
  
- Database manuaalisesti alumiiniprofiilesita. Funktiokoodattu tietojen haku. Tätä vertaillaan mahdollisella SQL tietojen haulla ja verrataan, onko PDF tiedostoissa / sähköposteissa samoja tietoja.
