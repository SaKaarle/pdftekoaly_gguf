# PDF RAG Embedded koodi
  

- Esimerkki Python skripti sovelluksista, joilla yritetään kerätä dataa PDF-tiedostoista, käyttämällä eri methodeja, kuten llama-cpp-python rakennettua "embedded" toimitoa.
- Testattu myös CHUNKien optimointia.
  
  
Toistaiseksi on vain testattu yhdellä PDF tiedostolla embedding ominaisuutta ja myös testattu embed mallia: `all-MiniLM-L6-v2.Q4_K_M.gguf`. Tämän jälkeen lisätää tarkempi promptaus ja "chät" interaktio käyttäjän kanssa tekstillä ja testataan eri malleja. Tässä esimerkissä on testattu Mistral 7B v0.3 ja Llama3.1 Uncensored -GGUF malleja.

Seuraavassa prosessissa testataan kuinka voitaisiin eka prosessoida pdfplumberilla PDF tiedostot, etsiä point-of-intrest ja myös mahdollisesti testata OpenCV:tä tai muita OCR (kuvantunnistus) sovelluksia.
  
  
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
    )```
  
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