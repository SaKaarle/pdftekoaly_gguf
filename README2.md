![Älykkäät teknologiat](https://storage.googleapis.com/seamk-production/2022/04/2b1d63e0-alykkaat-teknologiat_highres_2022-768x336.jpg)

# PDF RAG Tekoälykoodi
  
Paikallisten tekoälymallien käyttöönotto ja PDF tiedostojen käsittelyä Embed -mallien avulla.
Python -koodi on kirjoitettu toimimaan mahdollisimman monella eri tietokone kokoonpanoilla.

Sovelluksia joita käytetään Windows tietokoneella:
- Python 3.9 - 3.13
- Visual Studio Code
- Miniconda (venv)
- Uv (venv)
- CMAKE
- Git
- AMD / Intel GPU: Vulkan SDK 
- Nvidia GPU: Cuda SDK
- llama-cpp-python
  
# Julkaisun historiatiedot
Merkittävät muutokset julkaisuun
  
|pvm|Muutokset|Tekijä|
|---|---|---|
|21.06.2025|Versio 1.0 julkaisu|Saku Kaarlejärvi|
|xx.xx.2025|Zenodo julkaisu|Saku Kaarlejärvi|
  
# Sisällysluettelo
- [Julkaisun nimi](#pdf-rag-tekoälykoodi)
- [Julkaisun historiatiedot](#julkaisun-historiatiedot)
- [Sisällysluettelo](#sisällysluettelo)
- [Teknologiapilotti](#teknologiapilotti)
- [Hanketiedot](#hanketiedot)
- [Kuvaus](#kuvaus)
- [Tavoitteet](#tavoitteet)
- [Toimenpiteet](#toimenpiteet)
- [Asennus ja käyttö](#asennus-ja-käyttö)
- [Python ohjelman käyttö](#Python-ohjelman-käyttö)
- [Havaitut virheet ja ongelmatilanteet](#HAVAITUT-VIRHEET-JA-ONGELMATILANTEET)
- [Vaatimukset](#laserleikkurin-datan-keruu-ja-visualisointi)
- [Tulokset](#tulokset)
- [Lisenssi](#lisenssi)
- [Tekijät](#tekijät)
  
  
# Teknologiapilotti
vAI:lla tuottavuutta?-hankkeen työpaketti 1. valmistetaan teknologiademoja, joiden avulla tekoälyn mahdollisuuksia voidaan havainnollistaa osallistuville yrityksille.
  
# Hanketiedot
- Hankkeen nimi: vAI:lla tuottavuutta?
- Rahoittaja: Euroopan unionin osarahoittama. Euroopan aluekehitysrahasto (EAKR). Etelä-Pohjanmaan liitto.
- Hankkeen toteuttajat: Päätoteuttajana Seinäjoen Ammattikorkeakoulu Oy, osatoteuttajina Tampereen korkeakoulusäätiö sr ja Vaasan Yliopisto
- Aikataulu: 1.8.2024 – 31.12.2026
  
# Kuvaus
Python koodi jolla pystytään suorittamaan GGUF tekoälymalleja llama-cpp-pythonin avulla. Koodi lataa GGUF -tekoälymallin ja myös GGUF-embed tekoälymallin, joka käsittelee tekstin ja muodostaa siitä vektorikartan. Tämän avulla pystytään tekoälyn kanssa viestittelemään käsiteltyjen dokumenttitiedostojen kanssa.
  
# Tavoitteet
Pilotissa kehitettiin teknologiademo, jolla pystytään näyttämään yrityksille tekoälyn mahdollisuuksia käyttäen tekoälymalleja paikallisesti.
  
# Toimenpiteet
Teknologiademo ....
  
# Asennus ja käyttö
  
## Vaatimukset
Lista vaadittavista laitteista:
- Tietokone, jossa on vähintään 16GB keskusmuistia tai 8GB videomuistia
- Vähintään 10 - 20GB vapaata tilaa tekoälymalleja varten

## Vaadittavat asennukset Windows sekä Linux -tietokoneelle
Lista sovelluksista:
- Python 3.9 - 3.13
- Visual Studio Code
- Miniconda (venv)
- Uv (venv)
- CMAKE
- Git
- AMD / Intel GPU: [Vulkan SDK ](https://vulkan.lunarg.com/sdk/home)
- Nvidia GPU: [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
- llama-cpp-python
  
## Pip paketit
  
`pip install pdfplumber faiss-cpu aiofiles tqdm`

## Tekoälymallit

GGUF mallit:
  
- [gemma3-4b-it-abliterated.Q4_K_M.gguf](https://huggingface.co/mlabonne/gemma-3-4b-it-abliterated-GGUF)
- [Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf](https://huggingface.co/bartowski/Dolphin3.0-Llama3.2-3B-GGUF)
- [Reasoning / Thinking malli: Qwen3-1.7B-abliterated-iq4_nl.gguf](https://huggingface.co/Mungert/Qwen3-1.7B-abliterated-GGUF)
  
Embedding mallit:
  
- [all-MiniLM-L6-v2](https://huggingface.co/leliuga/all-MiniLM-L6-v2-GGUF)
- [nomic-embed-text-v1.5.Q8_0.gguf](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF)
- [nomic-embed-text-v1.5.Q4_K_M.gguf](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF)
- [nomic-embed-text-v2-moe.Q8_0.gguf](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF)


## Conda /miniconda/ anaconda asetuksia ja säätöjä:
  
Muista PATH / Enviroment Variables...
  
Powershell: Poistamalla rajoitteita `set-executionpolicy remotesigned` että pystytään avaamaan PowerShellillä/Visual Studio Codella
esim. `conda activate gguf` automaattisesti. Pystytään helpommin hallitsemaan VENVejä ilman, että tarvitsee vaihdella CMD ja
PS välillä. Myös helpompi laittaa `$env: ...` komentoja ja asentaa Vulkan tai Cuda versio Llama.cpp.pythonista.
  
`conda config --set auto_activate_base false` ottaa pois automaattisen aktivoinnin kun avataan esim Powershell tietokoneella.
  
## Python UV venv
  
Yksinkertainen asentaa noudattaen Astral-sh uv [GitHub repositoryä](https://github.com/astral-sh/uv/).
  
`CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install llama-cpp-python==0.3.9 --verbose --reinstall --no-cache-dir`
  
## Windows Powershell terminaaliin:
  
Aktivoi venv `conda activate tekoalyllama` korvaamalla `tekoalyllama` omalla virtuaaliympäristöllä.
  
AMD tai Intel tai muu GPU: `$env:CMAKE_ARGS="-DGGML_VULKAN=on"` Windowsilla aktivoidaan Vulkan ajurien asentaminen.
  
Nvidia GPU: Täytyy olla asennettuna Cuda Toolkit: https://developer.nvidia.com/cuda-downloads ja sen jälkeen: `$env:CMAKE_ARGS="-DGGML_CUDA=on"`
  
## Llama-cpp-python asennus
CMAKE täytyy olla asennettuna, että pystytään asentamaan Llama-cpp-python tietokoneelle. [CMAKE CPP compiler](https://visualstudio.microsoft.com/vs/features/cplusplus/) ja [visuaaliset ohjeet](https://code.visualstudio.com/docs/cpp/config-msvc#_prerequisites) Compilerin asennukselle.

Linux tietokoneilla täytyy varmistaa, että CMAKE on asennettu kirjoittamalla `cmake --version` terminaaliin. 
  
Llama-cpp-python asennetaan venv ympäristöön. Luomalla oman venv ympäristön ja aktivoimalla sen, pystytään asentamaan virtuaaliympäristöön llama-cpp-python käyttäen PIP-pakettien hallintaa.
  
Seuraavaksi täytyy laittaa ympäristömuuttuja muutokset Powershellille ja asentaa llama-cpp-python luotuun venv ympäristöön.
AMD/ROCM: `$env:CMAKE_ARGS="-DGGML_VULKAN=on"` tai `$env:CMAKE_ARGS="-DGGML_HIPBLAS=on"`
NVIDIA: `$env:CMAKE_ARGS="-DGGML_CUDA=on"`
CPU: `$env:CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"` tai pelkästään `pip install llama-cpp-python`
  
Kun on komento syötetty Powershelliin, voidaan aloittaa llama-cpp-pythonin asennus. `pip install llama-cpp-python --verbose`. Jos pitää asentaa uudelleen, lisää komento ` --force --no-cache-dir`.
  
# Tulokset
Tulossa pian...
  
# Lisenssi
Dokumentit lisensoitu:
- [![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
  
# Tekijät

Saku Kaarlejärvi