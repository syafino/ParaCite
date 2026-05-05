# ParaCite

CS410

ParaCite is an auto-citation assistant designed to help academic writers find and format citations efficiently. By leveraging advanced retrieval methods like BM25 and vector-based semantic search, ParaCite suggests relevant papers from a curated knowledge base and provides properly formatted citations.

## Features
- **Hybrid Retrieval**: Combines keyword-based and vector-based search for accurate results.
- **Metadata Indexing**: Filters results by year, venue, author, and more.
- **Formatted Citations**: Generates citations in standard formats like BibTeX, APA, and IEEE.
- **User-Friendly Interface**: Lightweight web or command-line interface for easy use.

## How It Works
1. **Ingestion**: Processes research papers, extracting text and metadata.
2. **Indexing**: Builds a hybrid index for fast and accurate retrieval.
3. **Retrieval**: Matches user queries to relevant papers using hybrid search.
4. **Output**: Returns top suggestions with formatted citations.

## Setup

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `.env` with your API keys:

```bash
COURTLISTENER_TOKEN=your_courtlistener_token
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key
LLM_MODEL=gpt-4o
```

The commands below use offline Hugging Face cache variables so local embeddings do not try to download during indexing or retrieval.

macOS/Linux:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.app.cli ask "example query"
```

Windows PowerShell:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
py -m src.app.cli ask "example query"
```

## Run Ingestion First

### Option 1: CourtListener legal opinions

Fetch raw CourtListener JSON files:

```bash
python3 -m src.ingest.fetch_courtlistener --query "Fourth Amendment" --court scotus --max-pages 2
```

Normalize the raw JSON into a metadata catalog:

```bash
python3 -m src.ingest.collect_metadata
```

Chunk the catalog into retrieval-ready text chunks:

```bash
python3 -m src.ingest.chunk_catalog
```

Build the vector index:

macOS/Linux:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.index.vector_store --chunks data/processed/text/chunks.jsonl --embedder sentence-transformer --output-dir data/indexes/vectors
```

Windows PowerShell:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
py -m src.index.vector_store --chunks data/processed/text/chunks.jsonl --embedder sentence-transformer --output-dir data/indexes/vectors
```

Build the BM25 index:

```bash
python3 -m src.index.build_bm25 --chunks data/processed/text/chunks.jsonl
```

### Option 2: Local papers or uploads

Ingest the labor market paper corpus:

macOS/Linux:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.app.cli ingest data/raw/labor_market/papers.json
```

Windows PowerShell:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
py -m src.app.cli ingest data/raw/labor_market/papers.json
```

You can also ingest a PDF, text, Markdown, or JSON file:

macOS/Linux:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.app.cli ingest path/to/file.pdf
```

Windows PowerShell:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
py -m src.app.cli ingest path/to/file.pdf
```

## Start Backend

macOS/Linux:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.app.api
```

Windows PowerShell:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
py -m src.app.api
```

Backend URL: `http://127.0.0.1:8000`

## Start Frontend

macOS/Linux:

```bash
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 streamlit run src/retrieve/app/streamlit_app.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
streamlit run src/retrieve/app/streamlit_app.py
```

Frontend URL: `http://localhost:8501`

## Ask From CLI

macOS/Linux:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.app.cli ask "H-1B demand rises and falls with economic conditions." --top-k 3 --style apa
```

Windows PowerShell:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
py -m src.app.cli ask "H-1B demand rises and falls with economic conditions." --top-k 3 --style apa
```
