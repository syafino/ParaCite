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

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your API keys:

```bash
COURTLISTENER_TOKEN=your_courtlistener_token
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key
LLM_MODEL=gpt-4o
```

## Run Ingestion First

Ingest the labor market paper corpus:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.app.cli ingest data/raw/labor_market/papers.json
```

You can also ingest a PDF, text, Markdown, or JSON file:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.app.cli ingest path/to/file.pdf
```

## Start Backend

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.app.api
```

Backend URL: `http://127.0.0.1:8000`

## Start Frontend

```bash
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 streamlit run src/retrieve/app/streamlit_app.py
```

Frontend URL: `http://localhost:8501`

## Ask From CLI

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.app.cli ask "H-1B demand rises and falls with economic conditions." --top-k 3 --style apa
```
