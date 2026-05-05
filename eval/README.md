# Retrieval Eval

This folder evaluates ParaCite retrieval separately from the application code.
It measures whether the retriever returns the expected supporting documents for
known claims.


## Metrics

**Recall@K** measures how many relevant documents appear in the top K retrieved
documents.

If a query has one relevant document, Recall@K is 1.0 when that document appears
in the top K and 0.0 otherwise.

**MRR** measures how highly the first relevant document is ranked. A relevant
document at rank 1 scores 1.0, rank 2 scores 0.5, rank 5 scores 0.2, and no
match scores 0.0.

Basically:

- Recall@K = did we find the right paper somewhere in the first K results?
- MRR = how close to the top was the first right paper?


## Query Format

`eval_queries.jsonl` uses one query per line:

```json
{"query_id":"lm_001","query":"H-1B demand rises and falls with economic conditions.","relevant_doc_ids":["paper_005"]}
```

Use `doc_id` values from the ingested corpus.


## Run

Ingest the corpus first:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m src.app.cli ingest data/raw/labor_market/papers.json
```

Then run the eval:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 eval/retrieval_eval.py
```


Optional settings:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 eval/retrieval_eval.py --top-k 10 --ks 1,3,5,10
```

Results are written to:

```text
eval/results/metrics.json
eval/results/retrieval_details.jsonl
```

Current run was on the labor market paper set, not the CourtListener opinion
set. Keep those numbers separate in the report.
