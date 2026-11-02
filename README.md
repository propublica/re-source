# Re-Source

Semantic + keyword search for newsrooms over reader submissions.
   
To learn more about how we built this system and how we thought about safeguarding the security of our data, read more [here](http://propublica.org/reader-submissions-semantic-search-tool).

This tool imports source submissions into a local database, indexes every field of every response into
Vespa, and lets you search across all of them by keyword, by meaning, or by *who* submitted a message. 

Runs as a Streamlit app, a FastAPI service, or a batch CLI.

Re-Source stays on your machine or deployment; only search queries and short result snippets are sent to LLM and tracing providers you configure. **For this reason, we recommend a local or attestably-confidential LLM**; the default is Ollama, which can run on your laptop.

---

## How it works

Re-Source works as a four-stage pipeline:

```
CSV/XLSX --import--> DuckDB --chunk+feed--> Vespa --search+LLM--> UI / API / CLI
```

1. **Files → DuckDB.** `scripts/import_file.py` imports CSV or XLSX records into
   a local DuckDB file (`data/submissions_data.duckdb`), one table per dataset.
2. **DuckDB → Vespa.** `python -m re_source.embed_vespa` reads each row, splits it into
   **atomic chunks** — one chunk per non-empty field — and feeds them to Vespa.
3. **Embeddings live in Vespa, not Python.** Vespa computes the vectors
   server-side: **e5-large-v2** (dense/semantic), **ColBERT** (token-level
   reranking), and **BM25** (lexical) all run inside the Vespa container. The
   Python code never loads an embedding model.
4. **Search → LLM validation.** A search runs several retrieval "lanes" in
   parallel, merges them, and asks an LLM to throw out false matches that
   embeddings can't catch. See [Search strategies](#search-strategies).

## Quickstart

We recommend running the stack via Docker.

### Prerequisites

- **Docker Desktop** with at least **16 GB** allocated to it. Settings →
  Resources → Memory. Vespa loads ~2.5 GB of ONNX models into native memory.
- The [`vespa` CLI](https://docs.vespa.ai/en/vespa-cli.html) — needed once to
  deploy the Vespa application package.
- A local **Ollama** model or credentials for a compatible hosted LLM endpoint. Again, we suggest either running a local model or using a private inference provider.
  Optionally add **Langfuse** keys for tracing.


### 1. Fetch the model files

```bash
./scripts/fetch_vespa_models.sh
```

---

### 2. Configure

```bash
cp .env.example .env
```

Configure the following settings at minimum:

- **`LLM_PROVIDER`** — defaults to `local`, which means Ollama running on your
  machine. Set `hosted` to use a compatible `/chat/completions` endpoint, along
  with `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_HOSTED_MODEL`.
- **`LLM_MODEL`** — the Ollama model used by the local provider.
- **`REQUIRE_LOGIN`** — defaults to **true**, which requires an OIDC `[auth]`
  block in `.streamlit/secrets.toml`. 

---

### 3. Start the stack

```bash
docker compose up -d --build
```

Brings up the UI (:8501), API (:8000), Vespa (:8080, :19071), and the model
sidecar.

---

### 4. Deploy the Vespa application package

One time per fresh Vespa volume.

```bash
vespa config set target local
vespa deploy --wait 600 vespa/
```

---

### 5. Load data in the browser

```bash
open http://localhost:8501
```

Pick **Upload data** in the sidebar.

1. **Upload** a CSV or XLSX. You get a preview, a picker for the ID column, and
   a table name prefilled from the filename.
2. **Import to database** writes it into DuckDB.
3. **Feed to Vespa** chunks every field of every row and indexes it.
4. **Author profile enrichment** populates the search by source by calling your configured model provider.

Embeddings are computed *inside* Vespa during the feed, so the first pass over a
large table is slow.

### You should be all set to search!

Tear down with `docker compose down` (add `-v` to also drop Vespa's volumes).

---

## Search strategies

Three strategies are selectable in the UI/API (registered in
`re_source/vespa_strategies.py`):

| Strategy | What it does | When to use |
|----------|--------------|-------------|
| **Content** | Recall-first content search. Runs 4 retrieval lanes in parallel (atomic-hybrid, HyDE, query-expansion, decomposition), unions them at the row level, reranks everything on one ColBERT scale, then LLM-validates the top slice. Paginated ("Load more"). | The default — finding submissions *about* a topic. |
| **BM25** | Pure lexical/keyword search over complete parent rows (`keyword` rank profile). No dense vectors, no ColBERT, no LLM. Fast. | Exact names/terms, or a quick sanity check. |
| **Author Profile + LLM** | Searches the per-row author-profile chunks, then LLM-verifies each candidate against the full row. Tuned for find-all sweeps. | Finding submissions *from* a kind of person (e.g. "teachers", "parents"). Requires the enrichment step. |


---

## Project structure

```
re-source/
├── app.py                       # Streamlit entry (wires the two UI pages)
├── api/
│   ├── main.py                  # FastAPI app: /health, /strategies, /search
│   └── schemas.py               # Pydantic request/response models
├── re_source/
│   ├── datasets.py              # Dataset labels, record links, and ownership
│   ├── audit.py                 # Audit trail: who saw/exported which records
│   ├── chunker.py               # Atomic (per-field) chunking
│   ├── config.py                # Env/config, paths, LLM provider selection
│   ├── db_loader.py             # DuckDB rows → Document objects
│   ├── embed_vespa.py           # Feed chunks to Vespa (incremental / full / one table)
│   ├── enrich.py                # Author-profile enrichment
│   ├── export.py                # XLSX export (Summary tab + one sheet per dataset)
│   ├── llm_utils.py             # LLM provider router, validation, HyDE, expansion
│   ├── search_runner.py         # Backend-agnostic dispatch shared by UI + API
│   ├── search_types.py          # SearchResult dataclasses (dependency-free)
│   ├── search_ui.py             # Both Streamlit views (card + columns)
│   ├── vespa_client.py          # Thin pyvespa wrapper (feed/query/get/delete)
│   └── vespa_strategies.py      # The strategy registry + retrieval lanes
├── scripts/                     # Ingestion, eval, and diagnostic CLIs
├── vespa/                       # Vespa application package (schemas, services.xml)
├── tests/                       # pytest suite
├── docker-compose.yml           # Models + Vespa + Streamlit + FastAPI
└── Dockerfile                   # Multi-stage, uv-based app image
```
