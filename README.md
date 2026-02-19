# Interact Vector Embedding

Vector database and **emotion classification API** using [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base) (sentence-transformers) and ChromaDB. Supports Chinese and English.

## Setup (uv)

```bash
uv sync
```

First run will download the M3E model from Hugging Face.

## Emotion classification

Zero-shot emotion classification + keyword extraction via embedding cosine similarity.

```bash
uv run uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

- **POST /classify** — body: `{"text": "今天天气真好，周末去旅行放松一下", "top_keywords": 5}` → `{"emotion": "开心", "score": 0.82, "keywords": ["周末", "旅行", "放松", ...]}`
- **GET /health** — encoder and category cache status

Config via env: `EMBEDDING_MODEL`, `KEYWORDS_TOP_K`, `CLASSIFY_TIMEOUT`, `EXECUTOR_WORKERS`. See `config.py`.

## Vector store (ChromaDB)

```bash
uv run python main.py
```

```python
from main import get_vector_store

store = get_vector_store()  # persist under ./data/vector_db by default
store.add(["your document 1", "document 2"], ids=["id1", "id2"])
results = store.search("query text", n_results=5)
# results["documents"], results["ids"], results["distances"]
```
