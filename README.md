# Semantic Search — Stack Overflow Questions

This project is a beginner-friendly semantic search app for Stack Overflow questions. It uses a FastAPI backend, a FAISS index, and the Hugging Face dataset `MartinElMolon/stackoverflow_preguntas_con_embeddings`.

## Features

- Loads precomputed question embeddings from Hugging Face with the `datasets` library.
- Uses FAISS `IndexFlatIP` with normalized vectors for cosine similarity.
- Converts user queries into embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Returns the top matching questions with similarity scores.
- Graceful fallback to a tiny local sample set if the remote dataset is unavailable.
- Built-in `/health` endpoint so the frontend can detect backend status cleanly.
- Self-contained frontend with a modern responsive design and improved error handling.

## Requirements

- Python 3.9+
- A virtual environment is strongly recommended

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The App

```bash
uvicorn backend.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Notes

- On first semantic startup, the app may download the dataset and sentence-transformer query model.
- The backend expects dataset rows to contain `question` and `embedding`.
- The app normalizes both stored embeddings and query embeddings so FAISS inner product behaves like cosine similarity.
- If remote loading fails, the app still starts with a small local sample set.
- If you open `frontend/index.html` directly, the page will try to auto-detect a backend at `127.0.0.1:8000` or `localhost:8000`, but serving the frontend from FastAPI is the most reliable option.
