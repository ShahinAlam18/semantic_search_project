# Semantic Search for Stack Overflow Questions

A production-style semantic search application that retrieves relevant Stack Overflow questions using vector similarity search, FAISS indexing, FastAPI, and a responsive frontend.

This project demonstrates practical machine learning integration in a web application: loading precomputed embeddings from Hugging Face, indexing them for fast retrieval, converting user queries into vectors, and serving ranked results through a clean API and UI.

## Project Summary

This application was built to explore how modern semantic search systems work beyond traditional keyword matching.

Key goals of the project:

- build a retrieval system using embedding vectors instead of exact text matches
- serve search results through a simple and maintainable backend API
- provide a polished frontend for interactive search
- design the system to degrade gracefully when remote resources are unavailable

## Highlights

- Semantic retrieval pipeline powered by vector embeddings
- FAISS-based nearest-neighbor search with cosine similarity
- FastAPI backend with health monitoring and structured search responses
- Hugging Face dataset integration using precomputed embeddings
- Resilient fallback mode for offline or limited-network environments
- Clean frontend with clear loading, status, and result states

## Demo Capabilities

- Accept a user query in English
- Convert the query into an embedding using Sentence Transformers
- Search the FAISS index for the top 5 most similar questions
- Return ranked matches with similarity scores
- Fall back to a local sample dataset when remote loading fails

## Tech Stack

- Python
- FastAPI
- FAISS
- Hugging Face `datasets`
- Sentence Transformers
- NumPy
- HTML, CSS, JavaScript

## Dataset

This project uses the Hugging Face dataset:

`MartinElMolon/stackoverflow_preguntas_con_embeddings`

Each record contains:

- `question`
- `embedding`

The backend loads the question text and embedding vectors, validates dimensional consistency, normalizes vectors, and stores them in a FAISS index for efficient retrieval.

## System Design

### 1. Data Loading

The backend loads question and embedding records from Hugging Face using the `datasets` library.

### 2. Vector Preparation

All stored embeddings are normalized so inner-product search behaves like cosine similarity.

### 3. Indexing

The system uses FAISS `IndexFlatIP`, which is efficient and simple for dense vector similarity search.

### 4. Query Encoding

User queries are embedded with:

`sentence-transformers/all-MiniLM-L6-v2`

### 5. Retrieval

The normalized query vector is searched against the FAISS index, and the top 5 most similar questions are returned.

## Architecture

```text
.
├── backend/
│   ├── dataset_loader.py   # loads questions and embedding vectors
│   ├── search.py           # FAISS indexing and query search logic
│   └── main.py             # FastAPI app and API routes
├── frontend/
│   └── index.html          # responsive search interface
├── requirements.txt
└── run_backend.sh          # quick local startup script
```

## API Endpoints

### `GET /health`

Returns backend readiness and search mode information.

Example:

```bash
curl http://127.0.0.1:8000/health
```

### `GET /search?q=<query>&k=5`

Returns the most relevant matching questions for a user query.

Example:

```bash
curl "http://127.0.0.1:8000/search?q=python%20list%20reverse&k=5"
```

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Project

Recommended:

```bash
./run_backend.sh
```

Manual:

```bash
uvicorn backend.main:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

## Engineering Decisions

- Used precomputed dataset embeddings to avoid recomputing document vectors at startup
- Used normalized vectors plus `IndexFlatIP` for simple cosine similarity search
- Added a fallback mode so the application remains usable without remote dataset/model access
- Kept the code beginner-friendly and modular so the data loading, indexing, and API layers are easy to understand independently

## Challenges Solved

- Handling embedding dimension consistency between stored vectors and query vectors
- Keeping the app responsive when network access is slow or unavailable
- Providing meaningful search results even in fallback mode
- Connecting machine learning search logic to a user-facing web application

## Future Improvements

- Add answer retrieval alongside question retrieval
- Add query caching for repeated searches
- Add result explanations or highlighted semantic relevance indicators
- Deploy the app with persistent model and dataset caching
- Add automated tests for retrieval quality and API responses

## Why This Project Is Valuable

This project demonstrates skills that are directly relevant to software engineering and machine learning engineering roles:

- backend API development
- vector search and retrieval systems
- practical use of machine learning models in applications
- resilient system design
- frontend and backend integration

## Resume-Friendly Description

Built a semantic search web application using FastAPI, FAISS, Hugging Face datasets, and Sentence Transformers to retrieve relevant Stack Overflow questions from embedding vectors with cosine similarity search.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
