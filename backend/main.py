from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

try:
    from .dataset_loader import DATASET_NAME, SAMPLE_ITEMS, load_stackoverflow_embeddings
    from .search import SemanticSearcher
except ImportError:
    from dataset_loader import DATASET_NAME, SAMPLE_ITEMS, load_stackoverflow_embeddings
    from search import SemanticSearcher


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_INDEX = BASE_DIR / "frontend" / "index.html"
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="bg" x1="0%" x2="100%" y1="0%" y2="100%">
      <stop offset="0%" stop-color="#14b8a6" />
      <stop offset="100%" stop-color="#0f172a" />
    </linearGradient>
  </defs>
  <rect width="64" height="64" rx="18" fill="url(#bg)" />
  <path d="M18 24h28M18 32h18M18 40h24" stroke="#ecfeff" stroke-width="5" stroke-linecap="round" />
  <circle cx="47" cy="20" r="5" fill="#f59e0b" />
</svg>"""


class SearchResult(BaseModel):
    question: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    meta: dict


app = FastAPI(title="Local Semantic Search - StackOverflowQA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ITEMS: List[dict] = []
SEARCHER: SemanticSearcher | None = None
APP_STATE = {
    "status": "starting",
    "search_backend": "unknown",
    "dataset_source": "unknown",
    "item_count": 0,
    "startup_message": "Starting up",
}


class StartupTimeoutError(TimeoutError):
    pass


class time_limit:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.previous_handler = None

    def _handle_timeout(self, signum, frame):
        raise StartupTimeoutError(f"Startup exceeded {self.seconds}s")

    def __enter__(self):
        if self.seconds <= 0 or not hasattr(signal, "SIGALRM"):
            return self
        self.previous_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if self.previous_handler is not None:
                signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def build_search_stack():
    items, embeddings = load_stackoverflow_embeddings(limit=2000)
    questions = [it["question"] for it in items]
    searcher = SemanticSearcher(questions, embeddings=embeddings)
    searcher.build_index()
    return items, searcher


def build_fallback_stack():
    items = SAMPLE_ITEMS.copy()
    questions = [it["question"] for it in items]
    searcher = SemanticSearcher(questions, embeddings=None)
    searcher.build_index()
    return items, searcher


def initialize_search(startup_timeout: int = 5) -> None:
    global ITEMS, SEARCHER

    print("Loading dataset and building search index...")

    remote_flag = os.getenv("ENABLE_REMOTE_STARTUP")
    remote_enabled = True if remote_flag is None else remote_flag.lower() in {"1", "true", "yes"}
    startup_message = f"Loading {DATASET_NAME}"

    if remote_enabled:
        try:
            with time_limit(startup_timeout):
                ITEMS, SEARCHER = build_search_stack()
            dataset_source = "remote" if len(ITEMS) > len(SAMPLE_ITEMS) else "fallback"
            if dataset_source == "fallback":
                startup_message = "Dataset unavailable, using built-in sample questions"
            elif SEARCHER.backend != "semantic":
                startup_message = "Dataset loaded but semantic model unavailable, using keyword search"
            else:
                startup_message = f"Loaded {DATASET_NAME} with FAISS cosine search"
        except Exception as exc:
            print(f"Falling back to lightweight startup: {exc}")
            ITEMS, SEARCHER = build_fallback_stack()
            dataset_source = "fallback"
            startup_message = f"Fallback mode enabled: {exc}"
    else:
        ITEMS, SEARCHER = build_fallback_stack()
        dataset_source = "fallback"
        startup_message = "Offline-first startup using built-in sample questions"

    APP_STATE.update(
        {
            "status": "ready",
            "search_backend": SEARCHER.backend,
            "dataset_source": dataset_source,
            "item_count": len(ITEMS),
            "startup_message": startup_message,
        }
    )

    print(
        f"App ready with {len(ITEMS)} items using "
        f"{SEARCHER.backend} search and {dataset_source} dataset source. "
        f"{startup_message}"
    )


@app.on_event("startup")
def startup_event():
    initialize_search()


@app.get("/")
def root():
    return FileResponse(FRONTEND_INDEX)


@app.get("/favicon.ico")
def favicon():
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/health")
def health():
    return APP_STATE


@app.get("/search", response_model=SearchResponse)
def search(q: str, k: int = 5):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    if SEARCHER is None:
        raise HTTPException(status_code=503, detail="Search engine is still starting up")

    hits = SEARCHER.search(q, top_k=k)
    results = []
    for idx, score in hits:
        if idx < 0 or idx >= len(ITEMS):
            continue
        item = ITEMS[idx]
        results.append({"question": item["question"], "score": round(score, 4)})

    return {
        "results": results,
        "meta": {
            "query": q,
            "count": len(results),
            "search_backend": APP_STATE["search_backend"],
            "dataset_source": APP_STATE["dataset_source"],
        },
    }


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
