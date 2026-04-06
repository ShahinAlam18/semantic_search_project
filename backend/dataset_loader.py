from __future__ import annotations

from typing import List, Tuple

import numpy as np

DATASET_NAME = "MartinElMolon/stackoverflow_preguntas_con_embeddings"

SAMPLE_ITEMS = [
    {"question": "How can I reverse a list in Python?"},
    {"question": "How do I sort a list of dictionaries in Python?"},
    {"question": "How can I remove duplicates from a Python list?"},
    {"question": "How do I merge two dictionaries in Python?"},
    {"question": "How do I read a JSON file in Python?"},
    {"question": "How do I create and activate a virtual environment in Python?"},
    {"question": "How do I install packages with pip?"},
    {"question": "How do I make an HTTP request in Python?"},
    {"question": "How do I parse command line arguments in Python?"},
    {"question": "How can I connect to MySQL using Python?"},
    {"question": "How do I write a for loop in JavaScript?"},
    {"question": "What is the difference between let, const, and var in JavaScript?"},
    {"question": "How do I make a GET request with fetch in JavaScript?"},
    {"question": "How do I make a POST request with fetch in JavaScript?"},
    {"question": "How can I remove an item from a JavaScript array?"},
    {"question": "How do I convert a string to JSON in JavaScript?"},
    {"question": "How do I debounce a function in JavaScript?"},
    {"question": "How do I use async and await in JavaScript?"},
    {"question": "How do I center a div with CSS?"},
    {"question": "How do I create a responsive layout with CSS Grid?"},
    {"question": "How do I align items horizontally with Flexbox?"},
    {"question": "How do I make an element sticky with CSS?"},
    {"question": "How can I hide overflow text with ellipsis in CSS?"},
    {"question": "How do I create a modal popup with HTML, CSS, and JavaScript?"},
    {"question": "How do I join two tables in SQL?"},
    {"question": "How do I find duplicate rows in SQL?"},
    {"question": "How do I update rows in SQL with a WHERE clause?"},
    {"question": "How can I paginate query results in SQL?"},
    {"question": "What is the difference between INNER JOIN and LEFT JOIN?"},
    {"question": "How do I create an index in PostgreSQL?"},
    {"question": "How do I clone a Git repository?"},
    {"question": "How do I create and switch branches in Git?"},
    {"question": "How do I undo the last commit in Git?"},
    {"question": "How do I resolve merge conflicts in Git?"},
    {"question": "How do I ignore files with .gitignore?"},
    {"question": "How do I push local commits to GitHub?"},
    {"question": "How do I create a Django model?"},
    {"question": "How do I run migrations in Django?"},
    {"question": "How do I create a Django superuser?"},
    {"question": "How do I handle forms in Django?"},
    {"question": "How do I create an API with Django REST Framework?"},
    {"question": "How do I use environment variables in Django?"},
    {"question": "How do I create a React component?"},
    {"question": "How do I use useState in React?"},
    {"question": "How do I fetch API data in React?"},
    {"question": "How do I pass props between React components?"},
    {"question": "How do I conditionally render content in React?"},
    {"question": "How do I create routes in React Router?"},
    {"question": "How do I send a file upload from the browser?"},
    {"question": "How do I build a login form with validation?"},
    {"question": "How do I hash passwords securely?"},
    {"question": "How do I create a REST API with FastAPI?"},
    {"question": "How do I enable CORS in FastAPI?"},
    {"question": "How do I upload files in FastAPI?"},
    {"question": "How do I test a FastAPI endpoint?"},
]


def _normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _coerce_embedding(raw_embedding) -> np.ndarray | None:
    if raw_embedding is None:
        return None

    vector = np.asarray(raw_embedding, dtype="float32")
    if vector.ndim != 1 or vector.size == 0:
        return None
    return vector


def load_stackoverflow_embeddings(limit: int = 2000) -> Tuple[List[dict], np.ndarray | None]:
    """Load question text plus precomputed embeddings from Hugging Face.

    Returns:
        items: [{"question": "..."}]
        embeddings: normalized float32 matrix with shape (n_items, dim)
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    except Exception:
        return SAMPLE_ITEMS[:limit], None

    items: List[dict] = []
    vectors: List[np.ndarray] = []
    expected_dim = None

    for row in ds:
        if len(items) >= limit:
            break

        question = (row.get("question") or "").strip()
        embedding = _coerce_embedding(row.get("embedding"))
        if not question or embedding is None:
            continue

        if expected_dim is None:
            expected_dim = int(embedding.shape[0])
        if embedding.shape[0] != expected_dim:
            continue

        items.append({"question": question})
        vectors.append(embedding)

    if not items:
        return SAMPLE_ITEMS[:limit], None

    matrix = np.vstack(vectors).astype("float32")
    matrix = _normalize_embeddings(matrix)
    return items, matrix
