#!/usr/bin/env zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
PIP_BIN="$ROOT_DIR/.venv/bin/pip"

if ! "$PYTHON_BIN" -c "import fastapi, uvicorn, pydantic, numpy" >/dev/null 2>&1; then
  "$PIP_BIN" install -r requirements.txt
fi

exec "$PYTHON_BIN" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level info
