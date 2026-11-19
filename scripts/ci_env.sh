#!/usr/bin/env bash
# Python environment on the self-hosted runner, shared between runs and rebuilt only when requirements.txt changes.
set -euo pipefail
BASE="${MLPIPE_HOME:-$HOME/mlops-ct-demo}"
VENV="$BASE/venv-$(shasum -a 256 requirements.txt | cut -c1-12)"
if [ ! -x "$VENV/bin/python" ]; then
  PYBIN="$(command -v python3.12 || command -v python3 || echo /opt/homebrew/bin/python3)"
  mkdir -p "$BASE"
  "$PYBIN" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r requirements.txt
fi
echo "PY=$VENV/bin/python" >> "${GITHUB_ENV:-/dev/null}"
"$VENV/bin/python" --version
