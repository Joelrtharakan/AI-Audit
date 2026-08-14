#!/usr/bin/env bash
# Runs the golden eval against both LLM providers and diffs the two reports side by
# side -- tells you directly whether the local Ollama model's calibration is good
# enough to trust, or only good enough for fast dev iteration with a cloud-model
# sanity check before shipping prompt changes.
#
# Usage: ./scripts/eval.sh [--threshold 0.85]

set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BACKEND_DIR"

THRESHOLD="${1:-}"
THRESHOLD_ARGS=()
if [[ -n "$THRESHOLD" ]]; then
  THRESHOLD_ARGS=(--threshold "${2:-0.85}")
fi

PORT=8100
BASE_URL="http://localhost:${PORT}"
REPORTS_DIR="$(mktemp -d)"
OLLAMA_REPORT="${REPORTS_DIR}/ollama.txt"
OPENROUTER_REPORT="${REPORTS_DIR}/openrouter.txt"

SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_health() {
  for _ in $(seq 1 30); do
    if curl -s -o /dev/null "${BASE_URL}/health"; then
      return 0
    fi
    sleep 1
  done
  echo "Backend did not become healthy on ${BASE_URL} in time." >&2
  return 1
}

run_eval_with_provider() {
  local provider="$1"
  local report_file="$2"

  echo "--- Starting backend with LLM_PROVIDER=${provider} ---"
  LLM_PROVIDER="$provider" .venv/bin/uvicorn app.main:app --port "$PORT" \
    > "${REPORTS_DIR}/${provider}.server.log" 2>&1 &
  SERVER_PID=$!

  wait_for_health

  echo "--- Running golden eval against ${provider} ---"
  set +e
  .venv/bin/python scripts/run_golden_eval.py --base-url "$BASE_URL" "${THRESHOLD_ARGS[@]}" \
    > "$report_file" 2>&1
  local eval_exit=$?
  set -e
  cat "$report_file"

  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""

  return $eval_exit
}

if ! curl -s -o /dev/null http://localhost:11434; then
  echo "Ollama not reachable at http://localhost:11434 -- starting \`ollama serve\` in the background."
  ollama serve > "${REPORTS_DIR}/ollama-serve.log" 2>&1 &
  disown
  for _ in $(seq 1 20); do
    curl -s -o /dev/null http://localhost:11434 && break
    sleep 1
  done
fi

OLLAMA_EXIT=0
run_eval_with_provider "ollama" "$OLLAMA_REPORT" || OLLAMA_EXIT=$?

OPENROUTER_EXIT=0
HAVE_OPENROUTER_KEY=$(grep -E '^OPENROUTER_API_KEY=.+' .env 2>/dev/null || true)
if [[ -n "$HAVE_OPENROUTER_KEY" ]]; then
  run_eval_with_provider "openrouter" "$OPENROUTER_REPORT" || OPENROUTER_EXIT=$?
else
  echo "OPENROUTER_API_KEY not set in .env -- skipping OpenRouter comparison run."
  echo "(ollama-only)" > "$OPENROUTER_REPORT"
fi

echo ""
echo "=================================================================="
echo "SIDE-BY-SIDE: ollama (left) vs openrouter (right)"
echo "=================================================================="
diff -y -W 160 "$OLLAMA_REPORT" "$OPENROUTER_REPORT" || true

echo ""
echo "Full reports saved at: $REPORTS_DIR"

if [[ $OLLAMA_EXIT -ne 0 ]]; then
  echo "ollama run did not meet the accuracy threshold." >&2
  exit $OLLAMA_EXIT
fi
exit 0
