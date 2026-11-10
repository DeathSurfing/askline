#!/usr/bin/env sh
set -eu

# One command to build and run Askline with Docker or Podman.
# Usage: ./run.sh [up|down|logs|build|open]

cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. The bundled model runs in compose." >&2
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
elif podman compose version >/dev/null 2>&1; then
  COMPOSE="podman compose"
elif command -v podman-compose >/dev/null 2>&1; then
  COMPOSE="podman-compose"
else
  echo "No compose provider found. Install Docker Compose or the podman compose provider." >&2
  exit 1
fi

env_value() {
  grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2 || true
}

COMPOSE_FILES="-f compose.yaml"
if [ -e /dev/nvidia0 ]; then
  for dir in "$(env_value NVIDIA_LIBS)" /usr/lib/x86_64-linux-gnu /usr/lib64 /opt/nvidia-libs; do
    if [ -n "$dir" ] && [ -e "$dir/libcuda.so.1" ]; then
      export NVIDIA_LIBS="$dir"
      export LLM_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
      COMPOSE_FILES="$COMPOSE_FILES -f compose.gpu.yaml"
      echo "NVIDIA GPU detected, using $dir"
      break
    fi
  done
fi

ACTION="${1:-up}"
PORT_VALUE="$(env_value PORT)"
PORT_VALUE="${PORT_VALUE:-8000}"
LLM_PORT="$(env_value LLM_PORT)"
LLM_PORT="${LLM_PORT:-8080}"
MODEL_NAME="$(env_value LLM_MODEL)"
MODEL_NAME="${MODEL_NAME:-lfm3-1.7b}"

wait_for_model() {
  tries=0
  while [ "$tries" -lt 600 ]; do
    if curl -fsS -m 2 "http://localhost:$LLM_PORT/health" >/dev/null 2>&1; then
      echo "Model ready: $MODEL_NAME"
      curl -fsS -m 120 "http://localhost:$LLM_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
        >/dev/null 2>&1 || true
      return 0
    fi
    if [ $((tries % 15)) -eq 0 ]; then
      echo "Waiting for the model. The first run downloads and loads it."
    fi
    tries=$((tries + 1))
    sleep 2
  done
  echo "Model still loading. Check ./run.sh logs and open http://localhost:$PORT_VALUE" >&2
}

wait_for_health() {
  url="http://localhost:$PORT_VALUE/api/health"
  tries=0
  while [ "$tries" -lt 60 ]; do
    if curl -fsS -m 2 "$url" >/dev/null 2>&1; then
      echo "Ready: $url"
      return 0
    fi
    tries=$((tries + 1))
    sleep 1
  done
  echo "Still starting. Check ./run.sh logs, then open http://localhost:$PORT_VALUE" >&2
}

open_browser() {
  if command -v open >/dev/null 2>&1; then
    open "http://localhost:$PORT_VALUE"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:$PORT_VALUE" >/dev/null 2>&1 || true
  else
    echo "Open http://localhost:$PORT_VALUE in a browser." >&2
  fi
}

case "$ACTION" in
  up)
    $COMPOSE $COMPOSE_FILES up -d --build
    wait_for_model
    wait_for_health
    echo "Askline is running at http://localhost:$PORT_VALUE"
    echo "Logs: ./run.sh logs"
    ;;
  down)
    $COMPOSE $COMPOSE_FILES down
    ;;
  logs)
    $COMPOSE $COMPOSE_FILES logs -f askline
    ;;
  build)
    $COMPOSE $COMPOSE_FILES build
    ;;
  open)
    open_browser
    ;;
  *)
    echo "Usage: ./run.sh [up|down|logs|build|open]" >&2
    exit 2
    ;;
esac
