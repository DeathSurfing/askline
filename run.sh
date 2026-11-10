#!/usr/bin/env sh
set -eu

# One command to build and run Askline with Docker or Podman.
# Usage: ./run.sh [up|down|logs|build|open]

cd "$(dirname "$0")"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add your LLM_API_KEY, then rerun." >&2
  echo "For a local model, set LLM_BASE_URL=http://host.docker.internal:11434/v1" >&2
  echo "and LLM_MODEL to a model you have pulled. On Linux, host.docker.internal" >&2
  echo "needs an extra_hosts entry, or use the host's LAN address instead." >&2
  exit 1
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

ACTION="${1:-up}"
PORT_VALUE="$(grep -E '^PORT=' .env 2>/dev/null | tail -1 | cut -d= -f2 || true)"
PORT_VALUE="${PORT_VALUE:-8000}"

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
    $COMPOSE up -d --build
    wait_for_health
    echo "Askline is running at http://localhost:$PORT_VALUE"
    echo "Logs: ./run.sh logs"
    ;;
  down)
    $COMPOSE down
    ;;
  logs)
    $COMPOSE logs -f askline
    ;;
  build)
    $COMPOSE build
    ;;
  open)
    open_browser
    ;;
  *)
    echo "Usage: ./run.sh [up|down|logs|build|open]" >&2
    exit 2
    ;;
esac
