#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-genai-app:latest}
CONTAINER_NAME=${CONTAINER_NAME:-genai-app}
PORT=${PORT:-8080}
DATA_PATH=${DATA_PATH:-/data/arxiv.jsonl}
HOST_DATA_DIR=${HOST_DATA_DIR:-"$(pwd)/data"}
CONTAINER_DATA_DIR=${CONTAINER_DATA_DIR:-/data}

say(){ printf "%b\n" "$*"; }
die(){ printf "❌ %s\n" "$*" >&2; exit 1; }

# 1) Ensure image
if ! sudo docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  say "⚠️  Image '$IMAGE_NAME' not found. Building..."
  sudo docker build -t "$IMAGE_NAME" . || die "Build failed"
fi

# Helpers
cid() { sudo docker ps -aq -f "name=^${CONTAINER_NAME}$"; }
is_running() {
  local id="$1"
  [[ "$(sudo docker inspect -f '{{.State.Running}}' "$id" 2>/dev/null || echo false)" == "true" ]]
}

# 2) If container exists, (re)use it
CID="$(cid || true)"
if [[ -n "${CID}" ]]; then
  if is_running "$CID"; then
    say "✅ Container '$CONTAINER_NAME' is running. Opening shell..."
    # sudo docker exec -it "$CONTAINER_NAME" /bin/bash || sudo docker exec -it "$CONTAINER_NAME" /bin/sh
    exit 0
  else
    say "ℹ️  Container exists but stopped. Starting..."
    sudo docker start -ai "$CONTAINER_NAME"
    say "✅ Started. Opening shell..."
    # sudo docker exec -it "$CONTAINER_NAME" /bin/bash || sudo docker exec -it "$CONTAINER_NAME" /bin/
    exit 0
  fi
fi

# 3) No container: ask GPU, then run
read -r -p "Use GPU? [y/N] " usegpu
GPU_ARGS=""
if [[ "${usegpu,,}" == "y" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_ARGS="--gpus all --runtime=nvidia"
  else
    say "⚠️  nvidia-smi not found. Running CPU-only."
  fi
fi

MOUNT_ARG=""
if [[ -d "$HOST_DATA_DIR" ]]; then
  MOUNT_ARG="-v ${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}:ro"
elif [[ -f "$HOST_DATA_DIR/arxiv.jsonl" ]]; then
  MOUNT_ARG="-v ${HOST_DATA_DIR}/arxiv.jsonl:${CONTAINER_DATA_DIR}/arxiv.jsonl:ro"
else
  say "⚠️  Data dir/file not found at ${HOST_DATA_DIR}. Running without a mount."
fi

say "🚀 Creating container '${CONTAINER_NAME}'..."
sudo docker run \
  --name "$CONTAINER_NAME" \
  -p ${PORT}:${PORT} \
  -e DATA_PATH="$DATA_PATH" \
  $GPU_ARGS \
  $MOUNT_ARG \
  "$IMAGE_NAME" 



# ok "Container started on port ${PORT}."
say "➡️  Attach: sudo docker attach $CONTAINER_NAME"
say "➡️  Shell:  sudo docker exec -it $CONTAINER_NAME /bin/bash"
say "➡️  Start:  sudo docker start -ai $CONTAINER_NAME" 
say "➡️  Stop:   sudo docker stop $CONTAINER_NAME"
say "➡️  Logs:   sudo docker logs -f $CONTAINER_NAME"
say "➡️  Remove: sudo docker rm $CONTAINER_NAME"
sudo docker ps