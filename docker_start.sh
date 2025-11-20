#!/usr/bin/env bash
set -e

MODEL_DIR=/app/models
MODEL_FILE=${MODEL_DIR}/best_model.h5

mkdir -p "${MODEL_DIR}"

# If MODEL_URL env var is set, download the model at container start
if [ -n "${MODEL_URL}" ] && [ ! -f "${MODEL_FILE}" ]; then
  echo "Downloading model from ${MODEL_URL} ..."
  wget -q --show-progress -O "${MODEL_FILE}" "${MODEL_URL}" || \
    { echo "Model download failed"; exit 1; }
  echo "Downloaded model to ${MODEL_FILE}"
fi

# Start Gunicorn; adjust workers/threads to suit memory/cpu
exec gunicorn --bind "0.0.0.0:${PORT}" --workers 2 --threads 4 --timeout 120 app:app
