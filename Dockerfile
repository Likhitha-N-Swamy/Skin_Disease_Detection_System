# Dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8080 \
    MODEL_URL=""

# system deps required by opencv, pillow, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app sources
COPY . /app

# small startup script to download model if MODEL_URL provided
COPY docker_start.sh /app/docker_start.sh
RUN chmod +x /app/docker_start.sh

EXPOSE 8080

CMD ["/app/docker_start.sh"]
