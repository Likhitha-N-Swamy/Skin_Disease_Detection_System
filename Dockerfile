# Dockerfile (use this full content)
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8080 \
    MODEL_URL=""

# Install system packages required to build python wheels like pycairo, pillow, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    pkg-config \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libcairo2-dev \
    libgirepository1.0-dev \
    libpango1.0-dev \
    libjpeg-dev \
    wget \
    ca-certificates \
    meson \
    ninja-build \
    python3-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

COPY docker_start.sh /app/docker_start.sh
RUN chmod +x /app/docker_start.sh

EXPOSE 8080
CMD ["/app/docker_start.sh"]
