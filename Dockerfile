# syntax=docker/dockerfile:1
FROM python:3.10-slim AS runtime

# cpu | cu118 | cu121
ARG TORCH_VARIANT=cu118
ARG TORCH_VERSION=2.4.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    PATH=/app/.local/bin:$PATH

WORKDIR ${APP_HOME}

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# --- Cài torch trước (chỉ torch) ---
RUN python -m pip install --upgrade pip setuptools wheel && \
    if [ "$TORCH_VARIANT" = "cpu" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"; \
    elif [ "$TORCH_VARIANT" = "cu118" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 "torch==${TORCH_VERSION}+cu118"; \
    elif [ "$TORCH_VARIANT" = "cu121" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 "torch==${TORCH_VERSION}+cu121"; \
    else \
      echo "Unknown TORCH_VARIANT: $TORCH_VARIANT (use cpu|cu118|cu121)" && exit 1; \
    fi

# --- Copy code & cài project (deps sẽ dùng torch đã có) ---
COPY backend/ ./
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
