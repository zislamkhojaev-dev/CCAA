FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY apps/backend/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY apps /app/apps

# Best-effort warm cache for semantic model artifacts to reduce cold-start.
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='intfloat/multilingual-e5-small', filename='onnx/model.onnx'); hf_hub_download(repo_id='intfloat/multilingual-e5-small', filename='tokenizer.json')" || true

EXPOSE 8000

CMD ["uvicorn", "apps.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
