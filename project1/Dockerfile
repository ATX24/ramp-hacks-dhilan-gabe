FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY packages ./packages
COPY apps ./apps

RUN pip install --no-cache-dir . && mkdir -p /data

ENV DISTILLERY_DATA_DIR=/data \
    PYTHONPATH=/app:/app/src:/app/packages/sdk

EXPOSE 8000
CMD ["uvicorn", "apps.api.distillery_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
