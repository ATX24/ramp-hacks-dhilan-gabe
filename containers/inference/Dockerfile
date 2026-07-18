# syntax=docker/dockerfile:1.7
# Thin serving derivative of the exact verified training runtime. No model weights.

ARG BASE_IMAGE=225989358036.dkr.ecr.us-east-1.amazonaws.com/distillery-training@sha256:fce8fe86d50f994ad7b38e443f55d0a675abb3793f23f5423c073bdb910f1976
FROM ${BASE_IMAGE}

ARG DISTILLERY_SOURCE_SHA
ARG DISTILLERY_SOURCE_TREE_SHA256
ARG DISTILLERY_LOCK_SHA256
ARG DISTILLERY_IMAGE_VERSION=0.2.0
ARG SOURCE_DATE_EPOCH=0

LABEL org.opencontainers.image.title="distillery-inference" \
      org.opencontainers.image.description="SageMaker real-time inference for Distillery Demo/Playground" \
      org.opencontainers.image.version="${DISTILLERY_IMAGE_VERSION}" \
      org.opencontainers.image.source="https://github.com/ATX24/ramp-hacks-dhilan-gabe" \
      org.opencontainers.image.revision="${DISTILLERY_SOURCE_SHA}" \
      distillery.component="inference" \
      distillery.runtime.uid="1000" \
      distillery.source.sha="${DISTILLERY_SOURCE_SHA}" \
      distillery.source.tree.sha256="${DISTILLERY_SOURCE_TREE_SHA256}" \
      distillery.package.lock.sha256="${DISTILLERY_LOCK_SHA256}" \
      distillery.training.image.digest="sha256:fce8fe86d50f994ad7b38e443f55d0a675abb3793f23f5423c073bdb910f1976" \
      distillery.ml.torch="2.4.1" \
      distillery.ml.cuda="12.4" \
      distillery.entrypoint="/opt/distillery/inference_entrypoint.py"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/distillery/.venv/bin:/opt/conda/bin:/usr/local/bin:${PATH}" \
    PYTHONPATH="/opt/distillery/apps/inference:/opt/distillery/src" \
    HF_HOME=/opt/distillery/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/distillery/hf-cache/hub \
    TRANSFORMERS_CACHE=/opt/distillery/hf-cache/transformers \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    DISTILLERY_REQUIRE_PINNED_REVISION=1 \
    DISTILLERY_INFERENCE_REQUIRE_OFFLINE=1 \
    DISTILLERY_INFERENCE_RUNTIME=torch \
    SM_MODEL_DIR=/opt/ml/model \
    DISTILLERY_INFERENCE_MODEL_ROOT=/opt/ml/model \
    DISTILLERY_IMAGE_VERSION=${DISTILLERY_IMAGE_VERSION} \
    DISTILLERY_SOURCE_SHA=${DISTILLERY_SOURCE_SHA} \
    DISTILLERY_SOURCE_TREE_SHA256=${DISTILLERY_SOURCE_TREE_SHA256} \
    DISTILLERY_LOCK_SHA256=${DISTILLERY_LOCK_SHA256}

USER root
WORKDIR /opt/distillery

COPY --chown=root:root apps/inference ./apps/inference
COPY --chown=root:root containers/inference/container_entrypoint.py ./inference_entrypoint.py
COPY --chown=root:root containers/inference/requirements-serve.txt ./

RUN set -eux; \
    test -n "${DISTILLERY_SOURCE_SHA}"; \
    test -n "${DISTILLERY_SOURCE_TREE_SHA256}"; \
    test -n "${DISTILLERY_LOCK_SHA256}"; \
    uv pip install \
      --python /opt/distillery/.venv/bin/python \
      --no-cache \
      -r requirements-serve.txt; \
    python verify_ml_compatibility.py runtime \
      --compatibility ml-compatibility.json \
      --require-bitsandbytes; \
    python -c "import distillery_inference.server; import uvicorn"

RUN python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "schema_version": "distillery.inference.image.version.v2",
    "component": "inference",
    "image_version": os.environ["DISTILLERY_IMAGE_VERSION"],
    "source_sha": os.environ["DISTILLERY_SOURCE_SHA"],
    "source_tree_sha256": os.environ["DISTILLERY_SOURCE_TREE_SHA256"],
    "package_lock_sha256": os.environ["DISTILLERY_LOCK_SHA256"],
    "entrypoint": "/opt/distillery/inference_entrypoint.py",
    "server_module": "distillery_inference.server",
    "runtime_uid": 1000,
    "torch_version": "2.4.1",
    "torch_cuda_version": "12.4",
    "training_image_digest": (
        "sha256:fce8fe86d50f994ad7b38e443f55d0a"
        "675abb3793f23f5423c073bdb910f1976"
    ),
    "hf_offline": True,
    "require_offline": True,
    "require_pinned_revision": True,
}
Path("/opt/distillery/VERSION.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.utime("/opt/distillery/VERSION.json", (0, 0))
PY

RUN set -eux; \
    chmod 0555 /opt/distillery/inference_entrypoint.py; \
    ! test -e /opt/distillery/.env; \
    ! test -e /home/distillery/.aws; \
    ! test -e /opt/distillery/credentials

WORKDIR /opt/ml/code
USER root
EXPOSE 8080
STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "/opt/distillery/inference_entrypoint.py"]
CMD ["--host", "0.0.0.0", "--port", "8080"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "/opt/distillery/inference_entrypoint.py", "--health"]
