# syntax=docker/dockerfile:1.7
# Distillery SageMaker real-time inference image. No credentials or model weights.

ARG BASE_IMAGE=pytorch/pytorch@sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726
FROM ${BASE_IMAGE}

ARG DISTILLERY_SOURCE_SHA
ARG DISTILLERY_SOURCE_TREE_SHA256
ARG DISTILLERY_LOCK_SHA256
ARG DISTILLERY_IMAGE_VERSION=0.1.0
ARG SOURCE_DATE_EPOCH=0
ARG UV_VERSION=0.6.14
ARG UV_ARCHIVE_SHA256=0aaf451c391d3913823bfb8ed354b446dcfd0553a32ed8266611e4181c61fd51

LABEL org.opencontainers.image.title="distillery-inference" \
      org.opencontainers.image.description="SageMaker real-time inference for Distillery Demo/Playground" \
      org.opencontainers.image.version="${DISTILLERY_IMAGE_VERSION}" \
      org.opencontainers.image.source="https://github.com/nominal-io/ramp-hackathon" \
      org.opencontainers.image.revision="${DISTILLERY_SOURCE_SHA}" \
      distillery.component="inference" \
      distillery.runtime.uid="1000" \
      distillery.source.sha="${DISTILLERY_SOURCE_SHA}" \
      distillery.source.tree.sha256="${DISTILLERY_SOURCE_TREE_SHA256}" \
      distillery.package.lock.sha256="${DISTILLERY_LOCK_SHA256}" \
      distillery.base.image="pytorch/pytorch@sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726" \
      distillery.ml.torch="2.4.1" \
      distillery.ml.cuda="12.4" \
      distillery.ml.bitsandbytes="0.44.1" \
      distillery.entrypoint="/opt/distillery/container_entrypoint.py"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/distillery/.venv \
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

RUN python - "${UV_VERSION}" "${UV_ARCHIVE_SHA256}" <<'PY'
import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path
import urllib.request

version, expected_sha256 = sys.argv[1:]
url = (
    f"https://github.com/astral-sh/uv/releases/download/{version}/"
    "uv-x86_64-unknown-linux-gnu.tar.gz"
)
with urllib.request.urlopen(url, timeout=60) as response:
    archive_bytes = response.read()
actual_sha256 = hashlib.sha256(archive_bytes).hexdigest()
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"uv archive checksum mismatch: expected {expected_sha256}, "
        f"found {actual_sha256}"
    )
member_name = "uv-x86_64-unknown-linux-gnu/uv"
with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
    member = archive.getmember(member_name)
    source = archive.extractfile(member)
    if source is None or not member.isfile():
        raise SystemExit(f"uv archive lacks regular file {member_name}")
    destination = Path("/usr/local/bin/uv")
    destination.write_bytes(source.read())
    os.chmod(destination, 0o755)
    os.utime(destination, (0, 0))
PY

RUN set -eux; \
    test -n "${DISTILLERY_SOURCE_SHA}"; \
    test -n "${DISTILLERY_SOURCE_TREE_SHA256}"; \
    test -n "${DISTILLERY_LOCK_SHA256}"; \
    groupadd --gid 1000 distillery; \
    useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin distillery; \
    mkdir -p \
      /opt/distillery/hf-cache \
      /opt/ml/model \
      /opt/ml/code; \
    python -m venv --system-site-packages /opt/distillery/.venv; \
    chown -R distillery:distillery /opt/distillery/hf-cache; \
    chmod 0755 /opt/ml /opt/ml/model /opt/ml/code

WORKDIR /opt/distillery

# Reuse training ML compatibility pins without editing training-container files.
COPY --chown=root:root pyproject.toml uv.lock README.md LICENSE ./
COPY --chown=root:root src/distillery ./src/distillery
COPY --chown=root:root apps/inference ./apps/inference
COPY --chown=root:root containers/inference/container_entrypoint.py ./
COPY --chown=root:root containers/inference/requirements-serve.txt ./
COPY --chown=root:root containers/training/ml-compatibility.json ./
COPY --chown=root:root containers/training/verify_ml_compatibility.py ./

RUN set -eux; \
    python verify_ml_compatibility.py lock \
      --lock uv.lock \
      --compatibility ml-compatibility.json; \
    python verify_ml_compatibility.py runtime \
      --compatibility ml-compatibility.json

RUN set -eux; \
    uv sync --frozen --no-dev --extra ml --no-editable --inexact \
      --no-install-package torch \
      --no-install-package triton \
      --no-install-package nvidia-cublas-cu12 \
      --no-install-package nvidia-cuda-cupti-cu12 \
      --no-install-package nvidia-cuda-nvrtc-cu12 \
      --no-install-package nvidia-cuda-runtime-cu12 \
      --no-install-package nvidia-cudnn-cu12 \
      --no-install-package nvidia-cufft-cu12 \
      --no-install-package nvidia-curand-cu12 \
      --no-install-package nvidia-cusolver-cu12 \
      --no-install-package nvidia-cusparse-cu12 \
      --no-install-package nvidia-nccl-cu12 \
      --no-install-package nvidia-nvjitlink-cu12 \
      --no-install-package nvidia-nvtx-cu12; \
    uv pip install --no-deps -r requirements-serve.txt; \
    python verify_ml_compatibility.py runtime \
      --compatibility ml-compatibility.json \
      --require-bitsandbytes; \
    python -c "import distillery_inference.server; import uvicorn"

RUN python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "schema_version": "distillery.inference.image.version.v1",
    "component": "inference",
    "image_version": os.environ["DISTILLERY_IMAGE_VERSION"],
    "source_sha": os.environ["DISTILLERY_SOURCE_SHA"],
    "source_tree_sha256": os.environ["DISTILLERY_SOURCE_TREE_SHA256"],
    "package_lock_sha256": os.environ["DISTILLERY_LOCK_SHA256"],
    "entrypoint": "/opt/distillery/container_entrypoint.py",
    "server_module": "distillery_inference.server",
    "runtime_uid": 1000,
    "torch_version": "2.4.1",
    "torch_cuda_version": "12.4",
    "bitsandbytes_version": "0.44.1",
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
    chmod 0555 \
      /opt/distillery/container_entrypoint.py \
      /opt/distillery/verify_ml_compatibility.py; \
    ! test -e /opt/distillery/.env; \
    ! test -e /home/distillery/.aws; \
    ! test -e /opt/distillery/credentials

RUN python - <<'PY'
from pathlib import Path

sensitive_names = {".env", "credentials", "id_ed25519", "id_rsa"}
private_key_markers = (
    b"-----BEGIN PRIVATE KEY-----",
    b"-----BEGIN RSA PRIVATE KEY-----",
    b"-----BEGIN OPENSSH PRIVATE KEY-----",
)
for path in Path("/opt/distillery").rglob("*"):
    if not path.is_file():
        continue
    if path.name in sensitive_names:
        raise SystemExit(f"credential-like filename in image: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(8192)
    if any(marker in prefix for marker in private_key_markers):
        raise SystemExit(f"private key material in image: {path}")
PY

WORKDIR /opt/ml/code

# PID 1 starts as root only to fix mount ownership, then drops to uid 1000.
USER root
EXPOSE 8080
STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "/opt/distillery/container_entrypoint.py"]
CMD ["--host", "0.0.0.0", "--port", "8080"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "/opt/distillery/container_entrypoint.py", "--health"]
