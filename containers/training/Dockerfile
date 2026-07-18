# syntax=docker/dockerfile:1.7
# Distillery SageMaker training image. No credentials or model weights in layers.

ARG BASE_IMAGE=pytorch/pytorch@sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726
FROM ${BASE_IMAGE}

ARG DISTILLERY_SOURCE_SHA
ARG DISTILLERY_SOURCE_TREE_SHA256
ARG DISTILLERY_LOCK_SHA256
ARG DISTILLERY_IMAGE_VERSION=0.1.0
ARG SOURCE_DATE_EPOCH=0
ARG UV_VERSION=0.6.14
ARG UV_ARCHIVE_SHA256=0aaf451c391d3913823bfb8ed354b446dcfd0553a32ed8266611e4181c61fd51

LABEL org.opencontainers.image.title="distillery-training" \
      org.opencontainers.image.description="SageMaker training image for Distillery/TinyFable" \
      org.opencontainers.image.version="${DISTILLERY_IMAGE_VERSION}" \
      org.opencontainers.image.source="https://github.com/nominal-io/ramp-hackathon" \
      org.opencontainers.image.revision="${DISTILLERY_SOURCE_SHA}" \
      distillery.component="training" \
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
    PYTHONPATH="/opt/distillery/src" \
    HF_HOME=/opt/distillery/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/distillery/hf-cache/hub \
    TRANSFORMERS_CACHE=/opt/distillery/hf-cache/transformers \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    DISTILLERY_REQUIRE_PINNED_REVISION=1 \
    DISTILLERY_MODEL_CACHE=/opt/distillery/models \
    DISTILLERY_S3_MODEL_CHANNEL=models \
    DISTILLERY_SAGEMAKER_MODEL_INPUT=/opt/ml/input/data/models \
    DISTILLERY_SAGEMAKER_MANIFEST_INPUT=/opt/ml/input/data/manifest \
    DISTILLERY_SAGEMAKER_RESPONSES_INPUT=/opt/ml/input/data/responses \
    DISTILLERY_SAGEMAKER_DATASET_INPUT=/opt/ml/input/data/dataset \
    DISTILLERY_SAGEMAKER_MODEL_OUTPUT=/opt/ml/model \
    DISTILLERY_SAGEMAKER_OUTPUT=/opt/ml/output \
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
import urllib.request
from pathlib import Path

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
      /opt/distillery/models \
      /opt/ml/code \
      /opt/ml/input/data \
      /opt/ml/model \
      /opt/ml/output; \
    python -m venv --system-site-packages /opt/distillery/.venv; \
    chown -R distillery:distillery \
      /opt/distillery/hf-cache \
      /opt/distillery/models; \
    chmod 0755 /opt/ml /opt/ml/input /opt/ml/input/data

WORKDIR /opt/distillery

# The build script creates this context from an explicit allowlist and installs
# the repository .dockerignore at context root.
COPY --chown=root:root pyproject.toml uv.lock README.md LICENSE ./
COPY --chown=root:root src/distillery ./src/distillery
COPY --chown=root:root containers/training/container_entrypoint.py ./
COPY --chown=root:root containers/training/ml-compatibility.json ./
COPY --chown=root:root containers/training/verify_ml_compatibility.py ./

# Stop before dependency installation unless the lock exactly matches the
# immutable Torch 2.4.1/CUDA 12.4/cuDNN 9 base and intended bitsandbytes build.
RUN set -eux; \
    python verify_ml_compatibility.py lock \
      --lock uv.lock \
      --compatibility ml-compatibility.json; \
    python verify_ml_compatibility.py runtime \
      --compatibility ml-compatibility.json

# Sync into a system-site-packages venv while excluding accelerator packages
# supplied by the base. Exact pre/post checks prevent a second Torch/CUDA stack.
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
    python verify_ml_compatibility.py runtime \
      --compatibility ml-compatibility.json \
      --require-bitsandbytes; \
    python -m distillery.training.entrypoint --help >/dev/null

RUN python - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "schema_version": "distillery.training.image.version.v2",
    "component": "training",
    "image_version": os.environ["DISTILLERY_IMAGE_VERSION"],
    "source_sha": os.environ["DISTILLERY_SOURCE_SHA"],
    "source_tree_sha256": os.environ["DISTILLERY_SOURCE_TREE_SHA256"],
    "package_lock_sha256": os.environ["DISTILLERY_LOCK_SHA256"],
    "entrypoint": "/opt/distillery/container_entrypoint.py",
    "trainer_module": "distillery.training.entrypoint",
    "runtime_uid": 1000,
    "torch_version": "2.4.1",
    "torch_cuda_version": "12.4",
    "bitsandbytes_version": "0.44.1",
    "hf_offline": True,
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

# PID 1 starts as root only to fix SageMaker mount ownership, then permanently
# drops to uid/gid 1000 before validating channels or launching the trainer.
USER root
STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "/opt/distillery/container_entrypoint.py"]
CMD ["--manifest", "/opt/ml/input/data/manifest/manifest.json", \
     "--responses", "/opt/ml/input/data/responses/responses.jsonl", \
     "--output-dir", "/opt/ml/model/validation", \
     "--validate-only"]

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=2 \
  CMD ["python", "/opt/distillery/container_entrypoint.py", "--health"]
