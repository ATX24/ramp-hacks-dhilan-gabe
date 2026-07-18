#!/usr/bin/env bash
# Build and push a digest-pinned Transformers benchmark image.
# Does not touch the training launch path. Defaults to dry-run.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKERFILE="${ROOT_DIR}/containers/benchmark/Dockerfile"
IMAGE_NAME="distillery-benchmark"
REPO_NAME="distillery-training"
AWS_PROFILE="${AWS_PROFILE:-gabriel-cli}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="225989358036"
BASE_DIGEST="sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
LOCAL_TAG="${IMAGE_NAME}:local"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON="python3"
fi

MODE="dry-run"
for arg in "$@"; do
  case "${arg}" in
    --dry-run) MODE="dry-run" ;;
    --build) MODE="build" ;;
    --push) MODE="push" ;;
    --help|-h) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "error: unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

log() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

SOURCE_SHA="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
LOCK_SHA="$("${PYTHON}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${ROOT_DIR}/uv.lock")"
TREE_SHA="$("${PYTHON}" - <<'PY' "${ROOT_DIR}"
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = []
for relative in (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "src/distillery",
    "experiments/benchmark",
    "experiments/__init__.py",
    "containers/benchmark",
    "containers/training/ml-compatibility.json",
    "containers/training/verify_ml_compatibility.py",
):
    path = root / relative
    if path.is_file():
        paths.append(path)
    else:
        paths.extend(sorted(p for p in path.rglob("*") if p.is_file()))
digest = hashlib.sha256()
for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
)"
CONTENT_TAG="pinned-benchmark-${SOURCE_SHA:0:12}-${TREE_SHA:0:12}"

log "== Distillery benchmark image ${MODE} =="
log "source_sha=${SOURCE_SHA}"
log "source_tree_sha256=${TREE_SHA}"
log "package_lock_sha256=${LOCK_SHA}"
log "content_tag=${CONTENT_TAG}"

if [[ "${MODE}" == "dry-run" ]]; then
  log "dry_run=true"
  exit 0
fi

command -v docker >/dev/null || die "docker required"
docker build \
  -f "${DOCKERFILE}" \
  -t "${LOCAL_TAG}" \
  --build-arg "BASE_IMAGE=pytorch/pytorch@${BASE_DIGEST}" \
  --build-arg "DISTILLERY_SOURCE_SHA=${SOURCE_SHA}" \
  --build-arg "DISTILLERY_SOURCE_TREE_SHA256=${TREE_SHA}" \
  --build-arg "DISTILLERY_LOCK_SHA256=${LOCK_SHA}" \
  "${ROOT_DIR}"
log "docker_build=ok"

if [[ "${MODE}" == "build" ]]; then
  exit 0
fi

aws --profile "${AWS_PROFILE}" --region "${AWS_REGION}" ecr get-login-password \
  | docker login --username AWS --password-stdin "${REGISTRY}"
REMOTE="${REGISTRY}/${REPO_NAME}:${CONTENT_TAG}"
docker tag "${LOCAL_TAG}" "${REMOTE}"
docker push "${REMOTE}"
DIGEST="$(aws --profile "${AWS_PROFILE}" --region "${AWS_REGION}" ecr describe-images \
  --repository-name "${REPO_NAME}" \
  --image-ids "imageTag=${CONTENT_TAG}" \
  --query 'imageDetails[0].imageDigest' \
  --output text)"
PINNED="${REGISTRY}/${REPO_NAME}@${DIGEST}"
log "pushed_tag=${REMOTE}"
log "digest_pinned_uri=${PINNED}"
printf '%s\n' "${PINNED}" > "${ROOT_DIR}/experiments/benchmark/.last_image_uri"
log "wrote experiments/benchmark/.last_image_uri"
