#!/usr/bin/env bash
# Build or plan the Distillery inference image. Default: local dry-run only.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKERFILE="${ROOT_DIR}/containers/inference/Dockerfile"
COMPATIBILITY="${ROOT_DIR}/containers/training/ml-compatibility.json"
IMAGE_NAME="distillery-inference"
BASE_DIGEST="sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${DISTILLERY_PACKAGING_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
else
  PYTHON="${DISTILLERY_PACKAGING_PYTHON:-python3}"
fi

MODE="dry-run"
REVIEWED_SOURCE_SHA=""

for arg in "$@"; do
  case "${arg}" in
    --dry-run) MODE="dry-run" ;;
    --build) MODE="build" ;;
    --reviewed-source-sha=*) REVIEWED_SOURCE_SHA="${arg#*=}" ;;
    --help|-h)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      printf 'error: unknown argument: %s\n' "${arg}" >&2
      exit 2
      ;;
  esac
done

log() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

sha256_file() {
  "${PYTHON}" -c \
    'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
    "$1"
}

[[ -f "${DOCKERFILE}" ]] || die "missing Dockerfile: ${DOCKERFILE}"
[[ -f "${COMPATIBILITY}" ]] || die "missing ML compatibility pin: ${COMPATIBILITY}"

SOURCE_SHA="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
LOCK_SHA="$(sha256_file "${ROOT_DIR}/uv.lock")"
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
    "apps/inference",
    "containers/inference",
    "containers/training/ml-compatibility.json",
    "containers/training/verify_ml_compatibility.py",
):
    path = root / relative
    if path.is_file():
        paths.append(path)
    else:
        paths.extend(sorted(p for p in path.rglob("*") if p.is_file()))
digest = hashlib.sha256()
for path in paths:
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
)"

log "== Distillery inference image ${MODE} =="
log "image=${IMAGE_NAME}"
log "dockerfile=${DOCKERFILE}"
log "base_digest=${BASE_DIGEST}"
log "source_sha=${SOURCE_SHA}"
log "source_tree_sha256=${TREE_SHA}"
log "package_lock_sha256=${LOCK_SHA}"
log "compatibility=${COMPATIBILITY}"

require_cmd "${PYTHON}"
"${PYTHON}" "${ROOT_DIR}/containers/training/verify_ml_compatibility.py" lock \
  --lock "${ROOT_DIR}/uv.lock" \
  --compatibility "${COMPATIBILITY}"
log "ml_compatibility_lock_check=ok"

if [[ "${MODE}" == "dry-run" ]]; then
  log "dry_run=true"
  log "docker_build=skipped"
  log "publish=skipped"
  exit 0
fi

[[ -n "${REVIEWED_SOURCE_SHA}" ]] \
  || die "--build requires --reviewed-source-sha=<40-hex commit>"
[[ "${REVIEWED_SOURCE_SHA}" == "${SOURCE_SHA}" ]] \
  || die "reviewed source SHA must equal HEAD"
[[ -z "$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=all -- \
  apps/inference containers/inference scripts/inference \
  containers/training/ml-compatibility.json \
  containers/training/verify_ml_compatibility.py \
  pyproject.toml uv.lock src/distillery README.md LICENSE)" ]] \
  || die "inference image inputs are dirty; commit before --build"

require_cmd docker
docker build \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_NAME}:local" \
  --build-arg "BASE_IMAGE=pytorch/pytorch@${BASE_DIGEST}" \
  --build-arg "DISTILLERY_SOURCE_SHA=${SOURCE_SHA}" \
  --build-arg "DISTILLERY_SOURCE_TREE_SHA256=${TREE_SHA}" \
  --build-arg "DISTILLERY_LOCK_SHA256=${LOCK_SHA}" \
  "${ROOT_DIR}"
log "docker_build=ok"
