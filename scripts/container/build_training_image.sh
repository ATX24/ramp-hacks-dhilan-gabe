#!/usr/bin/env bash
# Build or plan the Distillery training image. Default: local dry-run only.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DOCKERFILE="${ROOT_DIR}/containers/training/Dockerfile"
SCHEMA="${ROOT_DIR}/containers/training/build-manifest.schema.json"
COMPATIBILITY="${ROOT_DIR}/containers/training/ml-compatibility.json"
STAGE_TOOL="${ROOT_DIR}/scripts/container/stage_context.py"
MANIFEST_TOOL="${ROOT_DIR}/scripts/container/manifest_tool.py"
MANIFEST_OUT="${DISTILLERY_BUILD_MANIFEST:-${ROOT_DIR}/containers/training/build-manifest.generated.json}"
BASE_DIGEST="sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726"
IMAGE_NAME="distillery-training"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${DISTILLERY_PACKAGING_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
else
  PYTHON="${DISTILLERY_PACKAGING_PYTHON:-python3}"
fi

MODE="dry-run"
REVIEWED_SOURCE_SHA=""
STAGE_ONLY_DIR=""
STAGE_DIR=""
SOURCE_SNAPSHOT_DIR=""

for arg in "$@"; do
  case "${arg}" in
    --dry-run) MODE="dry-run" ;;
    --build) MODE="build" ;;
    --reviewed-source-sha=*) REVIEWED_SOURCE_SHA="${arg#*=}" ;;
    --stage-only=*)
      MODE="stage-only"
      STAGE_ONLY_DIR="${arg#*=}"
      ;;
    --help|-h)
      sed -n '2,22p' "$0"
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

assert_safe_path() {
  local value="$1"
  local label="$2"
  [[ -n "${value}" ]] || die "${label} must not be empty"
  [[ "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${label} contains a control character"
}

assert_digest() {
  [[ "$1" =~ ^sha256:[0-9a-f]{64}$ ]] || die "$2 is not sha256:<64 lowercase hex>"
}

assert_tag() {
  [[ "$1" =~ ^pinned-training-[0-9a-f]{12}-[0-9a-f]{12}$ ]] \
    || die "image tag is not the pinned content-derived form: $1"
  [[ "${1,,}" != *latest* ]] || die "image tag must not contain latest"
}

verify_commit_exists() {
  local candidate="$1"
  [[ "${candidate}" =~ ^[0-9a-f]{40}$ ]] \
    || die "reviewed source SHA must be exactly 40 lowercase hex characters"
  git -C "${ROOT_DIR}" cat-file -e "${candidate}^{commit}" 2>/dev/null \
    || die "reviewed source SHA is not an existing git commit: ${candidate}"
  local resolved
  resolved="$(git -C "${ROOT_DIR}" rev-parse "${candidate}^{commit}")"
  [[ "${resolved}" == "${candidate}" ]] \
    || die "reviewed source SHA did not resolve exactly: ${candidate}"
}

require_clean_reviewed_source() {
  [[ -n "${REVIEWED_SOURCE_SHA}" ]] \
    || die "--build requires --reviewed-source-sha=<existing 40-hex commit>"
  verify_commit_exists "${REVIEWED_SOURCE_SHA}"
  local head
  head="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  [[ "${head}" == "${REVIEWED_SOURCE_SHA}" ]] \
    || die "reviewed source SHA must equal HEAD (${head})"
  [[ -z "$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=all)" ]] \
    || die "real builds require a completely clean committed source tree"
  git -C "${ROOT_DIR}" diff --quiet "${REVIEWED_SOURCE_SHA}" -- \
    README.md LICENSE pyproject.toml uv.lock src/distillery experiments \
    containers/training \
    || die "staged image content differs from reviewed commit"
}

refuse_root_profile_if_selected() {
  [[ -n "${AWS_PROFILE:-}" ]] || return 0
  require_cmd aws
  local identity arn
  identity="$(aws sts get-caller-identity \
    --profile "${AWS_PROFILE}" \
    --region "${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}" \
    --output json)" || die "unable to resolve selected AWS profile"
  arn="$(printf '%s' "${identity}" \
    | "${PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["Arn"])')"
  [[ "${arn}" != *":root" ]] \
    || die "refusing to build with account root identity (${arn})"
}

stage_context() {
  local destination="$1"
  local source_root="${2:-${ROOT_DIR}}"
  assert_safe_path "${destination}" "stage destination"
  "${PYTHON}" "${STAGE_TOOL}" \
    --repo "${source_root}" \
    --destination "${destination}"
}

snapshot_reviewed_source() {
  local destination="$1"
  require_cmd tar
  mkdir -p "${destination}"
  git -C "${ROOT_DIR}" archive \
    --format=tar \
    "${REVIEWED_SOURCE_SHA}" \
    README.md \
    LICENSE \
    pyproject.toml \
    uv.lock \
    src/distillery \
    experiments/__init__.py \
    experiments/aws_smoke \
    containers/training \
    | tar -xf - -C "${destination}"
}

create_plan_manifest() {
  local source_sha="$1"
  local source_tree_sha="$2"
  local source_clean="$3"
  local commit_bound="$4"
  local lock_sha="$5"
  local tag="$6"
  local compatibility_result="$7"

  "${PYTHON}" "${MANIFEST_TOOL}" create \
    --schema "${SCHEMA}" \
    --output "${MANIFEST_OUT}" \
    --compatibility-result "${compatibility_result}" \
    --compatibility-config "${COMPATIBILITY}" \
    --commit-sha "${source_sha}" \
    --reviewed-commit-sha "${REVIEWED_SOURCE_SHA}" \
    --tree-sha256 "${source_tree_sha}" \
    --source-clean "${source_clean}" \
    --commit-bound "${commit_bound}" \
    --package-lock-sha256 "${lock_sha}" \
    --tag "${tag}"
  "${PYTHON}" "${MANIFEST_TOOL}" validate \
    --schema "${SCHEMA}" \
    --manifest "${MANIFEST_OUT}" >/dev/null
}

main() {
  require_cmd git
  require_cmd "${PYTHON}"
  [[ -f "${DOCKERFILE}" ]] || die "Dockerfile missing: ${DOCKERFILE}"
  [[ -f "${SCHEMA}" ]] || die "manifest schema missing: ${SCHEMA}"
  [[ -f "${COMPATIBILITY}" ]] || die "ML compatibility contract missing"
  [[ -f "${ROOT_DIR}/README.md" ]] || die "README.md required by package metadata"
  [[ -f "${ROOT_DIR}/LICENSE" ]] || die "LICENSE required by package metadata"
  assert_safe_path "${MANIFEST_OUT}" "manifest output path"

  log "== Distillery training image packaging =="
  log "mode=${MODE}"

  if [[ "${MODE}" == "stage-only" ]]; then
    [[ -n "${STAGE_ONLY_DIR}" ]] || die "--stage-only requires a destination"
    stage_context "${STAGE_ONLY_DIR}"
    log "stage_only_complete=${STAGE_ONLY_DIR}"
    return 0
  fi

  refuse_root_profile_if_selected

  local head source_clean="false" commit_bound="false"
  head="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
  if [[ -z "$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=all)" ]]; then
    source_clean="true"
  fi

  if [[ -n "${REVIEWED_SOURCE_SHA}" ]]; then
    verify_commit_exists "${REVIEWED_SOURCE_SHA}"
    if [[ "${source_clean}" == "true" && "${head}" == "${REVIEWED_SOURCE_SHA}" ]]; then
      commit_bound="true"
    fi
  fi
  if [[ "${MODE}" == "build" ]]; then
    require_clean_reviewed_source
    source_clean="true"
    commit_bound="true"
  fi

  STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/distillery-training-context.XXXXXX")"
  cleanup() {
    if [[ -n "${STAGE_DIR}" && -d "${STAGE_DIR}" ]]; then
      rm -rf "${STAGE_DIR}"
    fi
    if [[ -n "${SOURCE_SNAPSHOT_DIR}" && -d "${SOURCE_SNAPSHOT_DIR}" ]]; then
      rm -rf "${SOURCE_SNAPSHOT_DIR}"
    fi
  }
  trap cleanup EXIT

  local stage_source="${ROOT_DIR}"
  if [[ "${MODE}" == "build" ]]; then
    SOURCE_SNAPSHOT_DIR="$(
      mktemp -d "${TMPDIR:-/tmp}/distillery-reviewed-source.XXXXXX"
    )"
    snapshot_reviewed_source "${SOURCE_SNAPSHOT_DIR}"
    stage_source="${SOURCE_SNAPSHOT_DIR}"
  fi

  local source_tree_sha
  source_tree_sha="$(stage_context "${STAGE_DIR}" "${stage_source}")"
  [[ "${source_tree_sha}" =~ ^[0-9a-f]{64}$ ]] \
    || die "staging tool returned an invalid source tree hash"
  local lock_sha
  lock_sha="$(sha256_file "${STAGE_DIR}/uv.lock")"
  local tag="pinned-training-${head:0:12}-${lock_sha:0:12}"
  assert_tag "${tag}"

  local compatibility_result="${STAGE_DIR}/ML_COMPATIBILITY_RESULT.json"
  local compatibility_error="${STAGE_DIR}/ML_COMPATIBILITY_ERROR.txt"
  local compatibility_exit=0
  "${PYTHON}" "${STAGE_DIR}/containers/training/verify_ml_compatibility.py" lock \
    --lock "${STAGE_DIR}/uv.lock" \
    --compatibility "${STAGE_DIR}/containers/training/ml-compatibility.json" \
    >"${compatibility_result}" 2>"${compatibility_error}" || compatibility_exit=$?
  [[ -s "${compatibility_result}" ]] \
    || die "ML compatibility checker did not emit a result"

  create_plan_manifest \
    "${head}" \
    "${source_tree_sha}" \
    "${source_clean}" \
    "${commit_bound}" \
    "${lock_sha}" \
    "${tag}" \
    "${compatibility_result}"

  log "source_sha=${head}"
  log "source_tree_sha256=${source_tree_sha}"
  log "source_clean=${source_clean}"
  log "commit_bound=${commit_bound}"
  log "package_lock_sha256=${lock_sha}"
  log "manifest=${MANIFEST_OUT}"

  if (( compatibility_exit != 0 )); then
    while IFS= read -r line; do
      log "${line}"
    done <"${compatibility_error}"
    die "ML lock is incompatible with the pinned base; foundation must pin the declared stack"
  fi

  if [[ "${MODE}" == "dry-run" ]]; then
    log "dry_run=true (docker build not invoked)"
    return 0
  fi

  require_cmd docker
  local local_ref="${IMAGE_NAME}:${tag}"
  assert_tag "${tag}"
  local build_command=(
    docker build
    --platform linux/amd64
    --file "${STAGE_DIR}/containers/training/Dockerfile"
    --build-arg "BASE_IMAGE=pytorch/pytorch@${BASE_DIGEST}"
    --build-arg "DISTILLERY_SOURCE_SHA=${head}"
    --build-arg "DISTILLERY_SOURCE_TREE_SHA256=${source_tree_sha}"
    --build-arg "DISTILLERY_LOCK_SHA256=${lock_sha}"
    --build-arg "DISTILLERY_IMAGE_VERSION=0.1.0"
    --build-arg "SOURCE_DATE_EPOCH=0"
    --label "distillery.source.sha=${head}"
    --label "distillery.source.tree.sha256=${source_tree_sha}"
    --label "distillery.package.lock.sha256=${lock_sha}"
    --tag "${local_ref}"
    "${STAGE_DIR}"
  )
  "${build_command[@]}"

  local config_id
  config_id="$(docker image inspect --format '{{.Id}}' "${local_ref}")"
  assert_digest "${config_id}" "local image config ID"
  local image_platform
  image_platform="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "${local_ref}")"
  [[ "${image_platform}" == "linux/amd64" ]] \
    || die "built image platform must be linux/amd64, got ${image_platform}"
  "${PYTHON}" "${MANIFEST_TOOL}" set-local \
    --schema "${SCHEMA}" \
    --manifest "${MANIFEST_OUT}" \
    --config-id "${config_id}"
  log "local_config_id=${config_id}"
  log "local_platform=${image_platform}"
  log "build complete; no registry digest exists until verified post-push"
}

main "$@"
