#!/usr/bin/env bash
# Verify/publish the Distillery inference image. Default: dry-run only.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
REPOSITORY_NAME="distillery-inference"
IMAGE_NAME="distillery-inference"
LOCAL_IMAGE_REF="${IMAGE_NAME}:local"

MODE="dry-run"
for arg in "$@"; do
  case "${arg}" in
    --dry-run) MODE="dry-run" ;;
    --publish) MODE="publish" ;;
    --help|-h)
      sed -n '2,14p' "$0"
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

log "== Distillery inference image publish ${MODE} =="
log "root=${ROOT_DIR}"
log "repository=${REPOSITORY_NAME}"
log "region=${REGION}"
log "local_image=${LOCAL_IMAGE_REF}"

if [[ "${MODE}" == "dry-run" ]]; then
  log "dry_run=true"
  log "ecr_login=skipped"
  log "docker_push=skipped"
  log "remaining_inputs=AWS_PROFILE,ECR repository ${REPOSITORY_NAME}, local image digest"
  exit 0
fi

[[ -n "${AWS_PROFILE:-}" ]] || die "AWS_PROFILE is required for --publish"
require_cmd aws
require_cmd docker
identity="$(aws sts get-caller-identity --profile "${AWS_PROFILE}" --region "${REGION}" --output json)"
arn="$(printf '%s' "${identity}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Arn"])')"
[[ "${arn}" != *":root" ]] || die "refusing to publish with account root identity (${arn})"
account="$(printf '%s' "${identity}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Account"])')"
registry="${account}.dkr.ecr.${REGION}.amazonaws.com"
aws ecr get-login-password --profile "${AWS_PROFILE}" --region "${REGION}" \
  | docker login --username AWS --password-stdin "${registry}"
digest_uri="$(docker image inspect --format '{{index .RepoDigests 0}}' "${LOCAL_IMAGE_REF}" 2>/dev/null || true)"
[[ -n "${digest_uri}" ]] || die "local image missing RepoDigests; tag/push once then re-inspect"
log "publish path requires an existing ECR repository under ${ROOT_DIR} and prior local build"
die "publish not completed: wire ECR repository + digest tag explicitly before re-running"
