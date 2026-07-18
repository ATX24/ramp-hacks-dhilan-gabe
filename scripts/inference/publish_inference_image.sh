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
repository_uri="${registry}/${REPOSITORY_NAME}"
source_sha="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
tree_sha="$(git -C "${ROOT_DIR}" rev-parse HEAD:apps/inference)"
image_tag="inference-${source_sha:0:12}-${tree_sha:0:12}"
remote_ref="${repository_uri}:${image_tag}"

if ! aws ecr describe-repositories \
  --profile "${AWS_PROFILE}" \
  --region "${REGION}" \
  --repository-names "${REPOSITORY_NAME}" >/dev/null 2>&1; then
  aws ecr create-repository \
    --profile "${AWS_PROFILE}" \
    --region "${REGION}" \
    --repository-name "${REPOSITORY_NAME}" \
    --image-tag-mutability IMMUTABLE \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    --tags Key=Project,Value=RampHackathon \
      Key=Owner,Value=Gabriel \
      Key=TTL,Value=2026-07-20 \
      Key=Component,Value=distillery-inference >/dev/null
  log "repository_created=true"
fi

aws ecr get-login-password --profile "${AWS_PROFILE}" --region "${REGION}" \
  | docker login --username AWS --password-stdin "${registry}"
docker image inspect "${LOCAL_IMAGE_REF}" >/dev/null 2>&1 \
  || die "local image missing: ${LOCAL_IMAGE_REF}"
docker tag "${LOCAL_IMAGE_REF}" "${remote_ref}"
docker push "${remote_ref}"
digest="$(aws ecr describe-images \
  --profile "${AWS_PROFILE}" \
  --region "${REGION}" \
  --repository-name "${REPOSITORY_NAME}" \
  --image-ids "imageTag=${image_tag}" \
  --query 'imageDetails[0].imageDigest' \
  --output text)"
[[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "ECR returned invalid digest: ${digest}"
log "image_tag=${image_tag}"
log "digest_uri=${repository_uri}@${digest}"
log "publish_complete=true"
