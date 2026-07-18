#!/usr/bin/env bash
# Verify and publish a locally built Distillery image. Default: read-only dry-run.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCHEMA="${ROOT_DIR}/containers/training/build-manifest.schema.json"
MANIFEST_TOOL="${ROOT_DIR}/scripts/container/manifest_tool.py"
MANIFEST_IN="${DISTILLERY_BUILD_MANIFEST:-${ROOT_DIR}/containers/training/build-manifest.generated.json}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
REPOSITORY_NAME="distillery-training"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${DISTILLERY_PACKAGING_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
else
  PYTHON="${DISTILLERY_PACKAGING_PYTHON:-python3}"
fi

MODE="dry-run"
WORK_DIR=""
for arg in "$@"; do
  case "${arg}" in
    --dry-run) MODE="dry-run" ;;
    --publish) MODE="publish" ;;
    --help|-h)
      sed -n '2,20p' "$0"
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

manifest_get() {
  "${PYTHON}" "${MANIFEST_TOOL}" get \
    --schema "${SCHEMA}" \
    --manifest "${MANIFEST_IN}" \
    --path "$1"
}

json_file_get() {
  local path="$1"
  local key="$2"
  "${PYTHON}" -c \
    'import json,sys; value=json.load(open(sys.argv[1],encoding="utf-8")); print(value[sys.argv[2]])' \
    "${path}" "${key}"
}

assert_digest() {
  [[ "$1" =~ ^sha256:[0-9a-f]{64}$ ]] || die "$2 is not sha256:<64 lowercase hex>"
}

assert_tag() {
  [[ "$1" =~ ^pinned-training-[0-9a-f]{12}-[0-9a-f]{12}$ ]] \
    || die "manifest tag is not the pinned content-derived form"
  [[ "${1,,}" != *latest* ]] || die "manifest tag must not contain latest"
}

require_non_root_profile() {
  [[ -n "${AWS_PROFILE:-}" ]] || die "AWS_PROFILE must select a non-root profile"
  require_cmd aws
  local identity_file="$1"
  aws sts get-caller-identity \
    --profile "${AWS_PROFILE}" \
    --region "${REGION}" \
    --output json >"${identity_file}"
  local arn
  arn="$(json_file_get "${identity_file}" Arn)"
  [[ "${arn}" != *":root" ]] \
    || die "refusing to publish with account root identity (${arn})"
}

validate_repository_configuration() {
  local repository_uri="$1"
  local account="$2"
  local output_file="$3"
  aws ecr describe-repositories \
    --profile "${AWS_PROFILE}" \
    --region "${REGION}" \
    --repository-names "${REPOSITORY_NAME}" \
    --output json >"${output_file}"
  "${PYTHON}" - "${output_file}" "${repository_uri}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected_uri = sys.argv[2]
repositories = payload.get("repositories")
if not isinstance(repositories, list) or len(repositories) != 1:
    raise SystemExit("expected exactly one ECR repository")
repository = repositories[0]
if repository.get("repositoryUri") != expected_uri:
    raise SystemExit("ECR repositoryUri does not match the validated URI")
if repository.get("imageTagMutability") != "IMMUTABLE":
    raise SystemExit("ECR repository must enforce immutable tags")
scan = repository.get("imageScanningConfiguration")
if not isinstance(scan, dict) or scan.get("scanOnPush") is not True:
    raise SystemExit("ECR repository must enable scan-on-push")
encryption = repository.get("encryptionConfiguration")
if not isinstance(encryption, dict) or encryption.get("encryptionType") not in {
    "AES256",
    "KMS",
}:
    raise SystemExit("ECR repository encryption is not enabled")
PY
  [[ "${account}" =~ ^[0-9]{12}$ ]] || die "STS returned an invalid account"
}

require_tag_available() {
  local tag="$1"
  local stdout_file="$2"
  local stderr_file="$3"
  local status=0
  aws ecr describe-images \
    --profile "${AWS_PROFILE}" \
    --region "${REGION}" \
    --repository-name "${REPOSITORY_NAME}" \
    --image-ids "imageTag=${tag}" \
    --output json >"${stdout_file}" 2>"${stderr_file}" || status=$?
  if (( status == 0 )); then
    die "immutable ECR tag already exists: ${tag}"
  fi
  if ! "${PYTHON}" - "${stderr_file}" <<'PY'
import sys
from pathlib import Path

message = Path(sys.argv[1]).read_text(encoding="utf-8")
raise SystemExit(0 if "ImageNotFoundException" in message else 1)
PY
  then
    die "unable to prove immutable tag availability"
  fi
}

wait_for_scan() {
  local tag="$1"
  local scan_file="$2"
  local max_attempts="${DISTILLERY_SCAN_MAX_ATTEMPTS:-30}"
  local poll_seconds="${DISTILLERY_SCAN_POLL_SECONDS:-5}"
  [[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]] || die "invalid scan attempt count"
  [[ "${poll_seconds}" =~ ^[0-9]+$ ]] || die "invalid scan poll interval"

  local attempt status
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    aws ecr describe-image-scan-findings \
      --profile "${AWS_PROFILE}" \
      --region "${REGION}" \
      --repository-name "${REPOSITORY_NAME}" \
      --image-id "imageTag=${tag}" \
      --output json >"${scan_file}"
    status="$("${PYTHON}" - "${scan_file}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
description = payload.get("imageScanStatus")
if not isinstance(description, dict):
    raise SystemExit("scan response lacks imageScanStatus")
print(description.get("status", "UNKNOWN"))
PY
)"
    case "${status}" in
      COMPLETE) return 0 ;;
      ACTIVE|IN_PROGRESS|PENDING) sleep "${poll_seconds}" ;;
      *) die "ECR image scan failed or is unsupported: ${status}" ;;
    esac
  done
  die "ECR image scan did not complete after ${max_attempts} attempts"
}

scan_count() {
  local scan_file="$1"
  local severity="$2"
  "${PYTHON}" - "${scan_file}" "${severity}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
severity = sys.argv[2]
summary = payload.get("imageScanFindings", {}).get("findingSeverityCounts", {})
value = summary.get(severity, 0)
if not isinstance(value, int) or value < 0:
    raise SystemExit(f"invalid {severity} finding count")
print(value)
PY
}

main() {
  require_cmd "${PYTHON}"
  require_cmd git
  require_cmd docker
  [[ -f "${MANIFEST_IN}" ]] || die "build manifest not found: ${MANIFEST_IN}"
  [[ "${REGION}" =~ ^[a-z]{2}(-gov)?-[a-z]+-[1-9][0-9]*$ ]] \
    || die "invalid AWS region: ${REGION}"
  [[ -n "${DISTILLERY_ECR_REPOSITORY_URI:-}" ]] \
    || die "DISTILLERY_ECR_REPOSITORY_URI is required"

  "${PYTHON}" "${MANIFEST_TOOL}" validate \
    --schema "${SCHEMA}" \
    --manifest "${MANIFEST_IN}" >/dev/null

  local image_name tag config_id compatibility commit_sha reviewed_sha
  local source_tree_sha lock_sha source_clean commit_bound dry_run repository_name
  image_name="$(manifest_get image_name)"
  repository_name="$(manifest_get repository_name)"
  tag="$(manifest_get tag)"
  config_id="$(manifest_get local.config_id)"
  compatibility="$(manifest_get ml_compatibility.status)"
  commit_sha="$(manifest_get source.commit_sha)"
  reviewed_sha="$(manifest_get source.reviewed_commit_sha)"
  source_tree_sha="$(manifest_get source.tree_sha256)"
  lock_sha="$(manifest_get package_lock_sha256)"
  source_clean="$(manifest_get source.clean)"
  commit_bound="$(manifest_get source.commit_bound)"
  dry_run="$(manifest_get dry_run)"

  [[ "${image_name}" == "distillery-training" ]] || die "unexpected image name"
  [[ "${repository_name}" == "${REPOSITORY_NAME}" ]] \
    || die "repository is not allowlisted"
  assert_tag "${tag}"
  assert_digest "${config_id}" "local image config ID"
  [[ "${compatibility}" == "compatible" ]] || die "ML compatibility is blocked"
  [[ "${source_clean}" == "true" && "${commit_bound}" == "true" ]] \
    || die "publish requires clean, commit-bound source"
  [[ "${reviewed_sha}" == "${commit_sha}" ]] \
    || die "reviewed source SHA is not bound to the built commit"
  [[ "${dry_run}" == "false" ]] || die "publish requires a real local build manifest"
  git -C "${ROOT_DIR}" cat-file -e "${commit_sha}^{commit}" 2>/dev/null \
    || die "manifest source commit does not exist locally"
  [[ "$(git -C "${ROOT_DIR}" rev-parse HEAD)" == "${commit_sha}" ]] \
    || die "manifest source commit no longer equals HEAD"
  [[ -z "$(git -C "${ROOT_DIR}" status --porcelain --untracked-files=all)" ]] \
    || die "publish requires a completely clean source tree"

  WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/distillery-publish.XXXXXX")"
  cleanup() {
    if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]]; then
      rm -rf "${WORK_DIR}"
    fi
  }
  trap cleanup EXIT

  docker image inspect "${image_name}:${tag}" >"${WORK_DIR}/local-image.json"
  "${PYTHON}" - \
    "${WORK_DIR}/local-image.json" \
    "${config_id}" \
    "${commit_sha}" \
    "${source_tree_sha}" \
    "${lock_sha}" <<'PY'
import json
import sys

path, config_id, source_sha, tree_sha, lock_sha = sys.argv[1:]
images = json.load(open(path, encoding="utf-8"))
if not isinstance(images, list) or len(images) != 1:
    raise SystemExit("docker inspect must return exactly one local image")
image = images[0]
if image.get("Id") != config_id:
    raise SystemExit("local image config ID does not match the build manifest")
labels = image.get("Config", {}).get("Labels", {})
expected_labels = {
    "distillery.source.sha": source_sha,
    "distillery.source.tree.sha256": tree_sha,
    "distillery.package.lock.sha256": lock_sha,
}
for name, expected in expected_labels.items():
    if labels.get(name) != expected:
        raise SystemExit(f"local image label {name} does not match the manifest")
PY

  local identity_file="${WORK_DIR}/identity.json"
  require_non_root_profile "${identity_file}"
  local account
  account="$(json_file_get "${identity_file}" Account)"
  local repository_uri="${DISTILLERY_ECR_REPOSITORY_URI}"
  "${PYTHON}" "${MANIFEST_TOOL}" validate-repository \
    --uri "${repository_uri}" \
    --account "${account}" \
    --region "${REGION}" \
    --repository "${REPOSITORY_NAME}" >/dev/null
  validate_repository_configuration \
    "${repository_uri}" \
    "${account}" \
    "${WORK_DIR}/repository.json"
  require_tag_available \
    "${tag}" \
    "${WORK_DIR}/tag-lookup.json" \
    "${WORK_DIR}/tag-lookup.err"

  log "mode=${MODE}"
  log "aws_profile=${AWS_PROFILE}"
  log "account=${account}"
  log "region=${REGION}"
  log "repository_uri=${repository_uri}"
  log "tag=${tag}"
  log "local_config_id=${config_id}"

  if [[ "${MODE}" == "dry-run" ]]; then
    log "dry_run=true (no login, tag, push, or manifest mutation)"
    return 0
  fi

  local required_confirmation="PUSH ${tag} TO ${account}/${REGION}"
  [[ "${DISTILLERY_PUBLISH_CONFIRM:-}" == "${required_confirmation}" ]] \
    || die "typed confirmation must equal: ${required_confirmation}"

  local remote_tag="${repository_uri}:${tag}"
  [[ "${remote_tag,,}" != *latest* ]] || die "remote tag must not contain latest"
  aws ecr get-login-password \
    --profile "${AWS_PROFILE}" \
    --region "${REGION}" \
    | docker login \
        --username AWS \
        --password-stdin \
        "${account}.dkr.ecr.${REGION}.amazonaws.com" >/dev/null
  docker tag "${image_name}:${tag}" "${remote_tag}"
  docker push "${remote_tag}"

  aws ecr describe-images \
    --profile "${AWS_PROFILE}" \
    --region "${REGION}" \
    --repository-name "${REPOSITORY_NAME}" \
    --image-ids "imageTag=${tag}" \
    --output json >"${WORK_DIR}/pushed-image.json"
  local image_digest
  image_digest="$("${PYTHON}" - "${WORK_DIR}/pushed-image.json" "${tag}" <<'PY'
import json
import re
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected_tag = sys.argv[2]
details = payload.get("imageDetails")
if not isinstance(details, list) or len(details) != 1:
    raise SystemExit("expected one post-push image detail")
tags = details[0].get("imageTags")
if not isinstance(tags, list) or tags != [expected_tag]:
    raise SystemExit("post-push ECR response did not bind the exact immutable tag")
digest = details[0].get("imageDigest")
if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
    raise SystemExit("post-push ECR digest is invalid")
print(digest)
PY
)"
  assert_digest "${image_digest}" "verified ECR content digest"

  wait_for_scan "${tag}" "${WORK_DIR}/scan.json"
  local critical_findings high_findings
  critical_findings="$(scan_count "${WORK_DIR}/scan.json" CRITICAL)"
  high_findings="$(scan_count "${WORK_DIR}/scan.json" HIGH)"

  "${PYTHON}" "${MANIFEST_TOOL}" set-registry \
    --schema "${SCHEMA}" \
    --manifest "${MANIFEST_IN}" \
    --repository-uri "${repository_uri}" \
    --account "${account}" \
    --region "${REGION}" \
    --image-digest "${image_digest}" \
    --scan-status COMPLETE \
    --critical-findings "${critical_findings}" \
    --high-findings "${high_findings}" \
    --max-critical 0 \
    --max-high 0

  log "registry_image_digest=${image_digest}"
  log "digest_uri=${repository_uri}@${image_digest}"
  log "publish complete; manifest registry binding is verified"
}

main "$@"
