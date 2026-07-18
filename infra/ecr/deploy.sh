#!/usr/bin/env bash
# Distillery training ECR repository deploy helper.
#
# Safety defaults:
#   - Preflight only unless --apply is passed
#   - Requires an account/region-specific typed confirmation for --apply
#   - Refuses root account identity
#   - Never builds/pushes images or submits SageMaker jobs
#
# Usage:
#   ./infra/ecr/deploy.sh
#   AWS_PROFILE=hackathon-builder ./infra/ecr/deploy.sh --preflight

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="${ROOT_DIR}/infra/ecr/template.yaml"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
STACK_NAME="${DISTILLERY_ECR_STACK_NAME:-distillery-ecr-training}"
REPO_NAME="distillery-training"
ACCOUNT=""

if [[ -n "${DISTILLERY_ECR_REPOSITORY_NAME:-}" \
  && "${DISTILLERY_ECR_REPOSITORY_NAME}" != "${REPO_NAME}" ]]; then
  printf 'error: repository is not allowlisted: %s\n' \
    "${DISTILLERY_ECR_REPOSITORY_NAME}" >&2
  exit 1
fi

PROFILE_ARGS=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  PROFILE_ARGS=(--profile "${AWS_PROFILE}")
fi

MODE="preflight"
for arg in "$@"; do
  case "${arg}" in
    --preflight) MODE="preflight" ;;
    --apply) MODE="apply" ;;
    --help|-h)
      sed -n '2,15p' "$0"
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

preflight() {
  require_cmd aws
  require_cmd python3
  [[ -f "${TEMPLATE}" ]] || die "template not found: ${TEMPLATE}"
  [[ -n "${AWS_PROFILE:-}" ]] || die "AWS_PROFILE must select a non-root profile"
  [[ "${REGION}" =~ ^[a-z]{2}(-gov)?-[a-z]+-[1-9][0-9]*$ ]] \
    || die "invalid AWS region: ${REGION}"

  log "== Distillery ECR deploy preflight =="
  log "region=${REGION}"
  log "stack=${STACK_NAME}"
  log "repository=${REPO_NAME}"
  log "mode=${MODE}"

  python3 - <<'PY' "${TEMPLATE}"
import json, re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
template = json.loads(text)
resources = template["Resources"]
repository = resources["TrainingRepository"]
properties = repository["Properties"]
assert repository["Type"] == "AWS::ECR::Repository"
assert repository["DeletionPolicy"] == "Retain"
assert repository["UpdateReplacePolicy"] == "Retain"
assert properties["ImageTagMutability"] == "IMMUTABLE"
assert properties["ImageScanningConfiguration"]["ScanOnPush"] is True
assert properties["EncryptionConfiguration"]["EncryptionType"] == "AES256"
assert "LifecyclePolicy" in properties
assert "TrainingImagePushPolicy" in resources
assert "TrainingImagePullPolicy" in resources
for output in ("RepositoryUri", "PushPolicyArn", "PullPolicyArn"):
    assert output in template["Outputs"]
if re.search(r"arn:aws:iam::\d{12}:", text):
    raise SystemExit("template appears to hardcode an account ID ARN")
if re.search(r"(AKIA[0-9A-Z]{16}|aws_secret_access_key)", text, re.I):
    raise SystemExit("template appears to contain credentials")
print("template static checks: ok")
PY

  local identity arn
  identity="$(aws sts get-caller-identity --region "${REGION}" "${PROFILE_ARGS[@]}" --output json)"
  arn="$(printf '%s' "${identity}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Arn"])')"
  ACCOUNT="$(printf '%s' "${identity}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Account"])')"
  log "caller_arn=${arn}"
  if [[ "${arn}" == *":root" ]]; then
    die "refusing to deploy with account root identity (${arn})"
  fi

  aws cloudformation validate-template \
    --template-body "file://${TEMPLATE}" \
    --region "${REGION}" \
    "${PROFILE_ARGS[@]}" >/dev/null
  log "preflight complete (no resources mutated)"
}

apply() {
  preflight
  local expected_confirmation="CREATE ${REPO_NAME} IN ${ACCOUNT}/${REGION}"
  [[ "${DISTILLERY_ECR_DEPLOY_CONFIRM:-}" == "${expected_confirmation}" ]] || die \
    "typed confirmation must equal: ${expected_confirmation}"

  log "== Applying Distillery ECR stack =="
  aws cloudformation deploy \
    --template-file "${TEMPLATE}" \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    "${PROFILE_ARGS[@]}" \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides \
      "RepositoryName=${REPO_NAME}" \
      "CandidateRetentionDays=${DISTILLERY_ECR_CANDIDATE_RETENTION_DAYS:-30}" \
    --no-fail-on-empty-changeset
}

case "${MODE}" in
  preflight) preflight ;;
  apply) apply ;;
  *) die "unknown mode ${MODE}" ;;
esac
