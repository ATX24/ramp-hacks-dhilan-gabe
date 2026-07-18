#!/usr/bin/env bash
# Distillery inference stack helper. Default: plan-only / dry-run. Never deploys
# unless DISTILLERY_DEPLOY_CONFIRM=YES and --apply are both supplied.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="${ROOT_DIR}/infra/inference/template.yaml"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
STACK_NAME="${DISTILLERY_INFERENCE_STACK_NAME:-distillery-inference-hackathon}"
ENV_NAME="${DISTILLERY_ENVIRONMENT:-hackathon}"

MODE="plan"
for arg in "$@"; do
  case "${arg}" in
    --plan|--dry-run|--preflight) MODE="plan" ;;
    --apply) MODE="apply" ;;
    --help|-h)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "error: unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

log() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

plan() {
  [[ -f "${TEMPLATE}" ]] || die "template not found: ${TEMPLATE}"
  require_cmd python3
  log "== Distillery inference deploy plan (no changes) =="
  log "region=${REGION}"
  log "stack=${STACK_NAME}"
  log "environment=${ENV_NAME}"
  log "template=${TEMPLATE}"
  log "mode=${MODE}"

  python3 - <<'PY' "${TEMPLATE}"
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
required = [
    "AWSTemplateFormatVersion",
    "AWS::SageMaker::Model",
    "AWS::SageMaker::EndpointConfig",
    "AWS::SageMaker::Endpoint",
    "EnableNetworkIsolation: true",
    "AWS::IAM::Role",
    "AWS::Logs::LogGroup",
    "CostCenter",
    "ImageUri",
    "ModelDataUrl",
    "InstanceType",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit("template missing required markers: " + ", ".join(missing))
if "CreateTrainingJob" not in text:
    raise SystemExit("template must deny CreateTrainingJob in the execution role")
if re.search(r"AKIA[0-9A-Z]{16}", text):
    raise SystemExit("template appears to contain an access key id")
print("template_static_checks=ok")
PY

  if [[ -n "${AWS_PROFILE:-}" ]]; then
    require_cmd aws
    identity="$(aws sts get-caller-identity --profile "${AWS_PROFILE}" --region "${REGION}" --output json)"
    arn="$(printf '%s' "${identity}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Arn"])')"
    [[ "${arn}" != *":root" ]] || die "refusing plan/apply with account root identity (${arn})"
    log "caller=${arn}"
    aws cloudformation validate-template \
      --profile "${AWS_PROFILE}" \
      --region "${REGION}" \
      --template-body "file://${TEMPLATE}" \
      >/dev/null
    log "cloudformation_validate_template=ok"
  else
    log "AWS_PROFILE unset; skipped live CloudFormation validate-template"
  fi

  log "remaining_inputs=ImageUri,ModelDataUrl,ArtifactBucketName,ModelDataPrefix,EcrRepositoryArn,KmsKeyArn,AutoDeleteAtUtc"
  log "plan_complete=no_resources_changed"
}

apply() {
  [[ "${DISTILLERY_DEPLOY_CONFIRM:-}" == "YES" ]] \
    || die "refusing --apply without DISTILLERY_DEPLOY_CONFIRM=YES"
  [[ -n "${AWS_PROFILE:-}" ]] || die "AWS_PROFILE is required for --apply"
  for required in \
    DISTILLERY_INFERENCE_IMAGE_URI \
    DISTILLERY_INFERENCE_MODEL_DATA_URL \
    DISTILLERY_ARTIFACT_BUCKET \
    DISTILLERY_INFERENCE_MODEL_PREFIX \
    DISTILLERY_INFERENCE_ECR_REPO_ARN \
    DISTILLERY_INFERENCE_KMS_KEY_ARN \
    DISTILLERY_INFERENCE_AUTO_DELETE_AT_UTC
  do
    [[ -n "${!required:-}" ]] || die "missing required env for apply: ${required}"
  done
  plan
  require_cmd aws
  log "apply_confirmed=true"
  aws cloudformation deploy \
    --profile "${AWS_PROFILE}" \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --template-file "${TEMPLATE}" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides \
      EnvironmentName="${ENV_NAME}" \
      ImageUri="${DISTILLERY_INFERENCE_IMAGE_URI}" \
      ModelDataUrl="${DISTILLERY_INFERENCE_MODEL_DATA_URL}" \
      ArtifactBucketName="${DISTILLERY_ARTIFACT_BUCKET}" \
      ModelDataPrefix="${DISTILLERY_INFERENCE_MODEL_PREFIX}" \
      EcrRepositoryArn="${DISTILLERY_INFERENCE_ECR_REPO_ARN}" \
      KmsKeyArn="${DISTILLERY_INFERENCE_KMS_KEY_ARN}" \
      AutoDeleteAtUtc="${DISTILLERY_INFERENCE_AUTO_DELETE_AT_UTC}" \
      MaxEndpointCostUsd="${DISTILLERY_INFERENCE_MAX_COST_USD:-4.224}" \
      EndpointName="${DISTILLERY_INFERENCE_ENDPOINT_NAME:-distillery-demo-inference}" \
      InstanceType="${DISTILLERY_INFERENCE_INSTANCE_TYPE:-ml.g5.xlarge}"
  aws cloudformation describe-stacks \
    --profile "${AWS_PROFILE}" \
    --region "${REGION}" \
    --stack-name "${STACK_NAME}" \
    --query 'Stacks[0].Outputs' \
    --output json
}

case "${MODE}" in
  plan) plan ;;
  apply) apply ;;
  *) die "unknown mode ${MODE}" ;;
esac
