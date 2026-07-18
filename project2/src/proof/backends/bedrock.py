"""Bedrock backend: Nova Pro teacher, Nova Micro/Lite students.

sample/judge use the Converse API. train() submits a managed model
customization job (DISTILLATION) with data staged to S3. Same primitive
surface as the mock backend, so every recipe runs unchanged.
"""
from __future__ import annotations

import json
import time
from typing import Any

import boto3

from proof.backends.base import RecipeIncompatible, TrainedModel
from proof.metrics import REQUIRED_FIELDS

TEACHER = "amazon.nova-pro-v1:0"
STUDENTS = {"amazon:nova-micro": "amazon.nova-micro-v1:0:128k",
            "amazon:nova-lite": "amazon.nova-lite-v1:0:300k"}

# On-demand us-east-1 $/1k tokens (input+output blended, approx) for break-even math.
_COST = {"amazon.nova-pro-v1:0": 0.0032, "amazon.nova-micro-v1:0": 0.00014,
         "amazon.nova-lite-v1:0": 0.00024}

SYSTEM = (
    "Normalize the card transaction descriptor into JSON with exactly these keys: "
    f"{list(REQUIRED_FIELDS)}. category is one of software|travel|meals|office|"
    "advertising|infrastructure|professional_services|other. Respond with JSON only."
)

TAGS = [{"key": "Project", "value": "RampHackathon"},
        {"key": "Owner", "value": "Dhilan"},
        {"key": "TTL", "value": "2026-07-20"}]


class BedrockBackend:
    name = "bedrock"

    def __init__(self, region: str = "us-east-1", profile: str | None = None,
                 role_arn: str = "", bucket: str = ""):
        session = boto3.Session(profile_name=profile, region_name=region)
        self.rt = session.client("bedrock-runtime")
        self.ctl = session.client("bedrock")
        self.s3 = session.client("s3")
        self.role_arn = role_arn
        self.bucket = bucket

    async def sample(self, model: str, prompts: list[str]) -> list[dict | None]:
        out: list[dict | None] = []
        for p in prompts:
            resp = self.rt.converse(
                modelId=model,
                system=[{"text": SYSTEM}],
                messages=[{"role": "user", "content": [{"text": p}]}],
                inferenceConfig={"maxTokens": 300, "temperature": 0},
            )
            text = resp["output"]["message"]["content"][0]["text"]
            try:
                start, end = text.index("{"), text.rindex("}") + 1
                out.append(json.loads(text[start:end]))
            except (ValueError, json.JSONDecodeError):
                out.append(None)
        return out

    async def logprobs(self, model: str, prompt: str, completion: dict) -> float:
        raise RecipeIncompatible(
            "bedrock converse API does not expose token logprobs; "
            "logit-level KD requires the SageMaker backend"
        )

    async def judge(self, rubric: str, items: list[dict]) -> list[bool]:
        verdicts = []
        for it in items:
            resp = self.rt.converse(
                modelId=TEACHER,
                system=[{"text": f"You are a strict data-quality judge. Rubric: {rubric}. "
                                 "Answer only YES or NO."}],
                messages=[{"role": "user", "content": [{"text": json.dumps(it)}]}],
                inferenceConfig={"maxTokens": 5, "temperature": 0},
            )
            verdicts.append("YES" in resp["output"]["message"]["content"][0]["text"].upper())
        return verdicts

    async def train(self, student_base: str, records: list[dict], config: dict[str, Any]) -> TrainedModel:
        """Submit a Bedrock managed distillation job. Asynchronous on AWS's
        side; poll_job() reports status. Requires >=100 prompts (account quota)."""
        student_id = STUDENTS.get(student_base, student_base)
        stamp = int(time.time())
        key = f"proof/distillation/{stamp}/train.jsonl"
        lines = [json.dumps({"schemaVersion": "bedrock-conversation-2024",
                             "system": [{"text": SYSTEM}],
                             "messages": [
                                 {"role": "user", "content": [{"text": r["input"]}]},
                             ]})
                 for r in records]
        self.s3.put_object(Bucket=self.bucket, Key=key, Body="\n".join(lines).encode())

        job = self.ctl.create_model_customization_job(
            jobName=f"proof-distill-{stamp}",
            customModelName=f"proof-student-{stamp}",
            roleArn=self.role_arn,
            baseModelIdentifier=student_id,
            customizationType="DISTILLATION",
            customizationConfig={"distillationConfig": {"teacherModelConfig": {
                "teacherModelIdentifier": config.get("teacher", TEACHER),
                "maxResponseLengthForInference": 300}}},
            trainingDataConfig={"s3Uri": f"s3://{self.bucket}/{key}"},
            outputDataConfig={"s3Uri": f"s3://{self.bucket}/proof/distillation/{stamp}/out/"},
            jobTags=TAGS, customModelTags=TAGS,
        )
        return TrainedModel(ref=job["jobArn"], base=student_id, cost_usd=0.0)

    def poll_job(self, job_arn: str) -> dict:
        j = self.ctl.get_model_customization_job(jobIdentifier=job_arn)
        return {"status": j["status"], "model_arn": j.get("outputModelArn"),
                "failure": j.get("failureMessage")}

    def cost_per_call(self, model: str) -> float:
        base = model.split(":0")[0] + ":0" if ":0" in model else model
        for k, v in _COST.items():
            if k in model or k in base:
                return v * 0.4  # ~400 blended tokens per call
        return _COST["amazon.nova-micro-v1:0"] * 0.4
