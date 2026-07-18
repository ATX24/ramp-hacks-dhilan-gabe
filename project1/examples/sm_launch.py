"""Launch the fast SageMaker training job on ml.g5.12xlarge (best GPU quota
in this account). Packages sm_train.py + requirements, submits, streams status.

Usage: PYTHONPATH=.:src .venv/bin/python examples/sm_launch.py
"""
import io
import json
import tarfile
import time
from pathlib import Path

import boto3

BUCKET = "proof-ramp-hackathon-225989358036"
ROLE = "arn:aws:iam::225989358036:role/ProofSageMakerRole"
IMAGE = ("763104351884.dkr.ecr.us-east-1.amazonaws.com/"
         "huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04")
JOB = f"proof-fable-distill-{int(time.time())}"

session = boto3.Session(profile_name="ramp-hackathon", region_name="us-east-1")
s3, sm = session.client("s3"), session.client("sagemaker")

# package source: sm_train.py + requirements (transformers new enough for Qwen2.5)
buf = io.BytesIO()
here = Path(__file__).parent
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    tar.add(here / "sm_train.py", arcname="sm_train.py")
    req = io.BytesIO(b"transformers==4.46.3\npeft==0.13.2\naccelerate>=0.34\n")
    info = tarfile.TarInfo("requirements.txt")
    info.size = len(req.getvalue())
    tar.addfile(info, req)
buf.seek(0)
s3.put_object(Bucket=BUCKET, Key="sagemaker/src/sourcedir.tar.gz", Body=buf.read())

sm.create_training_job(
    TrainingJobName=JOB,
    RoleArn=ROLE,
    AlgorithmSpecification={"TrainingImage": IMAGE, "TrainingInputMode": "File"},
    HyperParameters={
        "sagemaker_program": "sm_train.py",
        "sagemaker_submit_directory": f"s3://{BUCKET}/sagemaker/src/sourcedir.tar.gz",
    },
    Environment={"SM_HP_MAX_STEPS": "200"},
    InputDataConfig=[{
        "ChannelName": "train",
        "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix",
            "S3Uri": f"s3://{BUCKET}/sagemaker/fable/",
            "S3DataDistributionType": "FullyReplicated"}},
    }],
    OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/sagemaker/out/"},
    ResourceConfig={"InstanceType": "ml.g5.12xlarge", "InstanceCount": 1,
                    "VolumeSizeInGB": 50},
    StoppingCondition={"MaxRuntimeInSeconds": 1800},
    Tags=[{"Key": "Project", "Value": "RampHackathon"},
          {"Key": "Owner", "Value": "Dhilan"},
          {"Key": "TTL", "Value": "2026-07-20"}],
)
print("submitted:", JOB)
while True:
    d = sm.describe_training_job(TrainingJobName=JOB)
    print(d["TrainingJobStatus"], d.get("SecondaryStatus"), flush=True)
    if d["TrainingJobStatus"] in ("Completed", "Failed", "Stopped"):
        print(json.dumps({"status": d["TrainingJobStatus"],
                          "failure": d.get("FailureReason"),
                          "model": d.get("ModelArtifacts", {}).get("S3ModelArtifacts")},
                         indent=2))
        break
    time.sleep(20)
