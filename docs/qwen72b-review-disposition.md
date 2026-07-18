# Qwen72B combined review disposition

Status: implementation complete; execution remains blocked pending re-review.

## Identity, policy, and provenance packet

1. Readiness booleans were replaced with `LiveVerifier` methods and immutable,
   hash-bound evidence. Only a complete live evidence bundle can issue a
   short-lived execution authorization.
2. The materializer exposes no rendered worker. Materialization, probe, and
   training commands run the same live gates and exact typed confirmation.
   Workers revalidate the bound authorization.
3. The exact Qwen license hash, output-use disposition, `QWEN_NOTICE.txt`, and
   attribution plan are executable artifacts. MIT remains the repo code
   license.
4. The adapted fallback uses real `finance_world.v2` envelopes for all four
   tasks, bound to generator revision and `latent_state_hash`. V1 generation is
   unchanged.
5. Teacher trajectories use an explicit `absent` state. Empty bundles cannot
   seal or make the teacher role ready.
6. Closed enums and discriminated role models distinguish the Qwen72B teacher,
   adapted fallback, TinyFable students, and demo arms. Memory fields use model,
   not student, terminology.
7. Tokenizer compatibility compares live file-body, chat-template, aggregate,
   and special-token-ID hashes for every registered model/revision pair.
8. S3 verification streams and hashes every inventory body. ECR verification
   exact-matches account, region, repository, digest, config-body digest, and
   the source/tree/lock/trainer/attention labels.
9. Blocked gates return nonzero. Identity, inventory, license, config,
   tokenizer, chat-template, and special-token hashes are all validated.
10. The documented test command is
    `PYTHONPATH=.:src .venv/bin/python -m pytest tests/qwen72b_fallback -q`.
    Adversarial coverage includes every requested failure.
11. Cost evidence includes active transfer/p4de exposure and disables resource
    launch retries. Stop and terminate paths poll and verify terminal state.

## Execution packet

1. `experiments.qwen72b_fallback.train` performs real QLoRA optimization.
   `scripts/qwen72b/launch.py` submits only a live-authorized SageMaker request
   through the hardened image wrapper.
2. FlashAttention 2 is not packaged or claimed. The exact profile uses SDPA
   math with flash kernels disabled and documents NF4 reproducibility limits.
3. Estimates are planning-only. DDP rehearsal/full authorization requires an
   eight-rank target-device model-load, forward, backward, optimizer-step memory
   probe bound to model, image, and profile.
4. The EC2 materializer has a hash-locked bootstrap, disables `hf_transfer`,
   requires a 250-GiB encrypted delete-on-termination volume and scoped instance
   profile, requires bucket versioning, verifies every uploaded body,
   conditionally merges the manifest with a recorded version ID, removes
   partial uploads, wipes local files, and terminates from both worker and
   coordinator paths.
5. ECR and S3 exact matching are shared with the identity packet.
6. The launcher exposes one physical GPU to each child. Model loading uses only
   `device_map={"": 0}`.
7. Fixed-length collation, deterministic no-duplication sharding, bounded NCCL
   synchronization, parent peer termination, and all-rank acknowledgement are
   implemented.
8. Rank 0 calls PEFT `save_pretrained`; every rank synchronizes; rank 0 reloads
   the adapter onto a fresh base and performs a finite-logit forward probe.
   Invalid safetensors cannot seal.
9. The 60-minute rehearsal explicitly budgets channel verification, load,
   three steps, save, reload, cleanup, and shutdown. Per-rank failure artifacts
   and launcher failure evidence survive failures.
10. Adversarial tests cover wrong image, same-size wrong S3 bytes, the removed
    FlashAttention claim, missing probe, rank death, shape mismatch, corrupt
    adapter, orphan cleanup, and duplicate launch.
11. Materialization and training have one-attempt policies. Ambiguous launch
    failures trigger verified stop/termination before returning an error.

## Exact prerequisites to unblock execution

1. Record two distinct cleared review packet SHA-256 values in
   `execution_bindings.json`.
2. Build the clean reviewed commit with the hardened training Dockerfile,
   publish it to `225989358036.dkr.ecr.us-east-1.amazonaws.com/distillery-training`,
   and seal its exact digest, source revision, source-tree hash, and lock hash.
3. Enable versioning on `distillery-225989358036-us-east-1`. Seal and
   live-verify a private transfer subnet, no-ingress security group,
   x86_64 EBS-backed AMI with Python 3, and a Qwen72B-prefix-scoped instance
   profile.
4. Confirm no g5, 14B/32B transfer, p4de, duplicate, or orphaned Qwen72B
   resources are active.
5. Run the live-gated materializer with its exact typed confirmation. Verify
   all S3 object bodies, both control objects, and the merged materialization
   manifest.
6. Run tokenizer body/ID/chat compatibility probes for every registered target.
7. Run a live-authorized target-profile memory probe, seal its S3 body hash,
   and update `execution_bindings.json` with the exact evidence.
8. Re-run the rehearsal gates. Only an `authorized` report permits the
   three-step launch.
