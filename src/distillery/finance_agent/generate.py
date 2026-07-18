"""Deterministic Finance Agent corpus generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from distillery.contracts.hashing import content_sha256, sha256_hex
from distillery.contracts.tasks import Difficulty, LabelSource, SplitName
from distillery.finance_agent.contracts import (
    AgentEpisodeEnvelope,
    AgentProvenance,
    CaseFamily,
    OracleMeta,
    ToolName,
)
from distillery.finance_agent.leakage import AgentLeakageReport, check_agent_leakage
from distillery.finance_agent.oracle import solve_case
from distillery.finance_agent.splits import (
    HELD_OUT_DOMAINS_OOD,
    HELD_OUT_TOOLS_OOD,
    PLANNED_SPLITS,
    SMOKE_SPLITS,
    AgentSplitSpec,
    domains_for_split,
    tools_for_split,
)
from distillery.finance_agent.validate import replay_gold_tools, validate_episode
from distillery.finance_agent.world import build_agent_world

CorpusName = Literal["smoke", "planned"]

_CASE_CYCLE: tuple[CaseFamily, ...] = (
    CaseFamily.HAPPY_PATH,
    CaseFamily.WRONG_TOOL,
    CaseFamily.CORRECT_TOOL_WRONG_ARGS,
    CaseFamily.STALE_POLICY,
    CaseFamily.AMBIGUOUS_MERCHANT,
    CaseFamily.MULTI_STEP_RECONCILIATION,
    CaseFamily.ARITHMETIC_TRAP,
    CaseFamily.CONFLICTING_EVIDENCE,
    CaseFamily.REFUSAL_MISSING_DATA,
)

_TRAIN_SAFE_CASES: frozenset[CaseFamily] = frozenset(
    {
        CaseFamily.HAPPY_PATH,
        CaseFamily.WRONG_TOOL,
        CaseFamily.CORRECT_TOOL_WRONG_ARGS,
        CaseFamily.STALE_POLICY,
        CaseFamily.AMBIGUOUS_MERCHANT,
        CaseFamily.CONFLICTING_EVIDENCE,
        CaseFamily.REFUSAL_MISSING_DATA,
    }
)

_OOD_ONLY_CASES: frozenset[CaseFamily] = frozenset(
    {
        CaseFamily.MULTI_STEP_RECONCILIATION,
        CaseFamily.ARITHMETIC_TRAP,
    }
)


@dataclass(frozen=True)
class AgentCorpusSpec:
    name: CorpusName
    seed: int
    splits: tuple[AgentSplitSpec, ...]

    @property
    def total_examples(self) -> int:
        return sum(split.count for split in self.splits)


CORPUS_SMOKE = AgentCorpusSpec(name="smoke", seed=17, splits=SMOKE_SPLITS)
CORPUS_PLANNED = AgentCorpusSpec(name="planned", seed=17, splits=PLANNED_SPLITS)


@dataclass
class GeneratedAgentCorpus:
    spec: AgentCorpusSpec
    examples: list[AgentEpisodeEnvelope]
    manifest: dict[str, Any]
    leakage: AgentLeakageReport
    by_split: dict[SplitName, list[AgentEpisodeEnvelope]] = field(default_factory=dict)

    def write(self, output_dir: str | Path) -> dict[str, str]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        for split, examples in self.by_split.items():
            path = root / f"{split.value}.jsonl"
            payload = "\n".join(
                json.dumps(example.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
                for example in examples
            )
            if examples:
                payload += "\n"
            path.write_text(payload, encoding="utf-8")
            hashes[path.name] = sha256_hex(payload.encode())
        manifest_body = {**self.manifest, "file_sha256": hashes, "leakage": self.leakage.to_dict()}
        text = json.dumps(manifest_body, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        (root / "manifest.json").write_text(text, encoding="utf-8")
        hashes["manifest.json"] = sha256_hex(text.encode())
        return hashes


def generate_agent_corpus(
    spec: AgentCorpusSpec | CorpusName = "smoke",
    *,
    seed: int | None = None,
    validate: bool = True,
) -> GeneratedAgentCorpus:
    if isinstance(spec, str):
        spec = CORPUS_SMOKE if spec == "smoke" else CORPUS_PLANNED
    if seed is not None:
        spec = AgentCorpusSpec(name=spec.name, seed=seed, splits=spec.splits)

    examples: list[AgentEpisodeEnvelope] = []
    by_split: dict[SplitName, list[AgentEpisodeEnvelope]] = {}
    split_hashes: dict[str, str] = {}
    order_hashes: dict[str, str] = {}
    case_counts: dict[str, int] = {}

    for split_spec in spec.splits:
        generated: list[AgentEpisodeEnvelope] = []
        for index in range(split_spec.count):
            example = _generate_one(
                corpus_seed=spec.seed,
                split_spec=split_spec,
                index=index,
            )
            if validate:
                validate_episode(example)
                world = _world_for_example(spec.seed, split_spec, index, example.case_family)
                replay_gold_tools(world, example)
            generated.append(example)
            family = example.case_family.value
            case_counts[family] = case_counts.get(family, 0) + 1

        ordered = _deterministic_shuffle(generated, spec.seed, split_spec.token)
        by_split[split_spec.name] = ordered
        examples.extend(ordered)
        split_hashes[split_spec.name.value] = content_sha256(
            [ex.model_dump(mode="json") for ex in ordered]
        )
        order_hashes[split_spec.name.value] = sha256_hex(
            ("\n".join(ex.example_id for ex in ordered) + "\n").encode()
        )

    leakage = check_agent_leakage(
        by_split,
        held_out_tools=frozenset(tool.value for tool in HELD_OUT_TOOLS_OOD),
    )
    if not leakage.ok:
        raise ValueError(f"finance agent leakage detected: {leakage.to_dict()}")

    manifest = {
        "schema_version": "finance_agent.v1.corpus_manifest",
        "envelope_schema_version": "finance_agent.v1",
        "technique_id": "agent_trajectory.v1",
        "corpus": spec.name,
        "seed": spec.seed,
        "n_examples": len(examples),
        "splits": {name.value: len(items) for name, items in by_split.items()},
        "case_family_counts": dict(sorted(case_counts.items())),
        "held_out_tools": [tool.value for tool in HELD_OUT_TOOLS_OOD],
        "held_out_domains": list(HELD_OUT_DOMAINS_OOD),
        "split_sha256": split_hashes,
        "order_sha256": order_hashes,
        "content_sha256": content_sha256([ex.model_dump(mode="json") for ex in examples]),
    }
    manifest["manifest_sha256"] = content_sha256(manifest)
    return GeneratedAgentCorpus(
        spec=spec,
        examples=examples,
        manifest=manifest,
        leakage=leakage,
        by_split=by_split,
    )


def _example_seed(corpus_seed: int, token: str, index: int) -> int:
    digest = hashlib.sha256(f"{corpus_seed}:{token}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _select_case(split_spec: AgentSplitSpec, index: int) -> CaseFamily:
    # Train/validation never require held-out tools. Eval splits may cycle all families.
    if split_spec.name in {SplitName.TRAIN, SplitName.VALIDATION}:
        pool = tuple(case for case in _CASE_CYCLE if case in _TRAIN_SAFE_CASES)
    else:
        pool = tuple(_CASE_CYCLE)
    return pool[index % len(pool)]


def _world_for_example(
    corpus_seed: int,
    split_spec: AgentSplitSpec,
    index: int,
    case_family: CaseFamily,
):
    seed = _example_seed(corpus_seed, split_spec.token, index)
    allowed_domains = domains_for_split(split_spec)
    if split_spec.hold_out_domains:
        domain = HELD_OUT_DOMAINS_OOD[index % len(HELD_OUT_DOMAINS_OOD)]
    elif allowed_domains:
        domain = allowed_domains[seed % len(allowed_domains)]
    else:
        domain = None
    return build_agent_world(
        seed=seed,
        index=index,
        domain=domain,
        ambiguous_merchant=case_family is CaseFamily.AMBIGUOUS_MERCHANT,
        stale_policy=case_family is CaseFamily.STALE_POLICY
        or case_family is CaseFamily.WRONG_TOOL
        or case_family is CaseFamily.CONFLICTING_EVIDENCE,
        missing_fields=("merchant_id",) if case_family is CaseFamily.REFUSAL_MISSING_DATA else (),
    )


def _generate_one(
    *,
    corpus_seed: int,
    split_spec: AgentSplitSpec,
    index: int,
) -> AgentEpisodeEnvelope:
    case_family = _select_case(split_spec, index)
    world = _world_for_example(corpus_seed, split_spec, index, case_family)
    available = tools_for_split(split_spec)
    # OOD episodes that need held-out tools must include them.
    if case_family in _OOD_ONLY_CASES:
        available = tuple(ToolName)
    oracle = solve_case(world, case_family=case_family, available_tools=available)
    difficulty = (
        Difficulty.HARD
        if case_family
        in {
            CaseFamily.MULTI_STEP_RECONCILIATION,
            CaseFamily.ARITHMETIC_TRAP,
            CaseFamily.CONFLICTING_EVIDENCE,
            CaseFamily.AMBIGUOUS_MERCHANT,
        }
        else Difficulty.MEDIUM
        if case_family is not CaseFamily.HAPPY_PATH
        else Difficulty.EASY
    )
    id_key = f"{world.world_id}:{index}:{case_family.value}"
    example_id = f"ex_{hashlib.sha256(id_key.encode()).hexdigest()[:18]}"
    held_out_tools = HELD_OUT_TOOLS_OOD if split_spec.hold_out_tools else ()
    held_out_domains = HELD_OUT_DOMAINS_OOD if split_spec.hold_out_domains else ()
    # Latency/cost estimates are deterministic functions of tool-call count.
    n_calls = len(oracle.expected_output.gold_tool_calls)
    latency = 40 + n_calls * 35
    cost = 100 + n_calls * 250
    return AgentEpisodeEnvelope(
        example_id=example_id,
        world_id=world.world_id,
        group_id=world.group_id,
        difficulty=difficulty,
        case_family=case_family,
        user_goal=oracle.user_goal,
        available_tools=oracle.available_tools,
        trajectory=oracle.trajectory,
        expected_output=oracle.expected_output,
        oracle=OracleMeta(latent_state_hash=world.latent_state_hash()),
        provenance=AgentProvenance(
            split=split_spec.name,
            case_family=case_family,
            template_family=oracle.template_family,
            label_source=LabelSource.ORACLE,
            held_out_tools=held_out_tools,
            held_out_domains=held_out_domains,
        ),
        estimated_latency_ms=latency,
        estimated_cost_usd_micros=cost,
    )


def _deterministic_shuffle(
    examples: list[AgentEpisodeEnvelope],
    seed: int,
    token: str,
) -> list[AgentEpisodeEnvelope]:
    keyed = []
    for example in examples:
        digest = hashlib.sha256(f"{seed}:{token}:{example.example_id}".encode()).hexdigest()
        keyed.append((digest, example))
    keyed.sort(key=lambda item: item[0])
    return [example for _, example in keyed]
