"""Deterministic Finance Agent corpus generation with separated input/gold artifacts."""

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
    AgentGold,
    AgentModelInput,
    AgentProvenance,
    CaseFamily,
    EconomicsObservation,
    OracleMeta,
)
from distillery.finance_agent.leakage import (
    AgentLeakageReport,
    check_agent_leakage,
    semantic_prompt_fingerprint,
)
from distillery.finance_agent.oracle import solve_case
from distillery.finance_agent.prompts import (
    build_prompt_plan,
    build_system_prompt,
)
from distillery.finance_agent.proof import (
    FinanceAgentProofBindings,
    FinanceAgentProofProtocol,
)
from distillery.finance_agent.splits import (
    HELD_OUT_DOMAINS_OOD,
    HELD_OUT_TOOLS_OOD,
    PLANNED_SPLITS,
    SMOKE_SPLITS,
    AgentSplitSpec,
    case_pool_for_split,
    domains_for_split,
    tools_for_split,
)
from distillery.finance_agent.technique.tokenization import (
    TRAJECTORY_RENDER_TEMPLATE_SHA256,
)
from distillery.finance_agent.tools import tool_definitions
from distillery.finance_agent.validate import validate_episode
from distillery.finance_agent.world import AgentWorld, build_agent_world

CorpusName = Literal["smoke", "planned"]


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
    worlds: dict[str, AgentWorld]
    manifest: dict[str, Any]
    proof_protocol: FinanceAgentProofProtocol
    leakage: AgentLeakageReport
    by_split: dict[SplitName, list[AgentEpisodeEnvelope]] = field(default_factory=dict)

    def write(self, output_dir: str | Path) -> dict[str, str]:
        """Write disjoint model, gold, and private-oracle directories."""
        root = Path(output_dir)
        model_root = root / "model"
        gold_root = root / "gold"
        oracle_root = root / "oracle"
        for directory in (model_root, gold_root, oracle_root):
            directory.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        for split, examples in self.by_split.items():
            model_text = _jsonl(example.model_record() for example in examples)
            gold_text = _jsonl(example.gold_record() for example in examples)
            model_path = model_root / f"{split.value}.jsonl"
            gold_path = gold_root / f"{split.value}.jsonl"
            model_path.write_text(model_text, encoding="utf-8")
            gold_path.write_text(gold_text, encoding="utf-8")
            hashes[str(model_path.relative_to(root))] = sha256_hex(model_text.encode())
            hashes[str(gold_path.relative_to(root))] = sha256_hex(gold_text.encode())
        world_text = _jsonl(
            {
                **self.worlds[world_id].private_payload(),
                "latent_state_hash": self.worlds[world_id].latent_state_hash(),
            }
            for world_id in sorted(self.worlds)
        )
        world_path = oracle_root / "worlds.jsonl"
        world_path.write_text(world_text, encoding="utf-8")
        hashes[str(world_path.relative_to(root))] = sha256_hex(world_text.encode())
        proof_text = (
            json.dumps(
                self.proof_protocol.model_dump(mode="json"),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        proof_path = root / "proof_protocol.json"
        proof_path.write_text(proof_text, encoding="utf-8")
        hashes[proof_path.name] = sha256_hex(proof_text.encode())
        manifest_body = {
            **self.manifest,
            "file_sha256": hashes,
            "access_boundaries": {
                "model/": "input_only_model_and_inference_access",
                "gold/": "trainer_and_evaluator_only",
                "oracle/": "generator_and_replay_validator_only",
            },
            "leakage": self.leakage.to_dict(),
        }
        manifest_body["written_manifest_sha256"] = content_sha256(manifest_body)
        manifest_text = (
            json.dumps(manifest_body, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(manifest_text, encoding="utf-8")
        hashes[manifest_path.name] = sha256_hex(manifest_text.encode())
        return hashes


def _jsonl(rows: Any) -> str:
    serialized = [
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for row in rows
    ]
    return "\n".join(serialized) + ("\n" if serialized else "")


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
    worlds: dict[str, AgentWorld] = {}
    by_split: dict[SplitName, list[AgentEpisodeEnvelope]] = {}
    model_hashes: dict[str, str] = {}
    gold_hashes: dict[str, str] = {}
    order_hashes: dict[str, str] = {}
    case_counts: dict[str, int] = {}

    for split_spec in spec.splits:
        generated: list[tuple[AgentEpisodeEnvelope, AgentWorld]] = []
        for index in range(split_spec.count):
            example, world = _generate_one(
                corpus_seed=spec.seed,
                split_spec=split_spec,
                index=index,
            )
            if validate:
                validate_episode(example, world=world)
            generated.append((example, world))
            case = example.case_family.value
            case_counts[case] = case_counts.get(case, 0) + 1
        ordered_pairs = _deterministic_shuffle(
            generated,
            seed=spec.seed,
            token=split_spec.token,
        )
        ordered = [example for example, _ in ordered_pairs]
        by_split[split_spec.name] = ordered
        examples.extend(ordered)
        for example, world in ordered_pairs:
            if example.world_id in worlds:
                raise ValueError(f"duplicate generated world_id {example.world_id}")
            worlds[example.world_id] = world
        model_hashes[split_spec.name.value] = content_sha256(
            [example.model_record() for example in ordered]
        )
        gold_hashes[split_spec.name.value] = content_sha256(
            [example.gold_record() for example in ordered]
        )
        order_hashes[split_spec.name.value] = sha256_hex(
            ("\n".join(example.example_id for example in ordered) + "\n").encode()
        )

    leakage = check_agent_leakage(
        by_split,
        held_out_tools=frozenset(HELD_OUT_TOOLS_OOD),
        held_out_domains=frozenset(HELD_OUT_DOMAINS_OOD),
    )
    if not leakage.ok:
        raise ValueError(f"finance agent leakage detected: {leakage.to_dict()}")

    corpus_sha256 = content_sha256([example.model_dump(mode="json") for example in examples])
    corpus_order_sha256 = content_sha256(order_hashes)
    system_prompt_set_sha256 = content_sha256(
        sorted({example.model_input.system_prompt_sha256 for example in examples})
    )
    tool_schema_set_sha256 = content_sha256(
        sorted({example.model_input.tool_schemas_sha256 for example in examples})
    )
    world_set_sha256 = content_sha256(
        [
            {
                "world_id": world_id,
                "latent_state_hash": worlds[world_id].latent_state_hash(),
            }
            for world_id in sorted(worlds)
        ]
    )
    proof_bindings = FinanceAgentProofBindings(
        seed=spec.seed,
        corpus_sha256=corpus_sha256,
        corpus_order_sha256=corpus_order_sha256,
        system_prompt_set_sha256=system_prompt_set_sha256,
        tool_schema_set_sha256=tool_schema_set_sha256,
        trajectory_render_template_sha256=TRAJECTORY_RENDER_TEMPLATE_SHA256,
    )
    proof_protocol = FinanceAgentProofProtocol.seal(bindings=proof_bindings)
    manifest: dict[str, Any] = {
        "schema_version": "finance_agent.corpus_manifest.v2",
        "envelope_schema_version": "finance_agent.v1",
        "technique_id": "agent_trajectory.v1",
        "technique_integration": "isolated_pending_review",
        "label_source_counts": {"oracle": len(examples), "teacher": 0},
        "teacher_rollout_artifact_sha256": None,
        "corpus": spec.name,
        "seed": spec.seed,
        "n_examples": len(examples),
        "splits": {name.value: len(items) for name, items in by_split.items()},
        "case_family_counts": dict(sorted(case_counts.items())),
        "held_out_tools": [tool.value for tool in HELD_OUT_TOOLS_OOD],
        "held_out_domains": list(HELD_OUT_DOMAINS_OOD),
        "model_input_split_sha256": model_hashes,
        "gold_split_sha256": gold_hashes,
        "order_sha256": order_hashes,
        "corpus_order_sha256": corpus_order_sha256,
        "corpus_sha256": corpus_sha256,
        "latent_world_set_sha256": world_set_sha256,
        "system_prompt_set_sha256": system_prompt_set_sha256,
        "tool_schema_set_sha256": tool_schema_set_sha256,
        "trajectory_render_template_sha256": TRAJECTORY_RENDER_TEMPLATE_SHA256,
        "model_id": None,
        "model_revision": None,
        "model_artifact_sha256": None,
        "tokenizer_sha256": None,
        "chat_template_sha256": None,
        "license_disposition": "unknown",
        "cost_disposition": "unknown",
        "proof_protocol_sha256": proof_protocol.protocol_sha256,
        "proof_status": proof_protocol.proof_status,
    }
    manifest["manifest_sha256"] = content_sha256(manifest)
    return GeneratedAgentCorpus(
        spec=spec,
        examples=examples,
        worlds=worlds,
        manifest=manifest,
        proof_protocol=proof_protocol,
        leakage=leakage,
        by_split=by_split,
    )


def _example_seed(corpus_seed: int, token: str, index: int) -> int:
    digest = hashlib.sha256(f"{corpus_seed}:{token}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _generate_one(
    *,
    corpus_seed: int,
    split_spec: AgentSplitSpec,
    index: int,
) -> tuple[AgentEpisodeEnvelope, AgentWorld]:
    seed = _example_seed(corpus_seed, split_spec.token, index)
    case_pool = case_pool_for_split(split_spec)
    case_family = case_pool[index % len(case_pool)]
    domains = domains_for_split(split_spec)
    domain = domains[seed % len(domains)]
    world = build_agent_world(
        seed=seed,
        index=index,
        domain=domain,
        ambiguous_merchant=case_family is CaseFamily.AMBIGUOUS_MERCHANT,
        policy_conflict=case_family is CaseFamily.CONFLICTING_EVIDENCE,
        missing_fields=(
            ("merchant_id", "amount_minor")
            if case_family is CaseFamily.REFUSAL_MISSING_DATA
            else ()
        ),
    )
    available_tools = tools_for_split(split_spec)
    variant = split_spec.template_variant_offset + (index % 32)
    prompt_plan = build_prompt_plan(
        case_family=case_family,
        world=world,
        variant=variant,
    )
    identifier_material = f"{world.world_id}:{split_spec.name.value}:{index}:{case_family.value}"
    example_id = f"ex_{hashlib.sha256(identifier_material.encode()).hexdigest()[:18]}"
    definitions = tool_definitions(available_tools)
    public_world = world.public_view()
    model_input = AgentModelInput.seal(
        example_id=example_id,
        system_prompt=build_system_prompt(
            public_world=public_world,
            tools=definitions,
        ),
        public_world=public_world,
        user_messages=prompt_plan.user_messages,
        tools=definitions,
    )
    oracle_episode = solve_case(
        world,
        case_family=case_family,
        available_tools=available_tools,
        prompt_plan=prompt_plan,
    )
    gold = AgentGold.seal(
        trajectory=oracle_episode.trajectory,
        expected_output=oracle_episode.expected_output,
        oracle=OracleMeta(
            latent_state_hash=world.latent_state_hash(),
            label_source=LabelSource.ORACLE,
        ),
    )
    hard_cases = {
        CaseFamily.MULTI_STEP_RECONCILIATION,
        CaseFamily.ARITHMETIC_TRAP,
        CaseFamily.CONFLICTING_EVIDENCE,
        CaseFamily.AMBIGUOUS_MERCHANT,
        CaseFamily.WRONG_TOOL,
        CaseFamily.CORRECT_TOOL_WRONG_ARGS,
        CaseFamily.STALE_POLICY,
    }
    difficulty = (
        Difficulty.HARD
        if case_family in hard_cases
        else Difficulty.EASY
        if case_family is CaseFamily.HAPPY_PATH
        else Difficulty.MEDIUM
    )
    user_text = "\n".join(message.text for message in prompt_plan.user_messages)
    example = AgentEpisodeEnvelope.seal(
        example_id=example_id,
        world_id=world.world_id,
        group_id=world.group_id,
        difficulty=difficulty,
        case_family=case_family,
        model_input=model_input,
        gold=gold,
        provenance=AgentProvenance(
            split=split_spec.name,
            case_family=case_family,
            template_family=prompt_plan.template_family,
            label_source=LabelSource.ORACLE,
            scenario_fingerprint=semantic_prompt_fingerprint(user_text),
            held_out_tools=HELD_OUT_TOOLS_OOD if split_spec.ood else (),
            held_out_domains=HELD_OUT_DOMAINS_OOD if split_spec.ood else (),
        ),
        economics=EconomicsObservation(),
    )
    return example, world


def _deterministic_shuffle(
    items: list[tuple[AgentEpisodeEnvelope, AgentWorld]],
    *,
    seed: int,
    token: str,
) -> list[tuple[AgentEpisodeEnvelope, AgentWorld]]:
    keyed = [
        (
            hashlib.sha256(f"{seed}:{token}:{example.example_id}".encode()).hexdigest(),
            example,
            world,
        )
        for example, world in items
    ]
    keyed.sort(key=lambda item: item[0])
    return [(example, world) for _, example, world in keyed]


__all__ = [
    "CORPUS_PLANNED",
    "CORPUS_SMOKE",
    "AgentCorpusSpec",
    "GeneratedAgentCorpus",
    "generate_agent_corpus",
]
