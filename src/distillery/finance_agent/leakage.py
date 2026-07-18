"""Cross-split identity, prompt-template, semantic, and gold-leakage checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from distillery.contracts.hashing import content_sha256
from distillery.contracts.tasks import SplitName
from distillery.finance_agent.contracts import AgentEpisodeEnvelope, ToolName
from distillery.finance_agent.validate import assert_model_record_is_input_only

_ID_RE = re.compile(
    r"\b(?:ex|world|grp|ent|mrc|led|bok|bnk|drv)_[a-z0-9_-]+\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b[0-9]{4}(?:-[0-9]{2}){1,2}\b")
_NUMBER_RE = re.compile(r"\b[0-9]+\b")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "then",
        "to",
        "use",
        "with",
    }
)


@dataclass(frozen=True)
class AgentLeakageReport:
    ok: bool
    identity_overlaps: tuple[str, ...]
    model_input_hash_overlaps: tuple[str, ...]
    normalized_prompt_overlaps: tuple[str, ...]
    template_family_overlaps: tuple[str, ...]
    semantic_prompt_overlaps: tuple[str, ...]
    held_out_tool_leaks: tuple[str, ...]
    held_out_domain_leaks: tuple[str, ...]
    gold_model_record_leaks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "identity_overlaps": list(self.identity_overlaps),
            "model_input_hash_overlaps": list(self.model_input_hash_overlaps),
            "normalized_prompt_overlaps": list(self.normalized_prompt_overlaps),
            "template_family_overlaps": list(self.template_family_overlaps),
            "semantic_prompt_overlaps": list(self.semantic_prompt_overlaps),
            "held_out_tool_leaks": list(self.held_out_tool_leaks),
            "held_out_domain_leaks": list(self.held_out_domain_leaks),
            "gold_model_record_leaks": list(self.gold_model_record_leaks),
        }


def normalize_prompt(text: str) -> str:
    value = _ID_RE.sub("<id>", text.lower())
    value = _DATE_RE.sub("<date>", value)
    value = _NUMBER_RE.sub("<number>", value)
    return " ".join(_TOKEN_RE.findall(value))


def semantic_prompt_fingerprint(text: str) -> str:
    normalized = normalize_prompt(text)
    tokens = sorted(token for token in _TOKEN_RE.findall(normalized) if token not in _STOPWORDS)
    return content_sha256(tokens)


def _combined_user_text(example: AgentEpisodeEnvelope) -> str:
    return "\n".join(message.text for message in example.model_input.user_messages)


def check_agent_leakage(
    by_split: dict[SplitName, list[AgentEpisodeEnvelope]],
    *,
    held_out_tools: frozenset[ToolName],
    held_out_domains: frozenset[str],
) -> AgentLeakageReport:
    identity_overlaps: list[str] = []
    input_hash_overlaps: list[str] = []
    normalized_overlaps: list[str] = []
    template_overlaps: list[str] = []
    semantic_overlaps: list[str] = []
    split_items = sorted(by_split.items(), key=lambda item: item[0].value)
    for (left_name, left), (right_name, right) in combinations(split_items, 2):
        label = f"{left_name.value}<->{right_name.value}"
        left_identities = {
            *(f"example:{example.example_id}" for example in left),
            *(f"world:{example.world_id}" for example in left),
            *(f"group:{example.group_id}" for example in left),
        }
        right_identities = {
            *(f"example:{example.example_id}" for example in right),
            *(f"world:{example.world_id}" for example in right),
            *(f"group:{example.group_id}" for example in right),
        }
        identity_overlaps.extend(
            f"{label}:{value}" for value in sorted(left_identities & right_identities)
        )
        left_hashes = {example.model_input.model_input_sha256 for example in left}
        right_hashes = {example.model_input.model_input_sha256 for example in right}
        input_hash_overlaps.extend(
            f"{label}:{value}" for value in sorted(left_hashes & right_hashes)
        )
        left_normalized = {
            content_sha256(normalize_prompt(_combined_user_text(example))) for example in left
        }
        right_normalized = {
            content_sha256(normalize_prompt(_combined_user_text(example))) for example in right
        }
        normalized_overlaps.extend(
            f"{label}:{value}" for value in sorted(left_normalized & right_normalized)
        )
        left_templates = {example.provenance.template_family for example in left}
        right_templates = {example.provenance.template_family for example in right}
        template_overlaps.extend(
            f"{label}:{value}" for value in sorted(left_templates & right_templates)
        )
        left_semantic = {
            semantic_prompt_fingerprint(_combined_user_text(example)) for example in left
        }
        right_semantic = {
            semantic_prompt_fingerprint(_combined_user_text(example)) for example in right
        }
        semantic_overlaps.extend(
            f"{label}:{value}" for value in sorted(left_semantic & right_semantic)
        )

    tool_leaks: list[str] = []
    domain_leaks: list[str] = []
    gold_leaks: list[str] = []
    for split, examples in split_items:
        is_ood = split is SplitName.OOD_TEST
        for example in examples:
            model_tools = {definition.name for definition in example.model_input.tools}
            gold_tools = {call.tool for call in example.gold.trajectory.tool_calls()}
            if not is_ood:
                leaked = sorted(
                    (model_tools | gold_tools) & held_out_tools,
                    key=lambda tool: tool.value,
                )
                tool_leaks.extend(
                    f"{split.value}:{example.example_id}:{tool.value}" for tool in leaked
                )
            domain = str(example.model_input.public_world["domain"])
            if not is_ood and domain in held_out_domains:
                domain_leaks.append(f"{split.value}:{example.example_id}:{domain}")
            if is_ood and domain not in held_out_domains:
                domain_leaks.append(
                    f"{split.value}:{example.example_id}:expected_held_out_got_{domain}"
                )
            try:
                assert_model_record_is_input_only(example.model_record())
            except ValueError as exc:
                gold_leaks.append(f"{split.value}:{example.example_id}:{exc}")

    fields = (
        identity_overlaps,
        input_hash_overlaps,
        normalized_overlaps,
        template_overlaps,
        semantic_overlaps,
        tool_leaks,
        domain_leaks,
        gold_leaks,
    )
    return AgentLeakageReport(
        ok=not any(fields),
        identity_overlaps=tuple(sorted(identity_overlaps)),
        model_input_hash_overlaps=tuple(sorted(input_hash_overlaps)),
        normalized_prompt_overlaps=tuple(sorted(normalized_overlaps)),
        template_family_overlaps=tuple(sorted(template_overlaps)),
        semantic_prompt_overlaps=tuple(sorted(semantic_overlaps)),
        held_out_tool_leaks=tuple(sorted(tool_leaks)),
        held_out_domain_leaks=tuple(sorted(domain_leaks)),
        gold_model_record_leaks=tuple(sorted(gold_leaks)),
    )


__all__ = [
    "AgentLeakageReport",
    "check_agent_leakage",
    "normalize_prompt",
    "semantic_prompt_fingerprint",
]
