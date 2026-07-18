#!/usr/bin/env python3
"""Generate the Finance Agent smoke corpus to a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from distillery.finance_agent.generate import generate_agent_corpus
from distillery.finance_agent.metrics import aggregate_metrics, score_episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/finance_agent/fixtures/smoke"),
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    corpus = generate_agent_corpus("smoke", seed=args.seed)
    hashes = corpus.write(args.out)
    metrics = aggregate_metrics([score_episode(example) for example in corpus.examples])
    print(
        {
            "n": corpus.manifest["n_examples"],
            "splits": corpus.manifest["splits"],
            "case_family_counts": corpus.manifest["case_family_counts"],
            "content_sha256": corpus.manifest["content_sha256"],
            "oracle_end_to_end_success_rate": metrics["end_to_end_success_rate"],
            "files": hashes,
        }
    )


if __name__ == "__main__":
    main()
