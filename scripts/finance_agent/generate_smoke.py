#!/usr/bin/env python3
"""Generate the Finance Agent smoke corpus to a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from distillery.finance_agent.baselines import always_refuse_baseline
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
    baseline_metrics = aggregate_metrics(
        [
            score_episode(
                example,
                predicted=always_refuse_baseline(example),
            )
            for example in corpus.examples
        ]
    )
    print(
        {
            "n": corpus.manifest["n_examples"],
            "splits": corpus.manifest["splits"],
            "case_family_counts": corpus.manifest["case_family_counts"],
            "corpus_sha256": corpus.manifest["corpus_sha256"],
            "corpus_order_sha256": corpus.manifest["corpus_order_sha256"],
            "proof_status": corpus.proof_protocol.proof_status,
            "always_refuse_baseline": baseline_metrics,
            "files": hashes,
        }
    )


if __name__ == "__main__":
    main()
