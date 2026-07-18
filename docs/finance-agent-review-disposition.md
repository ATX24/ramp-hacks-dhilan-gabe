# PR #2 review disposition

Disposition for both NOT CLEAR reviews of commit `7ecf50b`. All implementation
remains isolated from active UI, API, BYODT, and training paths.

1. **Model-visible solvability — resolved.** `AgentModelInput` includes sealed
   public dates, periods, COA, policy windows, merchant/entity IDs, reconciliation
   book/bank IDs, variance targets, system prompt, and canonical tool definitions.
   Validation rejects any gold scalar absent from input or a prior result.
2. **Role-aware objective — resolved.** `technique/tokenization.py` masks system,
   user, tool-result, and padding tokens with `-100`; only assistant messages,
   calls, and final answers are supervised. `mask_tool_results=false` is invalid.
3. **Unproven teacher claims — removed.** Corpus labels are explicitly oracle
   labels. Teacher evidence has a separate exact revision/license/output-use
   contract, but no teacher rollout or model is instantiated or advertised.
4. **Prompt/schema seals — resolved.** Every model input carries canonical schemas,
   prompt/public-world hashes, aggregate tool hash, and complete input hash.
5. **Replay validation — resolved.** `validate_episode` requires the private world,
   checks its latent hash/public projection, replays every call, and compares
   canonical result bytes including provenance.
6. **Ordered metrics — resolved.** Metrics score per-position tool, arguments,
   result bytes, action order, answer bindings, skipped/extra calls, final answer,
   and end-to-end exactness.
7. **Gold boundary/diversity/leakage — resolved.** Writers produce separate
   `model/`, `gold/`, and `oracle/` trees. Leakage checks cover all split pairs for
   identity, input hash, normalized prompts, template families, and semantic
   fingerprints. Prompts use split-disjoint template families and unique entities.
8. **Strict OOD — resolved.** Held-out tools are absent from train, validation,
   test, and IID. They appear only in OOD. Payroll OOD changes account 6400,
   ledger memo, policy, and merchant semantics.
9. **Tool/accounting correctness — resolved.** Temporal policy selection uses
   effective windows; calculator arity is schema-enforced; COA filters use AND;
   duplicate transaction IDs fail softly; variance reports full and returned
   totals; ordinary tool failures are sealed `ToolResult` values. Conflict labels
   compare actual totals to thresholds.
10. **Hard-case honesty — resolved.** Wrong-tool, wrong-argument, and stale-policy
    episodes contain the attempted path and explicit recovery. Ambiguous merchant
    includes an assistant clarification plus a second user turn.
11. **Specialist routing — removed.** The plan targets only the generalist.
    Registry metadata contains no model or specialist entries and no router.
12. **Economics — resolved.** Generated episodes and metrics use null unmeasured
    latency/cost. Proof readiness requires a measured cost artifact.
13. **Proof protocol — resolved as a gated contract.** `finance-agent-proof.v1`
    binds seed, corpus, order, model, tokenizer, chat template, prompt/schema,
    render template, license, and cost. It requires paired ordered evaluation and
    remains honestly `not_ready` while artifacts are null.
14. **Technique isolation/objective — resolved.** `agent_trajectory.v1` has a
    distinct role-masked objective and collator, no logits claim, no alias, and no
    BYODT builtin registration.
15. **Adversarial evidence — resolved.** Tests mutate tool name, arguments, order,
    skipped dependency, extra call, tool-result bytes/provenance, and final value.
    Independent always-refuse and invalid-first-tool baselines must score zero
    end-to-end.

Additional domain-review defects fixed: historical superseded-policy lookup,
actual threshold conflicts, full-period variance, payroll semantics, duplicate
IDs, structured soft errors, and exact no-shell/no-network rejection.
