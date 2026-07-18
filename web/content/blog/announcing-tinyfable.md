# Announcing TinyFable

Companies already have traces. They have policy decisions, accounting work, corrections, and the answers produced by larger models. Most of that material sits in logs while the next request goes back to the same expensive API.

We wanted to know when those traces could become a smaller model, and when they should not.

TinyFable is our answer to the first half of that question. It is one portable finance-generalist model that handles transaction review and variance analysis with the same set of weights. Distillery is the system that answers the second half. It curates the evidence, fills only the labels that are missing, trains the candidate, and tests whether the whole exercise was worthwhile.

Our released experiment reached `proved`. That word has a deliberately narrow meaning. TinyFable cleared the joint quality gate on both primary tasks, held up on the required out-of-distribution slices, and passed the economic gate under the utilization assumptions recorded in its proof report. It does not mean that a synthetic benchmark settles production finance.

## One model, not a bag of demos

TinyFable reviews transactions by choosing a GL treatment, producing a balanced journal entry, applying the relevant policy, citing evidence, and reporting confidence. It also analyzes financial variances by computing profit impact, direction, top drivers, and arithmetic closure.

The same model artifact does both jobs. There is no hidden specialist router and no second checkpoint behind the variance demo. Cash reconciliation is included as a backup task, but it does not replace either primary task in the release claim.

We built a deterministic synthetic finance world for the experiment. The latent facts are generated first. Prompts, source documents, and answers are rendered from those facts. That gives us an oracle for journal balance, policy precedence, variance arithmetic, and reconciliation while keeping customer data out of the hackathon build.

The benchmark is still synthetic. It proves that the pipeline, accounting invariants, leakage controls, and model comparison work under a bounded protocol. It does not prove privacy, production generalization, or customer return on investment.

## Distillation has to beat the obvious alternatives

A successful training job is not a successful product.

We compared TinyFable with rules, its frozen base model, its teacher, a cheap off-the-shelf model, oracle supervised fine-tuning, sequence distillation, logit distillation, and a matched cross-entropy ablation. Every trainable student arm began from the same pinned student revision and used the same token budget, hardware class, decoding policy, and evaluation set.

That comparison matters because the sensible answer is sometimes to keep the rules or call a cheap API. Distillery can return `do_not_distill`, `failed_quality`, `failed_economics`, or `insufficient_evidence`. We count those as useful product outcomes, not embarrassing error states.

For TinyFable, the selected student passed. The exact scores, paired confidence intervals, seed variability, latency, throughput, peak memory, GPU hours, and gross cost live in the immutable proof report. The website reads those values from the report rather than copying them into a marketing chart by hand.

## What Distillery actually does

The public path is short:

```python
distillery = Distillery(api_key=os.environ["DISTILLERY_API_KEY"])
dataset = distillery.datasets.create("./finance_world.jsonl")
run = distillery.distill(dataset, recipe="auto").wait()
```

The work behind those lines is split into four visible stages.

**Curate** imports existing responses, validates them, records provenance, isolates groups and test regimes, and freezes the dataset hashes.

**Synthesize** calls a teacher only when a valid response is missing, rejected, or deliberately added for coverage. Existing traces are not thrown away so that a product can sell more teacher tokens.

**Train** resolves `auto` openly to a supported method. This release implements sequence distillation and exact same-tokenizer logit distillation. A requested method is never silently replaced by an easier one.

**Prove** runs the quality, uncertainty, systems, and economic comparisons. It returns an immutable report with the first failed gate if the candidate loses.

Before training, `plan_distillation()` performs the model, tokenizer, license, memory, leakage, and cost checks without launching a billable job. A sealed manifest maps to one finite SageMaker Training Job. The same manifest can run locally. There is no endpoint, notebook, persistent GPU, or deployment layer hidden in the MVP.

## What made the model smaller

TinyFable starts with a Qwen2.5 1.5B teacher and a Qwen2.5 0.5B student. QLoRA adapts the student efficiently. It does not shrink the teacher. The size difference comes from transferring behavior into a model with fewer parameters.

We tested two forms of transfer. `sequence.v1` trains on complete responses from imported traces, the oracle, or the teacher. `logit.v1` transfers the teacher's full output distribution at completion positions when the tokenizer and chat template match exactly. The logit objective was checked against a direct PyTorch reference and compared with a matched cross-entropy run.

These are established methods. Our claim is not a new distillation loss. The work is in making the method choice explicit, keeping the model portable, and packaging the evidence needed to decide whether the smaller model deserves to exist.

## What we are releasing

The TinyFable release includes the student artifact, tokenizer and chat-template files, the sealed run manifest, dataset and split hashes, prediction files, cost records, evaluation report, and load instructions. We are also publishing three working papers covering the model, the Distillery system, and the release evaluation.

We built Anthropic 2 for a hackathon, and the name is satire. The research question is not. Small models are easy to announce and harder to justify. TinyFable is our attempt to make that justification inspectable.

Smaller models. Proven economics.
