# cortec-framework

Two Apache-2.0 reference implementations of **CoRTeC**, cohort-conditioned differentially private
synthetic tabular data from a frozen language model. This repository holds the software; the paper,
the research harness, and the result records every published number is re-derived from live in the
companion repository **[cortec](https://github.com/Calyie/cortec)**.

| package | what it is | tests |
|---|---|---|
| [`cortec/`](cortec/) | The mechanism end to end: a public stratification rule, DP histograms, a DP conditional target table, and generation from those statistics alone by a frozen model. Capability gating, SHA-256 hash-locked prompts, an auditable privacy ledger whose `spend()` is the only source of a noise scale, a per-column coverage guard, class-conditional histogram blocks, exact-count batches, rake-and-refine selection from a generated pool, and an output property test | 127 |
| [`cortec-hybrid/`](cortec-hybrid/) | For institutions that will not put a language model in the data path: release a DP conditional table and relabel an existing marginal synthesiser's output to match it. No model server in the dependency tree | 26 |

Every guarantee the paper establishes is enforced in this code rather than documented, and every
regression test is named after the defect it prevents.

## Deploying it

`cortec/docs/deployment.md` is the operational manual: the controls mapped to NIST SP 800-226,
ISO/IEC 27559 and HIPAA Expert Determination, every operational failure mode with the guardrail
that enforces it, the deployment checklist, the two deployment patterns, and the cost model.

## Install

`cortec-hybrid` depends on `cortec`, so install in this order:

```bash
pip install ./cortec[anthropic]      # Claude, recommended on AWS Bedrock
pip install ./cortec[openai]         # GPT, recommended on Azure OpenAI
pip install ./cortec[gemini]         # Gemini, recommended on Google Vertex AI
pip install ./cortec-hybrid
```

**Where this should run.** CoRTeC is proposed for the **enterprise-hosted, tenant-isolated** model
surfaces inside your own cloud account, Bedrock, Azure OpenAI, Vertex AI, with private networking,
data residency, contractual exclusion of training on inputs, and a BAA where HIPAA applies. The
backends also reach the vendors' public developer APIs, and those are **not recommended for regulated
data**: the weights are the same, the contractual envelope is not, and the envelope is what a
compliance review assesses. (The paper's own measurements used the public APIs, and says so.) The
enterprise surfaces are reached with `Generator(..., surface="bedrock" | "azure" | "vertex")`; the
adapters are tested against the SDKs' real client classes with the transport mocked, **Vertex AI is
verified live**, and Bedrock and Azure OpenAI have **not** been run against a live account, see
`cortec/README.md`, "Backends".

## Privacy

The prompt carries only the DP release, noisy histograms, class balances, a conditional table, and
never a private record, so by post-processing immunity the trust boundary is crossed *before* the
model call, and unlimited synthetic datasets may be drawn from one release at no further privacy cost.

Where the guarantee does not hold (carried in every release audit, per NIST SP 800-226):

- **ε protects one ROW, not one person.** On multi-row-per-person data, cap each person's rows
  before Stage A; the library requires the per-person row bound and refuses a vacuous one.
- **Floating-point Laplace.** The library mechanism is vulnerable to the Mironov (2012) attack.
- **Pretraining provenance.** The guarantee says nothing about the model's training corpus.

## How this code is held to the paper

`cortec/paper/audit/verify_paper_tool_parity.py` reads this tree and the paper together, 173 assertions that the shipped code does what the paper says, packaging included. It locates this
repository via `CORTEC_TOOLS` or a side-by-side clone, and refuses to run without it.

```bash
git clone https://github.com/Calyie/cortec
git clone https://github.com/Calyie/cortec-framework        # beside it
cd cortec && python3 paper/audit/verify_paper_tool_parity.py
```

## Tests

```bash
python3 -m pytest cortec/tests cortec-hybrid/tests -q    # 127 + 26
```

## Citation and license

See [`CITATION.cff`](CITATION.cff); cite the paper for the method and this repository for the
software. Apache-2.0, [`LICENSE`](LICENSE), and each package carries its own copy.
