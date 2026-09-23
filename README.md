# cortec-framework

Two Apache-2.0 reference implementations of **CoRTeC**, cohort-conditioned differentially private
synthetic tabular data from a frozen language model. This repository holds the software. The paper
and the research harness live in the companion repository [cortec](https://github.com/Calyie/cortec);
the per-draw result records every published number is re-derived from are retained by the authors
and available on request.

| Package | What it is | Tests |
|---|---|---|
| [`cortec/`](cortec/) | The mechanism end to end: a public stratification rule, DP histograms, a DP conditional target table, and generation from those statistics alone by a frozen model, with a utility transmission bound (Stage C) for the release package | 136 |
| [`cortec-hybrid/`](cortec-hybrid/) | For institutions that will not put a language model in the data path: release a DP conditional table and relabel an existing marginal synthesiser's output to match it. No model server in the dependency tree | 26 |

Every guarantee the paper establishes is enforced in this code rather than documented, and every
regression test is named after the defect it prevents: capability gating, SHA-256 hash-locked
prompts, an auditable privacy ledger whose `spend()` is the only source of a noise scale, a
per-column coverage guard, class-conditional histogram blocks, exact-count batches, rake-and-refine
selection from a generated pool, and an output property test.

## Start here

1. [`cortec/README.md`](cortec/README.md) is the setup guide: install, choose the surface and set
   the credentials, prepare the data, declare the schema, run Stage A, Stage B and Stage C,
   evaluate, troubleshoot.
2. [`cortec/docs/deployment.md`](cortec/docs/deployment.md) is the operational manual for a
   regulated deployment: the architecture, the checklist, the controls mapped to NIST SP 800-226,
   ISO/IEC 27559 and HIPAA Expert Determination, the failure modes with the guardrail for each, the
   two deployment patterns, and the cost model.
3. [`cortec/docs/design-notes.md`](cortec/docs/design-notes.md) records the measurements behind
   the defaults.
4. [`cortec-hybrid/README.md`](cortec-hybrid/README.md) is the setup guide for the hybrid.

## Install

The packages are not on PyPI. `cortec-hybrid` depends on `cortec`, so install in this order from a
clone of this repository:

```bash
pip install './cortec[anthropic]'      # Claude, recommended on AWS Bedrock
pip install './cortec[openai]'         # GPT, recommended on Azure OpenAI
pip install './cortec[gemini]'         # Gemini, recommended on Google Vertex AI
pip install ./cortec-hybrid
python3 -m pytest cortec/tests cortec-hybrid/tests -q    # 136 + 26, offline
```

**Where this should run.** CoRTeC is proposed for the enterprise-hosted, tenant-isolated model
surfaces inside your own cloud account (Bedrock, Azure OpenAI or Vertex AI), with private
networking, data residency, contractual exclusion of training on inputs, and a BAA where HIPAA
applies. The backends also reach the vendors' public developer APIs, and those are not recommended
for regulated data: the weights are the same, the contractual envelope is not, and the envelope is
what a compliance review assesses. The paper's own measurements used the public APIs, and it says
so. Vertex AI is verified live; Bedrock and Azure OpenAI have not been run against a live account.
The setup guide's step 2 has the details.

## Privacy

The prompt carries only the DP release (noisy histograms, class balances and a conditional table)
and never a private record. By post-processing immunity the trust boundary is crossed before the
model call, and unlimited synthetic datasets may be drawn from one release at no further privacy
cost.

Where the guarantee does not hold, carried in every release audit as NIST SP 800-226 asks:

- **ε protects one ROW, not one person.** On multi-row-per-person data, cap each person's rows
  before Stage A. The library requires the per-person row bound and refuses a vacuous one.
- **Floating-point Laplace.** The library mechanism is vulnerable to the Mironov (2012) attack.
- **Pretraining provenance.** The guarantee says nothing about the model's training corpus.

Stage C (`cortec.bound`) adds a utility transmission bound to a release package: a simultaneous,
budgeted bound on how far the synthetic conditional rates can sit from the private ones, with a
real-sample ceiling and a permuted floor that must disagree before a verdict is issued. It bounds
utility only. It is not a privacy audit, and nothing in its report is presented as a privacy
guarantee.

## How this code is held to the paper

An audit script in the research project reads this tree and the technical report together and makes
182 checks that the shipped code does what the paper says, packaging included. It runs before every
change to these packages is published. It belongs to the project's internal audit suite, which is
available from the authors on request.

## Citation and licence

See [`CITATION.cff`](CITATION.cff): cite the paper for the method and this repository for the
software. The licence is Apache-2.0, [`LICENSE`](LICENSE); each package carries its own copy.
