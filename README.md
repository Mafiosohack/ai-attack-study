# Multimodal Prompt Injection Study

A security-research study series on **multimodal prompt injection** against
vision-language models (VLMs), built as a reproducible measurement harness rather
than a collection of one-off scripts. Each "system" is a self-owned vulnerable app
with a different trust boundary; every attack on it is measured as an **Attack
Success Rate (ASR) over N trials**, with full provenance saved for every run.

> **Authorized, defensive/educational research.** Every attack here targets a
> locally built, self-owned vulnerable app on `localhost`. The goal is to
> *measure* how readily a fixed VLM victim is steered by crafted input, and how
> much named defenses actually help — so the failure modes can be understood and
> mitigated. See [Scope & ethics](#scope--ethics).

## The systems

| System | Trust boundary | Victim app | Status |
|--------|----------------|------------|--------|
| [**System 1 — direct**](systems/system1_direct/) | The user uploads the malicious image themselves | Image-description assistant | ✅ measured |
| [**System 2 — indirect**](systems/system2_indirect/) | The malicious content arrives through **retrieval**; the user never sees it | Hiring assistant that fetches a candidate's own page | ✅ measured, one defense layer open |
| System 3 — agentic | Injection steers an assistant that can take actions | — | ⏳ not started |

Headline results (victim: Qwen2.5-VL-3B, 4-bit; sampled at temperature 0.7):

- **System 1:** a readable-text image injection succeeds **50/50** with no
  defense, **50/50** with a naive prompt defense and **49/50** with a strong
  spotlighting prompt. Prompt-only defenses don't help.
- **System 2, model decides:** the same kind of injection, retrieved into a
  hiring decision, flips the verdict far less often: **30%** (text) and **20%**
  (plain image) for an "HR pre-verified, shortlist" override, against a clean
  baseline of 2/100. Greedy decoding showed 0% — only sampling revealed it.
- **System 2, authority removed** (model fills a fact schema, Python decides):
  the override drops to **0/20** everywhere, but fake credentials in a plain
  image get shortlisted **14/20 (70%)**. The attack moves from instructions to
  facts.

Each system's README has the full tables.

## How the repo is organized

Both systems use the same skeleton, so once you know one you can read the other:

```
harness/                      shared measurement core — used by every system
  model.py                    ONE pinned 4-bit victim loader (Qwen2.5-VL-3B)
  pipeline.py                 VictimPipeline protocol: run() + describe_config()
  runner.py                   AttackRunner — fires N trials, records ASR
  results.py                  AttackResult + JSON provenance
  tests/

systems/
  system1_direct/
    README.md                 what it tests, how to run it, results
    victim/                   the vulnerable app + its named defenses
    attacks/                  payloads + success oracles
    experiments/              run scripts (existence N=5, reliability N=50)
    results/                  one JSON per run — the provenance record
  system2_indirect/
    README.md
    victim/                   hiring assistant, retriever, schema defense
    attacks/
    experiments/              attack matrix + run scripts
    results/
    tests/

tools/                        NOT measurement protocol
  smoke_test.py               the 4-bit load gate — run before anything else
  sample_images/              images for the smoke test / manual checks
  victim_chat/                desktop + CLI chat with the victim model
docs/
  claude_code_handoff.md      full build brief and methodology
  harness_core.py             the interface contract the harness implements
Victim Chat.cmd               double-click launcher for the desktop chat
Install Desktop Shortcut.cmd  puts a Victim Chat shortcut on the Desktop
```

Two ideas hold it together:

1. **A victim is a pipeline behind one interface** (`VictimPipeline`). Every
   system implements the same `run()` / `describe_config()` contract, so one
   runner measures them all identically and their ASRs stay comparable.
2. **An attack is a payload *plus* a success oracle** (`Attack`). The oracle
   travels with the attack, because "success" means something different for each
   attack. The runner stays agnostic to attack semantics.

## Running it

Hardware baseline: NVIDIA RTX 3050 6 GB (6 GB VRAM is a hard ceiling), native
Windows. The model must load in 4-bit and never spill into shared GPU memory.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt     # torch + cu124 wheels; install order matters
python tools/smoke_test.py tools/sample_images/test_image.png   # the load gate

# all unit tests (no GPU needed)
python -m unittest discover -s . -t . -p "test_*.py"

# a system's experiments — see each system's README for options
python -m systems.system1_direct.experiments.run_reliability
python -m systems.system2_indirect.experiments.run_reliability --defense schema_extract
```

Results land in that system's own `results/` folder, named
`<attack>_<UTC timestamp>.json`. Each file holds every raw output, the success
count, the full victim config (model revision, quantization, defense, decode
settings) and any attack-specific metrics.

See `docs/claude_code_handoff.md` for the full methodology, the fixed hardware
constraints and the measurement rules (one variable at a time; provenance on
every result).

## Victim model & its license

- **Victim:** `Qwen/Qwen2.5-VL-3B-Instruct`, pinned to a specific commit
  revision, loaded 4-bit NF4 (double quant, fp16 compute) to fit a 6 GB GPU.
- The pinned revision is deliberate: if weights shift mid-series, ASRs stop
  being comparable across systems and the cross-system synthesis collapses.
- **The Qwen2.5-VL model weights are under the Qwen Research License, NOT
  Apache-2.0 and NOT this repo's MIT license.** The MIT license in this
  repository covers *only the code written here*. The model weights are **not**
  redistributed in this repo — you download them from Hugging Face under their
  own terms. Research use is permitted under the Qwen Research License; check it
  before any commercial or redistribution use.

## Scope & ethics

- **Inference-time only.** Models are attacked as fixed victims at runtime.
  Training / data poisoning / fine-tuning are out of scope.
- **Self-owned targets, localhost only.** Every attack targets a locally built
  vulnerable app the researcher owns. No attacks against third-party systems, no
  live callback servers on the public internet.
- **Purpose is defensive.** The point is to measure how VLMs fail under crafted
  multimodal input and how well named defenses hold, so they can be hardened.

## License

Code in this repository: [MIT](LICENSE). Model weights: Qwen Research License
(obtained separately from Hugging Face; not included here).
