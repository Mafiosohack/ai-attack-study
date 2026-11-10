# Multimodal Prompt Injection Study — System 1

A security-research study series on **multimodal prompt injection** against
vision-language models (VLMs). This repository contains **System 1 (Direct
multimodal input)** of a planned three-system series, built as a reproducible
measurement harness rather than a collection of one-off scripts.

> **Authorized, defensive/educational research.** Every attack here targets a
> locally built, self-owned vulnerable app on `localhost`. The goal is to
> *measure* how readily a fixed VLM victim is steered by crafted image input,
> and how much named defenses actually help — so the failure modes can be
> understood and mitigated. See [Scope & ethics](#scope--ethics).

## What this is

The study treats a VLM as a **fixed victim** and measures **Attack Success Rate
(ASR) over N trials** — not a binary pass/fail — because an honest report of a
stochastic target needs a distribution, and because "defense raised the cost but
didn't make the model immune" only shows up in the numbers.

Two load-bearing ideas structure the code:

1. **A victim is a pipeline behind one interface** (`VictimPipeline`). Every
   system implements the same `run()` / `describe_config()` contract, so a single
   runner measures them all identically and their ASRs stay comparable.
2. **An attack is a payload *plus* a success oracle** (`Attack`). The oracle
   travels with the attack, because "success" means something different for each
   attack. The runner stays agnostic to attack semantics.

## Status

| Piece | State |
|-------|-------|
| Environment + 4-bit load gate | ✅ passed |
| Harness (`model` / `pipeline` / `runner` / `results`) | ✅ built |
| System 1 app #1 — image description victim | ✅ built, with `off` / `naive` / `spotlight` defenses |
| Attack #1 — **typographic** (readable-text *control case*) | ✅ built & measured |
| Success oracle (open question #1) | ✅ decided — fuzzy canary match, unit-tested |
| Pixel-space attack (steganographic / adversarial perturbation) | ⏳ next |
| System 2 (indirect/RAG) · System 3 (agentic) | ⏳ not started |

### Key finding (System 1, typographic attack)

On the 3B victim, the typographic attack reaches **~100% ASR** with defenses
`off` **and** `naive`, and **49/50 (98%)** against a strong `spotlight` prompt —
i.e. prompt-only defenses are essentially useless against a readable-text attack
on this model. Existence mode (greedy) is deterministic, confirming the
measurement assumption. The typographic attack is deliberately the **text-layer
control case**: "here is the attack that only needs the text layer — now here is
the pixel-space attack that survives when the text layer is clean" (the
pixel-space contrast is the next step).

> **Note on `results/*.json`:** the committed result files are **historical,
> strict-oracle** runs from initial bring-up. The fuzzy-oracle findings quoted
> above came from later runs whose JSONs were not persisted. A clean
> fuzzy-oracle re-run to produce the canonical report numbers is a tracked
> to-do; don't mix strict- and fuzzy-oracle counts.

## Layout

```
harness/                # shared measurement core
  model.py              # ONE shared 4-bit loader (inference + future gradients)
  pipeline.py           # VictimPipeline protocol
  runner.py             # AttackRunner — fires N trials, records ASR
  results.py            # AttackResult dataclass + JSON provenance
systems/
  system1_direct/
    image_description.py # victim app + DEFENSES {off, naive, spotlight}
    run_existence.py     # N=5 deterministic confirmation
    run_reliability.py   # N=50 ASR-as-distribution
attacks/
  typographic.py         # payload + fuzzy canary oracle (the control case)
results/                 # per-run JSON provenance records
harness_core.py          # the interface contract (reference, see note below)
smoke_test.py            # the 4-bit load gate
claude_code_handoff.md   # full build brief / methodology
app.py, chat.py          # desktop + CLI chat tools (NOT measurement protocol)
```

`harness_core.py` is the **interface contract** the harness was implemented
against. `app.py` (Tkinter desktop chat) and `chat.py` (CLI REPL) are
convenience tooling that reuse the same victim loader — they produce no ASR and
no result JSON, and are not part of the measurement protocol.

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

## Reproducing

Hardware baseline: NVIDIA RTX 3050 6 GB (6 GB VRAM is a hard ceiling), native
Windows. The model must load in 4-bit and never spill into shared GPU memory.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt   # torch + cu124 wheels; install order matters
python smoke_test.py              # the load gate — must pass before anything else
python -m systems.system1_direct.run_existence     # N=5 deterministic
python -m systems.system1_direct.run_reliability   # N=50 ASR distribution
```

See `claude_code_handoff.md` for the full methodology, the fixed hardware
constraints, and the measurement rules (one variable at a time; provenance on
every result).

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
