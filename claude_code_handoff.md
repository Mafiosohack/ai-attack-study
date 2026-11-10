# Multimodal Prompt Injection Study Series — Build Brief for Claude Code

This is the implementation brief for a security research study series on
multimodal prompt injection. It extends an existing, completed authentication
study series that used the same 4-phase methodology.

**How to read this document.** Sections marked **[IMPLEMENT EXACTLY]** are
decided — build them as written, do not substitute alternatives. Sections
marked **[ASK FIRST]** are undecided — do not guess; surface the question and
wait for a decision. Sections marked **[CHECKPOINT]** are moments where the
human stops to judge the design against real behaviour before continuing.

Two design files accompany this brief: `smoke_test.py` (the load gate) and
`harness_core.py` (the interface contract). Treat `harness_core.py` as the
contract to implement against and actually run — NOT as tested code. It was
written in a GPU-less environment and has never executed.

---

## 0. Hard constraints — [IMPLEMENT EXACTLY], non-negotiable

- **Hardware:** NVIDIA RTX 3050 6GB Laptop GPU, 16GB system RAM, Windows,
  native (NOT inside a VMware guest — GPU passthrough will fail). Treat 6GB
  VRAM as a hard ceiling. Never let the model spill into "shared GPU memory";
  a run that spills is broken, not slow.
- **RAM is the tight resource, not VRAM.** The machine idles near-full.
  Loading pulls weights through system RAM first. Assume the human closes
  other apps before a run; keep the harness's own footprint lean.
- **Scope boundary — inference-time only.** This series attacks models as
  fixed victims at runtime. Training/data poisoning and model fine-tuning are
  OUT of scope entirely — different threat model, separate project. Do not
  add them even if they seem adjacent.
- **Target boundary.** Every attack targets a locally built, self-owned
  vulnerable app. Localhost only. Never wire an attack at any system the
  human does not own. No live callback servers on the public internet.
- **Licensing note.** The target model (Qwen2.5-VL-3B) is under the Qwen
  Research License (NOT Apache 2.0). Fine for research use. Flag it if any
  code path implies redistribution or commercial use.

---

## 1. Environment — [IMPLEMENT EXACTLY]

Native Windows, fresh virtual environment. PyTorch installed BEFORE
bitsandbytes (bnb detects CUDA through torch at import; order matters).

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate bitsandbytes qwen-vl-utils pillow
```

- Verify `torch.cuda.is_available()` returns True and names the 3050. A
  CPU-only torch build is the most common silent failure.
- If the `Qwen2_5_VLForConditionalGeneration` class is missing, transformers
  is too old for Qwen2.5-VL — upgrade it.
- Pin every version in `requirements.txt` once the stack loads cleanly.

---

## 2. The load gate — [IMPLEMENT EXACTLY] then [CHECKPOINT]

Run `smoke_test.py` with any test image. It must pass BEFORE any harness code
is built. Its only job: prove the 3B loads in 4-bit and runs one inference
inside 6GB.

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit NF4, double quant, fp16 compute,
  `device_map="cuda"`.
- Cap image tokens via processor `min_pixels`/`max_pixels`. Uncapped high-res
  images are the likeliest OOM cause on 6GB.
- **Pin the model revision.** Fill the `REVISION` constant with the exact
  commit hash from the HuggingFace repo. This is not optional: if weights
  shift mid-series, ASRs across systems stop being comparable and the
  cross-system synthesis (the point of the series) collapses.

**[CHECKPOINT — human judges before proceeding]:** After the gate passes,
look at the reported peak VRAM and the wall-clock time of one inference. The
question to answer: *is iteration fast enough to fire 50 trials per condition
without it being agony?* If it's painfully slow or near the VRAM ceiling, the
model-size decision gets revisited NOW, not after building three systems on it.

---

## 3. Harness architecture — [IMPLEMENT EXACTLY]

Implement against the interfaces in `harness_core.py`. The two load-bearing
ideas, which must not be refactored apart without a deliberate decision:

1. **A victim is a pipeline behind one interface (`VictimPipeline`).** Each
   system implements the same `run()` / `describe_config()` contract, so a
   single runner measures all three identically and their ASRs stay
   comparable.
2. **An attack is a payload PLUS a success oracle (`Attack`).** The oracle
   travels with the attack, because success means different things for
   different attacks. The runner stays agnostic to attack semantics.

Directory layout:

```
multimodal-injection/
├── harness/
│   ├── pipeline.py     # VictimPipeline protocol
│   ├── runner.py       # AttackRunner — fires N trials, logs ASR
│   ├── model.py        # ONE shared loader, used for inference AND gradients
│   └── results.py      # AttackResult dataclass + JSON provenance logging
├── systems/
│   ├── system1_direct/
│   ├── system2_indirect/
│   └── system3_agentic/
├── attacks/            # payloads + oracles, reused across systems
├── results/            # per-run JSON — the provenance record
└── requirements.txt
```

- **One shared model loader.** The SAME transformers + bitsandbytes stack
  serves both inference attacks and the gradient attack, so model behaviour is
  identical everywhere. Do not introduce Ollama/llama.cpp as a second path —
  they can't do gradients and would split behaviour.
- Follow the existing security-coding conventions: dataclasses (or Pydantic)
  for structured output, `logging` not bare prints in library code, docstrings
  on measurement-relevant functions, comments explain *why* not *what*.

---

## 4. Measurement methodology — [IMPLEMENT EXACTLY]

- **ASR over N trials is the metric.** Not binary pass/fail. This is the
  honest way to report a stochastic target and what makes "attack cost raised,
  not immune" visible in numbers.
- **Two modes:**
  - *Existence* — temp 0 / greedy / batch 1. Near-deterministic on fixed
    hardware. Small N (~5) to confirm the vector works and is stable.
  - *Reliability* — realistic temperature. Large N (default 50). ASR as a
    distribution.
- **One variable at a time.** The victim config is FIXED within a single
  measurement and varied deliberately across conditions. The primary axis is
  defenses OFF vs ON for the same attack. Never randomize victim parameters
  across trials — it makes ASR unattributable.
- **Provenance on every result.** Each AttackResult records model id +
  revision, quantization, decode settings, and which defenses were active.
  Every claim in the eventual report must trace to a file in `results/`.

**[CHECKPOINT — human judges]:** After the first real ASR numbers, check two
things. (a) Do the numbers make sense for the attack? (b) Is existence-mode
actually near-deterministic on this setup, or does temp-0/batch-1 still vary
run to run? If it varies a lot, the core measurement assumption needs
re-examining before more systems are built.

---

## 5. Series structure — [IMPLEMENT EXACTLY at the boundary level]

Three systems, organised by trust boundary (where untrusted content enters).
Execution mechanisms (typographic, steganographic, adversarial perturbation)
are the attack MATRIX inside each system, not separate systems.

- **System 1 — Direct multimodal input.** User uploads a crafted image.
  OCR-pipeline injection is folded in here as the deliberate *text-layer
  control case* — "here's the attack that only needs the text layer, now
  here's the pixel-space attack that survives when the text layer is clean."
- **System 2 — Indirect injection via retrieval.** A RAG/browsing pipeline
  fetches poisoned content the victim never sees.
- **System 3 — Agentic tool-chaining + persistent memory.** Lethal-trifecta
  exfiltration, cross-tool propagation, memory poisoning.

Each system follows the series 4-phase methodology: vulnerable implementation
→ attack scripts → hardened rebuild → professional report. Build System 1
completely before starting System 2.

**Gradient/adversarial attack (System 1 matrix):** craft the perturbation
against the vision encoder IN ISOLATION (~300M params, fits on 6GB with fp16 +
gradient checkpointing), then evaluate the perturbed image against the full
VLM. Do NOT attempt end-to-end backprop through the whole model on 6GB.

---

## 6. Open questions — [ASK FIRST], do not guess

These are genuinely undecided. Surface each when it becomes relevant; wait for
the human's decision.

1. **Success oracle per attack.** What EXACTLY counts as success for each
   attack? A specific leaked canary string? A specific tool action taken? A
   refusal bypassed? This is the single most important undecided thing — a
   wrong oracle makes every ASR meaningless. Never invent oracle logic; get
   the human to define it per attack.
2. **Shared harness vs standalone per system.** The harness is a BET that
   reuse pays off across three systems. It is held loosely on purpose. Build
   System 1 against the skeleton, but do not assume the abstraction is right.
   See the checkpoint in §7.
3. **System 2 internals.** Which retrieval setup, which vector DB, which
   document format for the poison. Not designed yet — do not build ahead.
4. **System 3 internals.** Which agent loop / tool-calling structure, what
   persistent-memory store. Not designed yet — do not build ahead.
5. **Which defenses to implement in each hardened phase.** Candidates:
   spotlighting (transform untrusted data before the model), StruQ-style
   channel separation, a CaMeL-style privileged/quarantined split, capability-
   based access control. Which map to which system, and how fully to implement
   each, is not decided.
6. **Reliability-mode temperature and N.** Default is temp ~0.7 and N=50, but
   the realistic value and trial count trade off against time on this
   hardware. Confirm before committing to long runs.
7. **Canonical taxonomy for the write-up.** The source literature disagrees on
   four points (is adversarial perturbation "injection"? is jailbreak a subset
   of injection? etc.). This is a report decision, not a code decision — it
   does not block building, but flag it when reports start.

---

## 7. The design-vs-reality checkpoints — [CHECKPOINT], the whole point

The human is new to building this class of system and is deliberately learning
by building, then interrogating. These are the moments to stop and judge. Do
NOT smooth past them — surface each explicitly when reached.

- **After the load gate (§2):** is iteration fast enough / within VRAM? If
  not, revisit model size before building anything.
- **After System 1's vulnerable pipeline + first attack runs:** does the
  `VictimPipeline` abstraction feel natural, or is the human fighting it? Is
  the untyped `attack_payload` awkward? Is oracle-beside-attack clunky? THIS
  is where "hold the skeleton loosely" gets tested. If it rubs, refactor the
  harness now — before System 2 locks the interface in.
- **After the first ASR numbers (§4):** do they make sense, and is
  existence-mode actually deterministic here?
- **Before starting System 2:** re-decide shared-harness vs standalone with
  real System 1 code in hand. A leaner path — build each system standalone,
  learn what the abstractions actually need, refactor once — is a legitimate
  alternative that only becomes judgeable after System 1 exists.

---

## 8. Working principle

Claude Code does the mechanical, verifiable work: environment, load-verify
loop, wiring the skeleton, debugging real tracebacks. The human keeps the
decisions (oracles, scope, structure, honest reporting) and reads/runs the
code rather than approving summaries. When something breaks or an interface
rubs, that friction is the signal to interrogate the design — treat it as
information, not an obstacle to smooth over.
