# Roadmap: longterm extension strategy

What exists, what is promised, what can be made — and in what order to make it. Design
constraints live in [brief.md](brief.md), per-probe state in `probes/*/spec.md`, per-backend
findings in the backend READMEs. This document adds only ordering and the items those documents
do not yet name.

## Where we are

The instrument is complete and verified on two vendors; the founding question is unanswered.

| Done | Evidence |
|---|---|
| Six probes, all self-verifying (per-launch nonce oracle; a fast wrong answer is a failure) | `probes/*/spec.md`, `runs/` |
| Two backends: CUDA (sm_121a/sm_120a/sm_90) and Vulkan (RADV) | `backends/`, CI build matrix |
| Real-hardware results: RTX 5090 (CUDA, incl. fp4 ≈ 514 TFLOPS) and RX 9070 XT (Vulkan, baseline + load grids) | `runs/amd-baseline-report.md`, README status table |
| Environment layer with no vendor libraries: NVML, amdgpu sysfs, DRM fallback; 1 s sampler; kernel-log watcher with gap accounting | `env/` |
| Harness: `run` / `grid` / `crosscheck` / `report`; self-contained run directories; partial runs kept on interrupt | `run/`, [record-format.md](record-format.md) |
| Record format v1: every stream on one monotonic clock, oracle verdicts are data | [record-format.md](record-format.md) |
| CI: unit tests, CUDA compile matrix, GPU-free phase tests, device suite that skips cleanly | `.github/workflows/ci.yml` |

The corpus holds two (vendor, machine) pairs: 5090/CUDA and 9070 XT/Vulkan. Every "done" row in
the README is still "needs a first run on GB10" — the box the repo was built for. The strategy is
organized around that fact: before GB10 arrives, the repository should be a *corpus* into which
GB10 is a data point; after it arrives, the corpus should make every later machine and driver
update a comparison instead of a first experience.

## What is promised

Committed in this repository, not yet done:

| Item | Promised in |
|---|---|
| Phase 2: `constants` probe — each device fact read from independent sources, agreement required | [crosscheck.md](crosscheck.md) |
| Phase 3: backend-equivalence tolerances; bit-exact mma pinned as a hardware check; row alignment / tag mapping across backends | [crosscheck.md](crosscheck.md) |
| Phase 4: `docs/platforms.md` generated from run directories, one row per (vendor, backend, machine) | [crosscheck.md](crosscheck.md) |
| Grid axis "minutes-long thermal soak" | [brief.md](brief.md), principle 5 |
| Open question: what did the near-crashes look like (hangs, wrong results, sudden slowdowns, killed processes) — the answer decides which probes come first | [brief.md](brief.md) |
| Open question: which other edge devices are in scope next | [brief.md](brief.md) |
| Per-ASIC throttle-word bit decode; `gpu_metrics` family-2 layout; slow-hwmon caching in the sampler | [amd.md](amd.md), Gaps 1–3 |
| fp8 cooperative matrices on Vulkan — driver-exposed, toolchain-unreachable; no number claimed | `backends/vulkan/README.md` |
| NVK on GB10 as an independent open path — "check before relying on it" | [brief.md](brief.md) |
| Crosscheck candidates: file-backed page-cache reads (CUDA-only mechanism); load-condition invariance (delivered load vs requested) | [crosscheck.md](crosscheck.md) |
| The two older Vulkan probes still carry their own `vk_ctx` copies instead of the shared header | `backends/vulkan/README.md` |

## What can be made

New items, each grounded in a fact already in the repository:

1. **Fault and resilience probe family.** The kernel side of the story is already captured
   (xid/reset/oom/hang categories, gap accounting, the "work was discarded" semantics of `reset`).
   The probe side is missing: what happens to in-flight work when a co-tenant dies, OOMs or
   resets, and how long the device is unusable afterwards. Three probes:
   - `oom`: allocation storm with a co-tenant; failure mode and recovery time measured, not
     assumed. The documented GB10 fact — a page cache left full by a large read makes later
     device allocations fail — becomes a probe with an oracle instead of folklore.
   - `recovery`: a co-tenant kill or device reset mid-run. The streams already share one clock,
     so the unusable window is a subtraction from existing data, not a new estimate.
   - `soak` (watchdog): a long run with a heartbeat; a run that stops producing records is
     itself the finding (incomplete manifest plus the kernel events around it).
   This family is what answers brief's open question 1 — and the answers from it reorder phase D.
2. **Unified-memory budget curve.** The memory probe's contention section measures two points
   (GPU alone, GPU + CPU): one shared ≈ 241 GB/s budget on GB10, ratio 1.01 on the 9070 XT
   (discrete, independent resources). Sweeping the CPU read duty from 0 to 100 % turns the two
   points into the curve: where the sum plateaus is the budget, and the curve's shape — knee or
   none — is the integrated-versus-discrete distinction as a measurement, not an adjective.
3. **Energy and thermal derivatives.** `energy_mj` is already sampled on both vendors, and the
   mma spec already ties a latency budget to "the SM clock it ran at". The same discipline for
   power: energy/GB and energy/GFLOP per format, plus DVFS behaviour under the soak condition
   (the clock probe's per-window trace already exists for it).
4. **Corpus tooling.** The report lays N run directories side by side but only renders them;
   the verdict — within tolerance, and *what changed* — is missing. Three commands:
   - `suite`: all probes × backends × conditions on this machine, one corpus, one command.
   - `diff` / `regress`: compare metric distributions between runs with environment context
     (driver, firmware, forced performance level, kernel events) named in the verdict. The
     driver/firmware update is the variable under test; this is the instrument for it.
   - `export`: a portable corpus for publication.
   This is the payoff of record format v1: once runs are comparable data, the "keystone" claim in
   the crosscheck plan stops being an aspiration.
5. **A third vendor (Intel), via the existing Vulkan backend.** The Vulkan code is already
   vendor-neutral; the environment layer already has the DRM sysfs fallback for machines without
   NVML. Proving the suite on a third vendor is what "one probe design" means. Gate: the
   `constants` probe must pass first — device facts must agree across independent sources before
   the numbers are trusted.
6. **Multi-GPU.** The test box already enumerates two AMD cards and both KFD nodes in `env.json`,
   and `crosscheck` can warn when its two backends picked different devices, so per-backend
   device selection exists. Missing: a deliberate two-card run and the P2P path.
7. **fp8 on Vulkan via hand-written SPIR-V.** The driver exposes the configs; the blocker is
   glslang, and the machine-readable path (assembler, ISA docs) is the same one `amd.md` already
   endorses. If the assembler story fails again, the claim boundary stays exactly where the
   Vulkan README puts it.
8. **An inference-like background load.** `gpu50` is a synthetic duty-cycled read; the 5090 runs
   happened "alongside a resident inference engine". An `inference` load condition (a real decode
   loop through the load protocol) is the most ecologically valid stress the grid can deliver.
   It pulls in an external engine, so it is a candidate, not a commitment.
9. **KFD directly, then HIP/ROCr** — only if a question is left open that Vulkan provably cannot
   answer, in that order, per `amd.md`.

## Phases

The only external gate in this plan is GB10 hardware; everything else is code and can start now.

### Phase A — close the promises (no new hardware)

Done 2026-11-06/07, executed per [plan-phase-ab.md](plan-phase-ab.md).

- [x] `constants` probe (crosscheck phase 2) — the closest thing to a pure oracle in the domain;
      gates every later vendor. Verified on the 9070 XT and the Raphael iGPU (all agreement
      checks pass; the iGPU's memory "sources" measure different quantities on UMA — heaps and
      KFD size the host pool, the DRM value is the carve-out — so that check is deliberately
      absent and recorded as a finding in the run). CUDA side ran first on the RTX 5070 Ti
      (sm_120a): API facts agree with `/proc/driver/nvidia` (70 SMs, 2452 MHz,
      15.5 GiB); the host backend (NVML/nvidia-smi/proc/amdgpu/KFD/vulkaninfo) is verified on
      the 9070 XT + iGPU box, after the env/amdgpu API-drift fix (lessons.md 10).
- [x] Equivalence suite, bit-exact mma hardware check and the cross-backend tag mapping
      (crosscheck phase 3): `crosscheck --check` with documented tolerances, `report/tags.py`
      mapping, and the mma `ref_xor` records that make "fp16/bf16/int8 are bit-exact" a
      comparison of data. The first real two-backend verdict still needs a machine with both
      backends — none here. (The work also found the old shared mma operand pattern was
      degenerate — every tile row constant, so the cross-backend checksum would have been
      structurally zero; it is fixed in both backends, see docs/lessons.md.) The split
      cross-machine run (9070 XT vulkan vs 5070 Ti cuda) confirmed the contracts hold across
      architectures: fp16 and int8 dumps bit-exact, all three ref_xor checksums equal, tile
      order identical (docs/crosscheck.md, "Split run").
- [x] `platforms.md` generator (crosscheck phase 4); data fills in as runs happen
      (`docs/platforms.md`, 40 runs and counting: 9070 XT + iGPU vulkan/host, 5070 Ti cuda).
- [x] Record-format lint: `report/records.py` validates every record type, stream rules, and
      every real run directory; CI rejects malformed records.
- [x] Thermal-soak grid condition: `soak`/`soakN` heats every source the machine offers,
      ready only after a minimum soak and a temperature plateau, `load.jsonl` carries the
      achieved duty and the GPU-skip evidence.
- [x] Debt: slow-hwmon discovery caching, `vk_ctx` dedup in the two older Vulkan probes
      (record shape identical to baseline, oracles green), per-ASIC throttle-word decode
      (gfx10.3 and gfx12 tables from the amdgpu driver, `EDC_CPU` decoded on the iGPU),
      `gpu_metrics` family-2 (v2.1) layout with the iGPU now decoding and cross-checking.

Acceptance: `constants` passes on the 9070 XT, the iGPU and the RTX 5070 Ti; CI rejects a
malformed record; the CUDA-vs-Vulkan verdict row lands on the first two-backend machine
(phase C).

### Phase B — corpus tooling

Done 2026-11-06/07 (plan-phase-ab.md), except the cross-generator half of invariance.

- [x] `suite` (probe × backend × condition matrix), `diff`/`regress` (metric verdicts +
      environment delta + kernel/load summaries), `export` (portable corpus archives).
- [~] Load-condition invariance: the per-run intended-vs-achieved comparison ships in
      `diff`/`regress` and the report's load section; the cross-generator comparison (the same
      requested condition through two different generators) awaits a machine that has two
      generators for one axis — this box has exactly one per axis.

Acceptance: a driver or firmware update on the test box is reported as a drift table — metric,
tolerance verdict, environment delta — via `./echolock regress`; verified against real
run-directory pairs.

### Phase C — GB10 (the founding question)

- [ ] Full suite + grids + soak on GB10; GB10 becomes a column in `platforms.md`.
- [ ] GB10 vs 5090: same vendor, same compute-capability family, different form factor — the
      comparison the repo was built for.
- [ ] File-backed page-cache reads crosscheck (the one workload where the two stacks use
      genuinely different mechanisms; on GB10 it is also the unified-memory story).
- [ ] NVK check, per the brief, before any reliance on it.
- [ ] Unified-memory budget curve (item 2).
- [ ] Log every near-crash — hang, wrong result, sudden slowdown, killed process — into brief's
      open question 1; the answers order phase D.

Acceptance: `platforms.md` carries three (vendor, machine) columns, every number
oracle-verified, each run's environment (driver, firmware, clocks, forced levels) recorded.

### Phase D — resilience and the regression cadence

- [ ] `oom`, `recovery`, `soak` probes — in the order phase C's near-crashes dictate.
- [ ] The cadence: `suite` + `regress` after every driver or firmware update, on every machine.

Acceptance: a device reset during a run produces a run directory whose status, kernel events and
recovery time are readable without re-running anything.

### Phase E — breadth

- [ ] Intel (iGPU or Arc) via Vulkan — gated on `constants`.
- [ ] Two-card runs; P2P after.
- [ ] fp8 on Vulkan via hand-written SPIR-V (item 7).
- [ ] KFD, then HIP, only if a question proves to be Vulkan's fault (item 9).
- [ ] The `inference` load condition, if the engine dependency is accepted (item 8).

### Phase F — publication

- [ ] `platforms.md` complete for the corpus; `export` + the published run corpus as an open
      dataset of oracle-verified edge-GPU numbers.
- [ ] README status table tracks platforms (what has been measured where), not just milestones.

## Not doing

- **No model or workload benchmarking.** The repository measures the substrate; a model is a
  different variable with its own noise.
- **No ROCm/HIP by default.** The ordering is Vulkan → KFD → HIP, and each step requires a
  question the previous layer cannot answer.
- **No guessed counter layouts.** The `gpu_metrics` discipline — cross-check against an
  independent source or refuse the decode — applies to every new counter.
- **No driver-bug attribution from one debugging session** (lessons 1 and 3): a second
  independent consumer or a hand-verified instruction path, or the observation stays an
  observation.
- **No forcing the CUDA shape onto Vulkan.** Where semantics differ, the divergence is recorded
  with its evidence.
- **No new Python dependencies.** The stdlib-only environment layer is part of the design.
