# Plan: executing roadmap phases A and B

The [roadmap](roadmap.md) ordered the work; this document is the execution plan for everything
in it that needs no new hardware — phase A (close the promises) and phase B (corpus tooling) —
and records the process as it runs. GB10 (phase C) is the only external gate and is not part of
this execution.

## Scope

| Slice | Roadmap items | Files owned |
|---|---|---|
| S1 `constants` probe | A: constants (crosscheck ph. 2); mma cross-backend bit-exactness as data (ph. 3 prerequisite) | `probes/constants/spec.md`, `backends/cuda/constants.cu`, `backends/cuda/Makefile`, `backends/vulkan/constants.cpp`, `backends/vulkan/Makefile`, `run/plans.py`, `backends/cuda/mma.cu`, `backends/vulkan/mma.cpp` (checksum records only), `tests/test_constants*.py` |
| S2 Python tooling | A: record-format lint, platforms.md generator; B: suite / diff / regress / export, tag mapping, crosscheck `--check` (ph. 3) | `echolock`, `report/records.py`, `report/tags.py`, `report/platforms.py`, `report/diff.py`, `run/suite.py`, `tests/test_{records,tags,platforms,diff,suite}.py`, `docs/crosscheck.md` (status) |
| S3 env + load | A: thermal-soak grid condition, slow-hwmon caching, per-ASIC throttle-word decode, `gpu_metrics` family-2 layout | `load/soak.py`, `load/README.md`, `run/grid.py`, `env/sampler.py`, `env/amdgpu.py`, `tests/test_soak.py`, `tests/test_amdgpu.py` (extend), `docs/amd.md` (gaps) |
| S4 Vulkan dedup | A: `vk_ctx` debt | `backends/vulkan/memory.cpp`, `backends/vulkan/pointer_chase.cpp`, `backends/vulkan/common/vk_ctx.h`, `backends/vulkan/README.md` |
| Integration | all of the above, verified | `docs/roadmap.md` (checklist), `README.md`, this file's process log, beads |

File ownership is exclusive per slice; the integration step owns the shared docs. Nothing a
slice owns overlaps another slice's list (the one deliberate exception: `run/plans.py` is S1's
alone, `run/grid.py` is S3's alone).

## Contracts shared by all slices

- **Record format v1** ([record-format.md](record-format.md)) is unchanged. New records use
  existing types (`declared`, `measure`, `check`, `feature`, `note`). No new fields on existing
  records; new record `type`s require a `record-format.md` row and a lint rule (S2's validator
  must accept them).
- **Oracles stand.** A probe whose numbers cannot be verified is not merged; a fast wrong answer
  is a failure, never a result. New probes emit per-launch nonces where a launch could no-op.
- **No new dependencies.** Python stays stdlib-only; C++ stays header-vendored Vulkan and
  entry-point CUDA. A counter or bit whose meaning cannot be cited from a primary source is
  stored raw and the gap documented (the `gpu_metrics` discipline in [amd.md](amd.md)).
- **This machine has no NVIDIA GPU.** CUDA work is compile-checked (sm_121a/sm_120a/sm_90, as CI
  does) plus GPU-free phases; Vulkan work runs for real on the AMD box (device 0 = RX 9070 XT,
  device 1 = iGPU; `vulkan` tests already exercise device 0 this way).
- **Evidence over claim.** Every slice reports what it actually ran and saw; compilation is not
  verification.

## Per-slice acceptance

### S1 — `constants` probe

The closest thing to a pure oracle in the domain: each device fact read from independent sources,
agreement required, disagreement is a finding.

- `probes/constants/spec.md` in the house format (stimulus, oracle, parameters, metrics, purpose),
  listing per key: the sources, and the agreement rule (exact for integers and names; a
  documented tolerance or band for values derived from a run, since a latency-map inflection is
  coarse by construction).
- Sources without new libraries:
  - CUDA: `cudaGetDeviceProperties` (name, total memory, SM count, L2, clocks), NVIDIA sysfs
    (`/proc/driver/nvidia/gpus/*/info`, `/sys/class/drm/card*`).
  - Vulkan: `vkGetPhysicalDeviceProperties` + limits + memory heaps, KFD sysfs
    (`/sys/class/kfd/kfd/topology/nodes/*`), amdgpu sysfs (DPM sclk table, VRAM).
  - Derived, via `--against RUN_DIR` (optional): pointer_chase's L2 inflection (the declared L2
    must sit between the last small-hop and first large-hop footprint), clock probe's per-SM
    stamp coverage vs SM count. A missing/wrong run dir is a `note`, never a guess.
- Records: one `declared` per (key, source); one `check` per key with ≥2 present sources
  (`pass` = agreement per the spec's rule, `detail` = the values); a `feature`
  (`requested: 2 sources, granted: N, effect_observed: null, evidence: why fewer`) when a key
  has fewer sources than the spec lists.
- Registers in `run/plans.py` for both backends, so `run`, `grid` and `crosscheck` pick it up.
- **Mma checksum (phase-3 prerequisite):** both mma backends emit one additional verified record
  per format — the XOR of every D-fragment word the oracle already checked (stable across runs,
  timing-free) — so "fp16/bf16/int8 are bit-exact across backends" becomes a comparison of data.
- Acceptance: compiles for sm_121a, sm_120a, sm_90 and the Vulkan target; runs for real on the
  9070 XT and the iGPU with all `check`s passing (or failing with a finding recorded — a real
  disagreement is a valid result, a crash is not); host-phase tests follow the `test_vulkan.py`
  / `test_cuda_host_phases.py` conventions (skip cleanly without hardware).

### S2 — Python tooling

- **Lint:** `report/records.py` validates every record type in `record-format.md` (enums,
  required fields, `v`, `src`, `t_ns`). CI test: synthetic valid corpus passes, corrupted
  records fail, and every real run directory in `runs/` that exists at test time validates.
- **Tag mapping:** `report/tags.py` maps (probe, metric, tags) from each backend to a canonical
  key — built by reading the real records in `runs/` and the probe sources, not from memory.
  This is the "row alignment" crosscheck phase 1 left open.
- **`./echolock platforms`** → `docs/platforms.md`: one table per probe, one row per
  (vendor, backend, machine), device name/driver from `declared`, headline metric per probe
  (a small per-probe mapping in code), oracle verdict from the manifest. Deterministic order.
- **`./echolock export RUNS... --out FILE`** → a self-contained archive of run directories with
  a content manifest.
- **`./echolock diff A B [--tol]`** → per canonical metric: value A, value B, relative delta,
  verdict; an environment-delta section (driver, firmware, toolchain.build, forced perf level,
  declared keys); kernel-event summary. Exit 0 within tolerance, 1 beyond. `regress RUN` =
  diff against an auto-selected baseline (latest same probe+backend+label+device) or `--against`.
- **`./echolock suite [--backends] [--conditions]`** → every probe × backend with a built
  binary under each condition (default `idle`), one combined report with a
  probe×backend status matrix; missing binary = "not built", not a failure.
- **`crosscheck --check`** (phase 3): after both runs, apply the documented invariants via the
  tag mapping — memory bandwidth/copy within tolerance, pointer_chase DRAM-hop and
  smallest-footprint hop within tolerance, mma fp16/bf16/int8 ceilings within tolerance, and the
  mma checksums equal exactly (bit-exact). Verdicts land as `crosscheck` records in both run
  directories and a section in the combined report. Tolerances are named in code and in
  `docs/crosscheck.md`; the status section there is updated.
- Acceptance: every command unit-tested (fake probes + real `runs/` directories where the test
  environment has them); `platforms` run for real over the existing `runs/` corpus and the
  generated file committed; lint green in CI shape (`python3 -m unittest discover`).

### S3 — env + load

- **Soak:** `load/soak.py` heats every source the machine offers — CPU threads at 100 % duty
  (reusing `cpuload.py`'s machinery), `gpuload --duty 100` when the binary and an NVIDIA GPU are
  both present, skipped with evidence otherwise — and reports `load_ready` only after the longer
  of a minimum soak (default 5 min, `--soak-min`) and a temperature plateau (rise < 1 °C over
  60 s). Grid condition `soak[MIN]` (default 5) wired into `run/grid.py`; `load.jsonl` carries
  intended, achieved duty, the skip evidence, and heartbeats with temperature.
- **Sampler cost:** cache the slow-hwmon *discovery* (device list, sensor names, slow-set) for
  the sampler's lifetime; values keep their current freshness. Report `sample_cost_us`
  before/after from a real sample run.
- **Throttle-word decode:** find the per-ASIC bit table in the AMD SMI source (or the amdgpu
  driver), port only the asics this box has (GFX1201, and the iGPU's asic), with the citation in
  a comment. If no citable table exists for an asic, that asic stays raw — documenting the
  search is the honest result. `clock_event_names` fills in from the table when present.
- **`gpu_metrics` family 2:** add the v2.x layout from the kernel's `kgd_pp_interface.h` for the
  version/revision the iGPU reports, keeping the existing cross-check (sclk level ∈ DPM table,
  edge temp ≈ hwmon). The iGPU that is currently refused should now decode — or refuse again,
  with the reason, if the cross-check says so.
- Acceptance: a real `grid` run under `soak` on this box with a sane `load.jsonl`; sampler
  before/after numbers; `./echolock env` shows the iGPU `gpu_metrics` decoded-and-cross-checked
  (or refused with reason); `docs/amd.md` gaps section updated with what is now closed and what
  remains.

### S4 — Vulkan dedup

`memory.cpp` and `pointer_chase.cpp` adopt `common/vk_ctx.h` (the way sync/clock/file_page/mma
already do), extending the header if it lacks something they need. No `Makefile` changes
(expected: include-only); if one becomes unavoidable, report it instead of making it.
Acceptance: rebuild clean; memory and pointer_chase re-run on the 9070 XT with every oracle
passing and headline numbers within run-to-run noise of
`runs/20261010T180655Z-memory-vulkan-amd-baseline` /
`runs/20261010T180711Z-pointer_chase-vulkan-amd-baseline` (side by side via `./echolock report`);
the Vulkan README line about the two older probes' own copies is corrected.

## Integration (after all slices land)

1. Full builds: CUDA sm_121a/sm_120a/sm_90, Vulkan. Full test suite.
2. Real runs on the AMD box: `constants` (both devices), `memory`, `pointer_chase`, `suite
   --backends vulkan`, `grid` under `soak`, a `diff` of the S4 before/after pair,
   `platforms` over the whole corpus.
3. `crosscheck --check` is implemented and unit-tested, but its real verdict needs two backends
   on one GPU — impossible on this box (no NVIDIA). Recorded as such; first real run happens on
   the 5090 or the GB10.
4. Docs: roadmap checklist updated, README (new commands, status), beads closed, process log
   below completed.

## Process log

- 2026-11-06 — Plan written; four slices dispatched in parallel with exclusive file ownership.
  Baseline: 39 tests green, 2 clean skips; toolchains present (nvcc 13.4, glslang); no NVIDIA
  GPU on this box. Decisions: mma checksum record folded into S1 (it owns the mma sources);
  tag mapping owned by S2 since it feeds diff/platforms/crosscheck alike; S4 barred from the
  Vulkan Makefile to avoid a parallel-edit race with S1's new target.
- 2026-11-06/07 — S1 (constants) stalled: 33 minutes, zero files written, one tool call open for
  9+ minutes with no build process. Killed, respawned (S1b) with the design fully preset in the
  task (per-key source table, inflection rule, ref_xor contract) and a timeboxed read list.
  S1b completed in 50 min. Its findings:
  - **The old shared mma operand pattern was degenerate.** `b = (k*11+n*5)%5 − 2` has
    `n*5 ≡ 0 (mod 5)`, so every tile row was constant and the 16×8 XOR was *structurally zero* —
    the cross-backend "bit-exact" checksum would have compared 0 to 0 for any correct (or
    identically wrong) hardware. Fixed to `n*3` in both backends (host-side fills and reference
    only; kernels untouched, oracles still bit-exact). New ref_xor = `0x0000000000000002`,
    identical in both backends by construction and matching an independent Python computation.
  - **iGPU memory "sources" measure different quantities on UMA**: Vulkan heaps and the KFD bank
    size the shared host pool (23/32 GiB), the DRM value is the 2 GiB carve-out. The memory
    agreement check is deliberately absent there and recorded in the run as a finding — a check
    would have been a false alarm.
  - Kernel 7.2.9's KFD sysfs has no `cu_count` or `memory/information` files; the probe uses
    `mem_banks/*/properties` and `array_count × cu_per_simd_array` instead (cited in spec).
  - ref_xor record shape adapted to S2's already-merged string-detail parser
    (`detail: "ref_xor=<hex>"` in a `<fmt>.ref_xor` check record); the cross-backend contract
    (equal strings ⇔ equal references) is unchanged.
- 2026-11-06/07 — S4 (vk_ctx dedup) done clean: 650→377 / 466→251 lines, record shape
  byte-identical to baseline (strict per-record diff), no Makefile changes. Two corpus notes:
  the `20261010T180711Z-pointer_chase` baseline is a dead usage-error artifact of an old plan
  revision (passed `--file`, which the Vulkan probe never accepted) — the effective baseline is
  `…180852Z`; and clock-sensitive series in the re-run ran 1.5–2× faster because the Oct 10
  baseline caught a down-clocked GPU (141 MHz, 7% util) while the re-run was warm (1800–2210
  MHz) — DVFS state, documented in the run's samples, not a probe change.
- 2026-11-06/07 — S3 (env+load) done: soak grid condition (real `grid … --conditions soak1`
  acceptance: ready at +62 s, cpu duty 0.93–0.99, GPU skipped with evidence), sampler hwmon
  discovery caching (steady hwmon ~110 µs; the 60 s tick dropped from ~19 ms to ~10 ms),
  `gpu_metrics` v2.1 layout with the iGPU (gfx10.3.6) now decoding and cross-checking, per-ASIC
  throttle tables ported from the **amdgpu driver** (the authoritative definition AMD SMI
  transcribes) with `EDC_CPU` decoded on the iGPU. One transient failure: the first real soak
  run stalled silent (process alive, zero records) under concurrent slice load; the re-run on a
  quiet box passed first try; treated as a shared-box condition, not a reproducible defect.
- 2026-11-06/07 — S2 (tooling) done: 39 new tests, all five commands verified against the real
  corpus; `crosscheck --check` unit-tested with synthetic two-backend pairs (a real verdict needs
  a two-backend machine — none here). Its "missing driver version" gap turned out to be honest
  history: pre-vk_ctx pointer_chase binaries genuinely lack the declared key; post-migration
  runs carry it.
- 2026-11-07 — Integration:
  - Fixed the `report/markdown.py` declared-column collision S4 flagged (two runs sharing
    label + backend overwrote each other in the values dict): columns now carry an ordinal when
    they collide; verified on the S4 baseline/re-run pair.
  - **Found and fixed the CUDA Makefile ARCH trap** (lesson 7): ARCH was not a prerequisite, so
    `make ARCH=sm_120a` silently kept sm_121a binaries while the PHONY stamp claimed sm_120a —
    `env.json`'s `toolchain.build` would have recorded the wrong architecture. First fix attempt
    was itself wrong (stamp file depended on a rule targeting the literal name `stamp`, so make
    never re-ran the check). Final form: the stamp file is the target, `FORCE`-re-evaluated,
    rewritten only on content change, deleting stale binaries on change. Verified with the
    120a→120a→90→90→121a sequence: 8 compiles, 0, 8, 0, 8.
  - Final state: 97 tests OK (3 clean skips), all three CUDA archs compile from source, Vulkan
    clean; corpus gains the constants runs (both AMD GPUs), the first mma run carrying ref_xor
    records, and the soak grid run; `docs/platforms.md` regenerated at 31 runs.
  - Remaining before phase C closes: a real two-backend `crosscheck --check` verdict and the
    first CUDA-side `constants` run — both need a 5090 or the GB10.
- 2026-11-07 — Divergence found at push time, merged; first CUDA-side runs landed on the 5070 Ti box.
  The push was rejected: the remote branch carried 9 commits from the 5090 and GB10 machines
  implementing much of the same roadmap (bit-exact mma via `--dump-matrix`, a platforms
  generator, GB10 first-run + independent-review docs, `load_check` in grid manifests, a host
  backend for the constants probe, CI removed). Merged, not rebased — those commits are the
  user's own work. Ten conflicts, resolved as:
  - mma: both cross-backend mechanisms kept — in-run `ref_xor` records (fp16/bf16/int8, the
    `n*3` pattern) *and* `--dump-matrix` JSON dumps for external file comparison; the
    fp32-exactness iteration cap (past ~262k iters want*iters overflows the fp32 exact-integer
    range) kept in both backends.
  - platforms: the per-run generator (canonical tag mapping) kept. The other side's generator
    was written to the 5090-era record names (`mma_throughput`, hostname as platform) and could
    not reproduce the current corpus; its committed 5090-row platforms.md stays in git history.
  - CLI: `crosscheck` keeps both `--check` (tolerances + ref_xor) and `--bitexact` (dumps);
    the duplicated `platforms` subparser collapsed to one.
  - `test_constants.py` split by family (GPU vs host) — colliding helper class names would
    have silently dropped one side's tests.
  - **Post-merge defect found by running, not reviewing:** the host constants probe (October
    code) was written against the pre-S3 `env/amdgpu` API — it crashed on AMD and reported
    "no GPU sources." Fixed: `dpm_table` now returns `{"states": {level: MHz}, ...}`; KFD nodes
    join their card's bucket via the topology `device_id` (the two kernel views of one GPU now
    actually compare); a >4x memory spread is a UMA carve-out (different quantities, a note,
    not a failure); vulkaninfo is matched on `deviceID` (a PCI ID is not a name — the old code
    compared `0x7550` against "AMD Radeon RX 9070 XT"); `probe_end` now carries `failed_checks`
    per the record-format contract. Eight regression tests.
  - The 5070 Ti box (sm_120a, CUDA 13.3): `constants` ok — API facts agree with
    `/proc/driver/nvidia` (70 SMs, 2452 MHz, 15.5 GiB); `mma` fp16 100.3 / bf16 25.1 /
    int8 200.7 TFLOPS. The fp16/bf16/int8 `ref_xor` checksums are **all 0x2 — exactly the AMD
    backend's values**: CUDA WMMA and Vulkan cooperative-matrix, same `n*3` pattern, first
    cross-architecture agreement of the checksum records. The remaining phase-C item is now
    only the two-backend `--check` verdict (one machine, two drivers).
- 2026-11-07 — Split crosscheck (user: both machines granted, "do split testing if you can").
  memory/pointer_chase/mma run in parallel — Vulkan on the 9070 XT, CUDA on the 5070 Ti
  (around an idle co-resident vLLM, 0.5 GiB ceiling instead of killing it) — then compared
  with the same `crosscheck_verdicts` + `compare_dumps` functions the CLI uses. Findings:
  - **fp16 and int8 matrix dumps are bit-exact across architectures** — 0/256 result words
    differ, operands identical, tile storage order identical. The m16n16k16 accumulation is
    order-identical in practice on RDNA4 vs GB202. bf16 has no comparable dump (CUDA
    m16n8k16 vs RADV m16n16k16 tiles — filed echolock-0mq); its evidence is the equal
    ref_xor.
  - The tolerance checks read as the platform gap, not driver defects: dependent-load hop
    latency 67.1 vs 232.2 ns (64 KiB random), gpu_read 846 vs 634 GB/s; mma ceilings within
    14 % across formats. Documented in crosscheck.md "Split run".
  - Two process lessons recorded (lessons.md 10-11): the merge's green suite did not cover
    the host probe's stale env/amdgpu expectations (a consumer re-run on hardware is the
    check); and the probe CLI is per-backend (CUDA `--gib` vs Vulkan `--mib`) — the split
    run's first Vulkan memory run died on an unknown argument, diagnosed from stderr.log,
    with the broken run dir failing the record lint until removed.
  - Corpus: 6 new runs (40 in platforms.md); beads 3bb closed, 74k updated, 0mq filed.
