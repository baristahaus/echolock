# Cross-platform validation plan

Echolock's core claim — measure what a GPU does, not what it declares — is strongest when the
same stimulus produces numbers from independent stacks. Each stack then validates the others,
and a divergence is a finding instead of a mystery. This plan turns that idea into work.

Three axes of cross-platform variation exist today:

| Axis | Pairs | What it validates |
|---|---|---|
| Backend | CUDA vs Vulkan (same GPU) | two independent implementations of the same probe semantics |
| Vendor | NVIDIA vs AMD | driver/ABI semantics (sync lag, page residency, clock sources) |
| Form factor | GB10 vs discrete | the repo's founding question: does family silicon behave like the spec sheet? |

## Phase 1 — Foundations

*Keystone: everything later is a data change once this exists.*

- `report/markdown.py` renders runs from different backends in one table: column identity
  becomes label + backend when labels collide, so a CUDA run and a Vulkan run of the same probe
  sit side by side under one report.
- `./echolock crosscheck PROBE`: runs the same probe on two backends (`--backend-a cuda
  --backend-b vulkan`), collects both run directories, and renders one combined
  `crosscheck-report.md`. Each run keeps its own oracle verdict; the crosscheck's job is to put
  the numbers next to each other where a human or a later phase can compare them.

## Phase 2 — Device-constant validation

New probe, `constants`: read each device fact from independent sources and require agreement.

- L2 bytes: CUDA properties vs Vulkan limits vs the pointer_chase latency-map inflection.
- SM count: CUDA properties vs the clock probe's per-SM stamp coverage.
- Device name / memory size: NVML vs CUDA `totalGlobalMem` vs Vulkan memory heaps.

This is the closest thing to a pure oracle the domain has: the sources share no code path, so
disagreement means someone is misreporting. Runs on any backend, both vendors. Skipped entries
are recorded, never guessed.

## Phase 3 — Backend-equivalence suites

For probes where both backends express the same semantics, assert invariants rather than
equality, with documented tolerances:

- memory: bandwidth and copy throughput within a tolerance (both stacks should find the same
  DRAM ceiling on the same hardware).
- pointer_chase: latency-vs-footprint curves with the same cache-level inflection points.
- mma: fp16/bf16/int8 tensor ceilings within tolerance (accumulator semantics differ, so
  bit-exactness does not hold across backends; the ceiling must).
- mma bit-exact: int8 and fp16 ARE bit-exact across the backends per the Vulkan README — pin
  that as a hardware check, not prose.

Each check lands as `crosscheck` records (`agrees` / `diverges` / `within-tolerance`) in the run
directories, so the verdict is data the report can render.

## Phase 4 — Publication layer

- `docs/platforms.md`: generated from run directories, one table per probe, one row per
  (vendor, backend, machine). The AMD box, the RTX 5090 and GB10 fill it as runs happen.
- When GB10 arrives it becomes a third column automatically; GB10-vs-5090 (same vendor, same
  compute-capability family, different form factor) is the comparison the repo was built for.

## Later candidates (not scheduled)

- File-backed GPU reads of page-cache-resident files: NVIDIA CUDA can import file-backed host
  memory; AMD Vulkan cannot (documented in backends/vulkan/README.md). The one workload where
  the two stacks use genuinely different mechanisms for the same job.
- Load-condition invariance: the grid's `cpu50`/`mem80`/`gpu50` conditions validated by running
  the same requested condition through different generators (Python memfill vs CUDA gpuload) and
  asserting the delivered load matches the request.

## Status

- [x] Phase 1: cross-backend report columns; `crosscheck` mode (per-backend args via
  `--a-args`/`--b-args`; warns when the two runs declared different devices).
- [x] Phase 2: constants probe (host backend; NVML / nvidia-smi / proc-information / amdgpu
  sysfs / KFD / vulkaninfo sources compared per device fact).
- [x] Phase 3: cross-stack alignment (tag + metric-name alias map, canonical sync form) with
  tolerance verdicts; bit-exact mma: fp16 is BIT-EXACT across CUDA WMMA and Vulkan
  cooperative-matrix on the RTX 5090 (`crosscheck --bitexact`). int8 bit-exact is impossible on
  this GPU: NVIDIA's cooperative matrix lists no M16N16K16 int8 config (AMD's does) — itself a
  platform finding. The bit-exact run also exposed and fixed an oracle limit: past ~262k
  iterations want*iters exceeds the fp32 exact-integer range, so the exact throughput check
  would fail on rounding; iteration counts are now capped in both backends. (Earlier note: the
  "Vulkan mma/sync fail on NVIDIA" finding was device selection — the Vulkan probes defaulted
  to the RADV iGPU on this hybrid machine; they now take --device and pass on the 5090.)
- [x] Phase 4: `./echolock platforms` -> docs/platforms.md, regenerated from run directories.
- [x] Load-condition invariance: every grid condition run records a `load_check` in the manifest
  (requested vs the generator's self-reported achieved, plus a sampler view that must show at
  least the target); the report flags a run whose load was not what its label says. The check
  caught two real issues on its first version: loadavg cannot validate a 10 s run (the sampler
  now derives a per-sample cpu_busy_frac from /proc/stat deltas), and the sampler's view
  includes the probe's own work, so it bounds the ambient load from below only.

## Next (the one candidate left) -- blocked on ONE driver, works on GB10

- File-backed GPU reads: NVIDIA CUDA can import file-backed host memory; AMD Vulkan cannot
  (backends/vulkan/README.md). One workload, two genuinely different mechanisms; the residency
  model is file_page's.
- Feasibility tested on this machine (RTX 5090, open kernel module 615.71.09): the GPU can read
  registered host memory and the XOR oracle works, but `cudaHostRegister` REJECTS file-backed
  pages -- MAP_SHARED and MAP_PRIVATE mappings of a real file both fail with
  cudaErrorInvalidArgument, while anonymous malloc and anonymous mmap register fine (control
  tested). The probe is blocked on the driver accepting file-backed registration.
- **Re-tested on GB10 as requested (Asus GX10, sm_121a, open kernel module 580.173.02, ext4 on
  NVMe): it is not blocked here.** `cudaHostRegister` accepted file-backed pages in every variant
  tried -- MAP_SHARED and MAP_PRIVATE, `PROT_READ|PROT_WRITE` and `PROT_READ` only, flags 0 and
  `cudaHostRegisterMapped` -- and the anonymous control also passed, so nothing about the harness
  is doing the work. Register success alone is not the evidence; the device then read 8 MiB through
  the registered file-backed pointer and a checksum over it matched the host-written pattern
  exactly (1,069,547,520 = 1,069,547,520), for both flag settings. Unregistered file-backed reads
  also work through the second-level page tables (`cudaMemcpy` from a `MAP_SHARED` file mapping
  succeeds), and `file_page --phase res-seq` on this box measures that path at 163.8 GB/s with
  `page_cache_resident` 1.0 on a 2 GiB file.
- So the block is **specific to the 5090 / 615.71.09 pair**, not a CUDA-wide rule, and a CUDA
  `gpu_read` phase is viable on this platform rather than needing a staging copy. Two caveats
  before designing on it. (1) Fault state is still untested: `mincore` reports *page-cache*
  residency, not the mapping's fault state, so my seven variants all measured 16384/16384 resident
  and none of them was genuinely unfaulted -- settling it means evicting between map and register,
  which on a node with a resident engine means doing it somewhere else. (2) I did not reproduce
  their exact conditions (5090 geometry, their file's filesystem, hugepages, `O_DIRECT`), so this
  is "works on GB10/580", not "the 5090 report was wrong". The check is a standalone ~60-line
  binary, not a repo artifact: map a file, write a pattern through the mapping, register, get the
  device pointer, checksum from a kernel, compare -- and print unbuffered, because a probe that
  segfaults with buffered stdout takes its own diagnosis with it (docs/lessons.md item 8). The
  three variants used here, with their output, are kept outside the repo at
  `192.168.0.25:~/echolock-runs-gb10-20261010/repros/` alongside the run records.
