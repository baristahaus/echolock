# echolock: design brief

Carried over from the planning discussion that preceded this repository. It records what was
settled; the README tracks what exists.

## Purpose

Measure what an edge GPU actually does rather than what it declares. Every probe sends a known
stimulus, checks the result against an independent oracle, and times the response. It repeats
this under controlled background load, memory pressure and thermal soak, and records time series,
not single numbers. GB10 (DGX Spark class, `sm_121a`) comes first, but the design must not assume
NVIDIA.

## Principles

1. **Declared vs measured.** Reported attributes only decide which probes to run. The report
   always shows declared, measured idle and measured under load side by side.
2. **Every probe verifies its own result.** An exact oracle for exact results, a tolerance for
   floating point. A fast wrong answer counts as a failure, not a result.
3. **Derive, don't ask.**
   - SM clock from `clock64` against `%globaltimer`, not from nvidia-smi.
   - Cache and TLB sizes from pointer-chase latency steps.
   - Host↔GPU timing from a polled `%globaltimer` calibration that never calls a synchronize.
4. **Detect silent fallbacks.** For every requested feature, check that its effect happened:
   compression → read-speed difference; sync schedule → wake-up latency; residency → still
   resident under pressure.
5. **The load grid is a first-class axis.** The grid is:
   - background SM or bandwidth share;
   - CPU memory traffic;
   - unified-memory fill at 50/80/95%;
   - a minutes-long thermal soak.

   Output is time series, so step changes show.
6. **Record the environment alongside every result:** driver, CUDA, kernel, firmware, clocks,
   temperature and power at 1 s, plus `dmesg` Xid lines captured during each run.
7. **Know the floor.** The closed GSP firmware and hardware power management sit below every API,
   CUDA, NVIDIA's Vulkan driver and open stacks alike. Below that, timing is the only instrument.

## Backends

- **CUDA/PTX first.** It's the precise instrument on NVIDIA and the only way to reach the NVFP4
  tensor cores and unified-memory fault behaviour. It uses the runtime and the driver API through
  entry points, so no `-lcuda`, and no vendor libraries.
- **Vulkan compute second.** It gives one probe design across NVIDIA, AMD, Intel, Mali and
  Adreno. On NVIDIA, the proprietary Vulkan driver is a second closed view of the hardware, not an
  unwrapped one; where it disagrees with CUDA, that points at the driver. NVK (Mesa) would be an
  independent open path if it supports GB10; check before relying on it.
- **AMD is open all the way down, which changes what can be derived there.** The environment layer
  already reads an AMD GPU from the `amdgpu` driver's sysfs files with no library at all, so
  nothing in the observation path is vendor code. RADV, the Vulkan driver for AMD, is open source
  (unlike NVIDIA's), so a disagreement between CUDA and Vulkan on NVIDIA points at the driver
  while agreement on AMD is evidence, not coincidence. AMD also publishes machine-readable ISA
  and per-IP performance-counter documentation, which is enough to *check* an instruction-level
  claim instead of repeating one. Details and citations: [amd.md](amd.md).
- Probe *definitions* are backend-neutral: stimulus, oracle, parameters, metrics. Each backend
  implements them.

## Known on GB10 before this repository

From ninfer-gb10 `tools/gb10/` (the three probes ported here):

| Probe | GB10 results |
|---|---|
| memory | 246 GB/s (spec 273). Pinned host memory 174–216 GB/s, pageable 161 GB/s, host-to-device copy 59 GB/s. CPU+GPU share one ~241 GB/s budget. Compression 8.4× on zero-filled data, 1.00–1.01× on real weights |
| file_page | Resident 162–169 GB/s. Cold: 96 µs for one fault, ~1 µs per row when faults run in parallel. No hangs |
| sync | Blocking sync returns 0.4–1.2 ms late (worse after longer waits); yield/spin return in ~3 µs. Graph launch doesn't scale with node count (4,000 nodes in 14 µs) |

Also known:

- The GPU clock held at 2,405–2,489 MHz under a busy host core; 41 W mean and 56 W peak during
  decode.
- A page cache left full by a large read makes later device allocations fail. Drop caches between
  probes.
- GPU/driver: device 0x2E12, driver 580.173.02, CUDA 13.0.88, 20 Arm cores (A725 at 2.8 GHz +
  X925 at 3.9 GHz).

## Milestones

1. The skeleton and the environment capture, including the Xid watcher and the 1 s
   clock/power/temp sampler.
2. Port the three probes as-is, with results written as JSONL.
3. New probes:
   - derived SM clock;
   - pointer-chase latency map (L1/L2/TLB);
   - per-format compute throughput with correctness checks (FP32, BF16, FP8, NVFP4 MMA).
4. Load generators and the grid runner; first map of the GB10 idle vs 50% load vs 95% memory
   fill.
5. Vulkan backend for the memory and latency probes; compare against CUDA on GB10.

## Open questions

- What did the near-crashes look like (hangs, wrong results, sudden slowdowns, killed processes)?
  That decides which probes come first.
- Which other edge devices are in scope next?
- Public or private repo, and which license?
