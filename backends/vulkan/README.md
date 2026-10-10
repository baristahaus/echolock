# Vulkan compute backend (milestone 5)

One probe design across NVIDIA, AMD, Intel, Mali and Adreno. The first probe is implemented:
**memory** — GPU reads of each kind of Vulkan memory, copies, and CPU/GPU contention, timed
with GPU timestamp queries and verified with the XOR oracle.

On this machine it runs on AMD through RADV (Mesa) — an open-source driver, so the whole path
from the probe to the hardware is open, unlike the NVIDIA proprietary Vulkan driver. First
verified results on a Radeon RX 9070 XT (RADV GFX1201, Mesa 26.2.4):

| Measurement | Result |
|---|---|
| GPU reads of device-local memory (VRAM) | 626 GB/s |
| GPU reads of host-visible memory (system RAM over PCIe) | 53.6 GB/s |
| Host-to-device copies (verified by a read of the destination) | 44.8 GB/s |
| CPU reads of the same host buffer | 159.5 GB/s |
| GPU under CPU contention | ratio 1.01 (independent resources on a dGPU) |
| Submit→fence-return lag (sync probe, 50 µs…60 ms kernels) | p50 ≈ 100–820 µs; first submit after idle ≈ 30–900 ms (GPU wake) |
| GLSL shader-clock rate (clock probe) | ≈ 800 MHz idle, ≈ 1517 MHz loaded |
| GPU reads of imported host memory (file_page) | ≈ 25 GB/s resident, recovery verified |
| Tensor throughput, verified vs exact reference (mma) | fp16 ≈ 97, bf16 ≈ 99, int8 ≈ 188 TFLOPS |

The load grid is wired for this backend too (`./echolock grid <probe> --backend vulkan`);
the first baseline (2026-10-10, single runs plus idle/cpu50/mem80/io500 grids) is in
`runs/amd-baseline-report.md`, `runs/grid-memory.md`, `runs/grid-pointer_chase.md` and
`runs/grid-mma.md`. Grid highlights: VRAM bandwidth is insensitive to CPU/host-IO load
(632 GB/s under every condition), host-RAM hops stretch under cpu50 and io500 (host-visible
64 MiB random: 831 → 961 ns/hop), and tensor throughput holds across host-load conditions
with per-rep DVFS spreads visible in the series.

## What is implemented

- `backends/vulkan/memory.cpp`: the memory probe. Device-local and host-visible buffers, read
  dispatches timed by `VK_QUERY_TYPE_TIMESTAMP`, every dispatch's XOR checked against the host's
  expectation with a per-dispatch nonce folded into the result slot (an unwritten slot cannot
  pass), copies verified by a read of the destination, and a CPU/GPU contention section where
  every CPU pass is checked.
- `backends/vulkan/pointer_chase.cpp`: the latency map. Dependent-load chains in a compute
  shader (one invocation walks a chain of next-indices; nothing can be prefetched), seq and
  random patterns at footprints from cache-sized to DRAM-sized, through device-local and
  host-visible memory, timed by timestamp queries and checked against the host's own walk.
  First verified numbers on the RX 9070 XT: VRAM DRAM hops ~254 ns, host-RAM hops over PCIe
  ~780-835 ns, and small host-visible footprints reaching the CPU's L3 (~112 ns).
- `backends/vulkan/mma.cpp`: tensor-core throughput per math format through
  VK_KHR_cooperative_matrix (the RDNA WMMA path), the open equivalent of the CUDA mma probe.
  fp16/bf16/int8 verify bit-exact and report TFLOPS; fp8 is driver-exposed but unreachable
  through glslang and is explicitly not claimed.
- `backends/vulkan/sync.cpp`: submit/fence completion lag. A device-side spin kernel is
  bracketed by timestamp queries (the device's real-nanosecond timeline) and the host measures
  submit-to-return spans; lag = host span minus device span. Durations 50 us to 60 ms.
  First verified numbers on the RX 9070 XT: lag p50 ~120-500 us across durations.
- `backends/vulkan/clock.cpp`: device clocks. Derives the GLSL shader clock's rate
  (ticks/ns) from a burn kernel bracketed by timestamps, idle vs loaded; cross-checks against
  amdgpu's pp_dpm_sclk (sysfs, declared context only).
- `backends/vulkan/file_page.cpp`: host-memory pages under the GPU. Imports an mmap'd region
  with VK_EXT_external_memory_host, then warm / resident-read / evict (madvise + mincore) /
  cold-pass / resident-again, every pass verified per 4 KiB row.
- `backends/vulkan/shaders/`: the read-fold, chase, cooperative-matrix, spin, clock-dual and
  row-read compute shaders.
- `backends/vulkan/common/echolock_vk.h`: the same JSONL record contract as the CUDA header,
  without any CUDA dependency.
- `backends/vulkan/common/vk_ctx.h`: shared Vulkan machinery (context, buffers with memory-type
  selection, pipeline from SPIR-V with optional pinned requiredSubgroupSize, timestamped
  submissions, host-pointer import). The two older probes predate it and carry their own copy.
- Vulkan headers vendored (`include/`, pinned to `vulkan-sdk-1.4.363`): no Vulkan SDK install
  needed, only the system loader (`libvulkan.so`).

## Findings from this backend (machine truths, not folklore)

- **The GLSL cooperative-matrix accumulate trap.** `coopMatMulAdd(x, y, z)` returns the new
  accumulator; it does not update `z` in place the way CUDA's `mma_sync` does. Discard the
  return value and the compiler dead-code-eliminates the entire multiply and stores the
  untouched zero initializer -- silent wrong answers, no warning anywhere. The disassembly
  shows a kernel that only stores zeros. This backend's first mma attempt had exactly that
  bug, and without the per-result oracle it would have printed confident garbage or been
  mistaken for a driver defect (which it initially was, in the notes of an earlier revision).
- **Cooperative matrix works on RADV GFX1201 once the setup is complete**, and every format
  the driver exposes except fp8 now verifies bit-exact: fp16 ≈ 102 TFLOPS, bf16 ≈ 125 TFLOPS,
  int8 ≈ 198 TFLOPS (measured with the shared exact pattern, every workgroup and accumulator
  chain checked per rep; DVFS moves reps by tens of percent, see the per_rep spread). Setup
  requirements, all mandatory:
  - pin `requiredSubgroupSize` (with the RequireFullSubgroups stage flag and the
    subgroup-size-control features); without it the pipeline misbehaves silently;
  - enable 16-bit/8-bit storage access, shaderFloat16/Int8, Vulkan memory model and bfloat16
    features explicitly -- RADV compiles the shader even when a feature is off, and the result
    is wrong;
  - compile with `--target-env=vulkan1.4` or later: glslang only emits the
    CooperativeMatrixOperands signed bits there, and without them the spec treats int8
    operands as unsigned (wrong results for any negative operand);
  - B loaded column-major reads element (k,n) at n*16+k -- a row-major fill is silently
    transposed.
- **fp8 is driver-supported but toolchain-unreachable.** RADV lists fp8 e4m3/e5m2
  cooperative-matrix configs (and RDNA4 has fp4 in the ISA, unexposed), but the GLSL frontend
  this project compiles with (glslang 16.6.1) cannot express fp8 cooperative matrices, and the
  system SPIR-V assembler is too old to rewrite one by hand. No fp8 number is claimed.
- **The GLSL shader clock is not real time on AMD.** Unlike CUDA's %globaltimer (nanoseconds
  since boot), clock2x32ARB ticks at the core-clock rate: the clock probe derives ~799 MHz at
  idle and ~1485 MHz under load on the RX 9070 XT. There is no GLSL-accessible nanosecond
  real-time clock; host<->device time mapping therefore uses timestamp queries (see sync).
- **RADV rejects imports of file-backed host memory.** VK_EXT_external_memory_host accepts
  anonymous mappings only: MAP_SHARED file mappings and memfd (tmpfs, i.e. actual page-cache
  pages) both fail with VK_ERROR_INVALID_EXTERNAL_HANDLE. CUDA's UVA maps any file. The
  file_page probe measures the rejection, then runs the evict/fault structure on an anonymous
  mapping; on this driver madvise DONTNEED does free the imported pages (mincore verified),
  and the cold pass recovers (27 vs 31 GB/s).
- **Submit->fence lag under RADV is ~0.1-0.5 ms** across 50 us to 60 ms kernels (sync probe,
  timestamp-bracketed device spans). The pipeline underneath is open: RADV submits through
  the amdgpu DRM scheduler, so the wait is a kernel uAPI sequence, not a binary blob.

## A subtlety worth knowing: HOST_CACHED

RADV's host-visible memory type for system RAM is HOST_VISIBLE|HOST_COHERENT but **not**
HOST_CACHED, and maps write-combined. Writes through it are fast; CPU *reads* are
pathologically slow. The first version of the CPU contention section read that mapping and
measured nothing (zero passes in 500 ms, while the same loop on ordinary memory does ~4800).
The probe now asks for a HOST_CACHED type for buffers the CPU streams (two-pass selection:
cached first, then relaxed) and records the chosen memory type per buffer. A GPU-side-only
measurement would never notice this; a CPU/GPU shared one cannot avoid it.

## A second subtlety: freeing descriptor sets

The first pointer-chase run failed after four series with `VK_ERROR_FRAGMENTED_POOL`: the probe
freed each series' descriptor set, but the pool was created without
`VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT`, which makes that free invalid -- RADV does
not return such sets, so the pool exhausted. The pool now carries the flag and is sized for a
whole run.
