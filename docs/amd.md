# AMD enablement

How much of AMD an edge-GPU instrument can use without any closed code, what this repository
reads today, and what a probe backend would need. Written on a machine with two AMD GPUs
(`1002:13c0` iGPU, `1002:7550` dGPU with 16 GiB, amdgpu, kernel 7.2.7); the sources checked are
listed at the end.

## What AMD makes available

| Layer | State | What it gives echolock |
|---|---|---|
| `amdgpu` kernel driver | GPL, in the kernel | Everything the environment layer reads. No ROCm, no daemon, no vendor library. |
| sysfs ABI | Documented in `Documentation/ABI/testing/sysfs-driver-amdgpu` | The stable promise: clocks, busy percentages, VRAM, power state, DPM tables, features. |
| `gpu_metrics` | One binary file per card, versioned struct in `drivers/gpu/drm/amd/include/kgd_pp_interface.h` | Energy counter, throttle word, per-engine activity. |
| AMD SMI (`ROCm/amdsmi`) | MIT, `libamd_smi.so` + Python + `amd-smi`; installable without the rest of ROCm | **Not used.** Its sysfs backend is the same data; it would add a dependency and an init cost per sample for nothing. `env/nvml.py`'s equivalent would be ~60 ctypes lines if it ever exposes a field sysfs does not. |
| ROCm runtime + `rocprofiler-sdk` | Open (MIT) | Kernel launches and performance counters, if a probe ever needs them. |
| RADV (Mesa) | Open source; AMD discontinued AMDVLK and moved to RADV | The compute surface for the milestone-5 probe backend. |
| ISA and architecture docs | Published, including **machine-readable ISA (XML)** and per-IP performance-counter documentation | Assembling probe kernels, and knowing what a counter actually counts. |

The important difference from NVIDIA: on AMD the whole path to the hardware is open source, so a
disagreement between two AMD paths is evidence about the hardware rather than about a driver's
mood. That makes the Vulkan-vs-CUDA comparison idea stronger on AMD than on NVIDIA, where the
Vulkan driver is closed.

## What this repository reads now

`env/amdgpu.py` (sysfs only, standard library only). Used by:

- **the 1 s sampler** — `gpu_source` becomes `amdgpu_sysfs` when NVML is absent and an amdgpu card
  exists, otherwise the chain stays NVML → DRM sysfs. Samples carry `sm_clock_mhz`,
  `mem_clock_mhz`, `temp_c`, `power_w`, `energy_mj`, `util_gpu_pct`, `util_mem_pct`, `pstate`,
  `vram_used_bytes`, `gtt_used_bytes`, and the throttle word with its bit names
  (`clock_event_names`) where a per-ASIC table exists — gfx10.3 and gfx12 — or null on other
  families, under the same key names the NVIDIA path uses so the report needs no AMD branch.
  The generic hwmon sensor set is discovered once per process; its slow sensor values are
  re-read at the 60 s rescan rate without the directory walk.
- **the environment snapshot** — `env.json` gains an `amd` section: driver version, the module
  parameters that change behaviour, KFD topology (CU count, waves per SIMD, LDS size, memory
  banks), and per card the ids, VBIOS, VRAM and vendor, PCIe current against max link, the DPM
  clock tables, the forced performance level, whether an overclock offset is applied, the
  power-gating features that are on, and `gpu_metrics` decoded.
- **the kernel log watcher** — `reset` is now a category of its own, and the amdgpu error forms
  (`ring <x> timeout`, `ring <x> trigger`, `MODE: (N)`) match. On a machine with no NVIDIA
  driver, this is what tells you a run was garbage.
- `report/markdown.py` prints the AMD lines in the Environment section, including the sclk table
  and the forced level, because "the clocks were locked" is the first explanation to eliminate.

Cost: on this box (both cards) the per-sample cost is dominated by the per-card amdgpu sysfs
reads, not the generic hwmon walk — a 30 s run measured 14.4 ms p50 before the hwmon cache and
14.0 ms after, with one 45 ms discovery sample at the start; the cache removes the walk from the
60 s rescan, so long runs stop paying it repeatedly.

### Why `gpu_metrics` is cross-checked

`gpu_metrics` starts with `{size, struct_version, struct_revision}`, and the field layout differs
per `(version, revision)`. Rather than ship a table of layouts and hope, `env/amdgpu.py` decodes
the v1 prefix, then demands agreement with two sources that are independent of it: the clock must
be a level the `pp_dpm_sclk` table lists, and the edge temperature must match hwmon. If neither
passes, the decode is dropped and the reason recorded:

```
"gpu_metrics": {"error": "layout v2.1 did not cross-check, not trusting it", "cross_check": {...}}
```

On the test machine both cards decode and cross-check: the dGPU (struct 1.3, 120 bytes) matches
hwmon on edge temperature and both clock fields, and the iGPU (struct 2.1, also 120 bytes) decodes
through the family-2 layout from `kgd_pp_interface.h` and cross-checks on its own terms —
`average_gfxclk_frequency` is a level in the card's DPM table, and `temperature_gfx` (centi-
Celsius on family 2) matches hwmon's edge temperature within 0.5°C. A struct whose family has no
verified layout is refused with a record that says so — the alternative to a verified layout is a
table of guessed offsets.

Fields the driver fills with all-ones mean "this GPU has no such sensor"; they are dropped rather
than reported.

## Declared values that are worth recording on AMD

These are the knobs that silently change a measurement, so `env.json` carries them:

- `power_dpm_force_performance_level` — a locked `level N` makes the clock a constant of your own
  choosing, and every "measured clock" in the report a tautology.
- `pp_od_clk_voltage` — the overclock table. Its mere presence is not an overclock (some parts
  print the table and range with nothing applied), so `overclock_applied` looks for an offset.
- `pp_features` — GFXOFF, DS_*, BACO, THROTTLERS, spread spectrum. GFXOFF in particular decides
  whether the first kernel of a step pays a wake-up.
- `cg_mask`, `runpm`, `gpu_timeout`, `noretry` from the module parameters.
- KFD topology: CU count, waves per SIMD, LDS size, GWS, memory banks. This is AMD's closest
  thing to `cudaGetDeviceProperties`, and it is plain text.

## Gaps

1. ~~`throttle_status` bit meanings are asic-specific, so it is stored raw and
   `clock_event_names` stays null.~~ **Closed for this box's ASICs** (2026-11-06). The driver's own
   bit tables are transcribed into `env/amdgpu.py` — gfx10.3 (SMU v12, `renoir_ppt.c` +
   `smu12_driver_if.h`) and gfx12 (MP1 v14.0.2, `smu_v14_0_2_ppt.c` +
   `smu14_driver_if_v14_0.h`) — and the card's family is read from KFD's `gfx_target_version`
   (matched to the card by PCI address), so `clock_event_names` carries the active names: this box
   idled with the iGPU at 0x800 = `EDC_CPU` and the dGPU with nothing active. Other families still
   store the word raw with null names — the table is per-ASIC, and guessing would be worse.
2. ~~No `gpu_metrics` layout is verified for the family-2 struct.~~ **Closed** (2026-11-06). The
   v2.1 layout from `kgd_pp_interface.h` is in `env/amdgpu.py`; the iGPU decodes and cross-checks
   (see [Why `gpu_metrics` is cross-checked](#why-gpu_metrics-is-cross-checked)).
3. ~~The sampler's generic hwmon walk should be cached.~~ **Closed** (2026-11-06). Discovery now
   runs once per process and the 60 s rescan re-reads only the slow sensor values, without the
   directory walk. The per-sample p50 barely moved (14.4 → 14.0 ms over a 30 s run), because on
   this box the per-card amdgpu reads — not the walk — dominate the sample.

## What an AMD probe backend uses

**The Vulkan backend now exists and has run on AMD hardware** (milestone 5, first probe):
`backends/vulkan/memory.cpp` measures GPU reads of device-local and host-visible memory, copies
and CPU/GPU contention through RADV, with the XOR oracle and `VK_QUERY_TYPE_TIMESTAMP` timing.
First verified numbers on the RX 9070 XT in this box: 626 GB/s VRAM reads, 53.6 GB/s host-RAM
reads over PCIe, 44.8 GB/s host-to-device copies, CPU reads of the same host buffer at
159.5 GB/s, and GPU-under-CPU-contention ratio 1.01 (independent resources on a dGPU).

One AMD-specific subtlety it caught: RADV's host-visible system-RAM memory type is
HOST_VISIBLE|HOST_COHERENT but not HOST_CACHED, and maps write-combined — writes through it are
fast, CPU *reads* pathologically slow. The first CPU contention section read that mapping and
measured nothing (zero passes in 500 ms, where the same loop on ordinary memory does ~4800).
The probe now selects a HOST_CACHED type for buffers the CPU streams. A GPU-side-only
measurement would never notice this; a CPU/GPU shared one cannot avoid it.

`./echolock run memory --backend vulkan` is the way to run it. The rest of the original plan,
for what Vulkan has not covered yet:

1. **Latency map**: done -- `backends/vulkan/pointer_chase.cpp`, dependent-load chains in a
   compute shader through device-local and host-visible memory. First verified numbers on the
   RX 9070 XT: VRAM DRAM hops ~254 ns, host-RAM hops over PCIe ~780-835 ns, small host-visible
   footprints reaching the CPU's L3 (~112 ns). File-backed pages have no Vulkan equivalent, so
   the cold-fault measurement stays CUDA-side on GB10.
2. **Completion latency**: `vkQueueSubmit` + `vkWaitForFences`, the direct analogue of the sync
   probe. Not yet implemented.
3. **Derived clock (principle 3)**: AMD GPUs have `s_get_breg_b32` for the shader clock counters
   and `s_memrealtime` for a nanosecond timestamp. Both are in the machine-readable ISA — read it,
   don't repeat folklore, and check the result against `pp_dpm_sclk` the same way the environment
   layer now checks `gpu_metrics`.
4. **KFD directly** (`/dev/kfd` ioctls via rocm-thunk headers) if a probe needs
   instruction-level control Vulkan will not give, e.g. the pointer-chase latency map.
5. **A HIP/ROCr port** only if a question is left open that Vulkan cannot answer; it is the
   heaviest dependency of the five.

## Sources

- Kernel: `Documentation/ABI/testing/sysfs-driver-amdgpu`,
  `drivers/gpu/drm/amd/include/kgd_pp_interface.h` (gpu_metrics_v1_0 … v3_0), read 2026-10-05.
  Throttle bit tables and the decimal `gfx_target_version`: `pm/swsmu/smu12/renoir_ppt.c`,
  `pm/swsmu/inc/pmfw_if/smu12_driver_if.h`, `pm/swsmu/smu14/smu_v14_0_2_ppt.c`,
  `pm/swsmu/inc/pmfw_if/smu14_driver_if_v14_0.h`, `amdkfd/kfd_device.c` (torvalds/linux, read
  2026-11-06).
- AMD SMI: `github.com/ROCm/amdsmi` (MIT; `include/amd_smi/amdsmi.h` — `amdsmi_init`,
  `amdsmi_get_socket_handles`, `amdsmi_get_gpu_metrics_info`, `amdsmi_get_clock_info`,
  `amdsmi_get_power_info`), and rocm.docs.amd.com/projects/amdsmi for the standalone package.
- `github.com/ROCm/rocprofiler-sdk` (MIT).
- RADV: docs.mesa3d.org/drivers/radv; AMD's move from AMDVLK to RADV, and RDNA3/RDNA4 support
  landing in Mesa, per Phoronix/Mesa 25.x–26.0 coverage.
- ISA and counters: gpuopen.com/machine-readable-isa, and the AMD GPU architecture documentation
  index (ISA references, white papers, performance-counter docs) on the Instinct/ROCm docs.
