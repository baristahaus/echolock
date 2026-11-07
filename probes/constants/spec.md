# constants

The device's declared facts — name, memory size, compute units, L2 size, max clock — each read
from every independent source the machine exposes, with agreement required. Every other probe
divides a number by one of these facts, so a disagreement is a finding and a missing source is
an honest record, never a guess. No GPU work: the probe is a set of reads.

CUDA: `backends/cuda/constants.cu`. Vulkan: `backends/vulkan/constants.cpp`.

## Sources and oracle

| Key | CUDA sources | Vulkan sources |
|---|---|---|
| `device_name` | `cudaGetDeviceProperties.name`; the "Model:" line of `/proc/driver/nvidia/gpus/*/information` (first PCI-order dir = device 0) | `vkGetPhysicalDeviceProperties.deviceName` (single source; the KFD node name is a gfx-family format, not comparable) |
| `memory_bytes` | `.totalGlobalMem`; the "Video Memory:" line of the same file (MiB -> bytes) | sum of device-local heap sizes; the matching KFD node's VRAM mem bank (`heap_type 1`, `size_in_bytes`); `mem_info_vram_total` of the matching drm card |
| `sm_count` | `.multiProcessorCount` | the matching KFD node's `cu_count` file, else `array_count * cu_per_simd_array` from its `properties` |
| `l2_bytes` | `.l2CacheSize` | none |
| `clock_max_mhz` | `cudaDeviceGetAttribute(ClockRate)` kHz -> MHz | the highest level of the matching card's amdgpu `pp_dpm_sclk` table |

One `declared` per (key, source); one `check` per key with >=2 comparable present sources
(`constants.<key>.agreement`, detail = {source: value}). Rule: exact equality for integers; for
names, trim and drop the leading vendor word ("NVIDIA ", "AMD ", "Intel ") on both sides, then
exact equality; for `l2_bytes`, the band rule below. Vulkan side: the device's PCI device ID
(`vkGetPhysicalDeviceProperties.deviceID`) picks the KFD node whose `properties` `device_id` is
that ID (GPU nodes only, `vendor_id != 0`) and the drm card whose PCI device ID is that ID
(first match wins; this box: device 0 = 0x7550 = card1 = KFD node 1, device 1 = 0x13c0 =
card0 = KFD node 2). If the device-local heap (or the KFD bank) exceeds the amdgpu carve-out,
the device is integrated (UMA): the heap and bank size the host memory pool, the carve-out is
the device's VRAM, and all sources are declared but no agreement check is applied (different
quantities).

## Derived sources (`--against RUN_DIR`, optional)

- **pointer_chase inflection**: the run's `chase_hop_latency` summary measures (tags carry
  `per_rep`), device-local kind, pattern `seq` (the documented choice: the baseline runs carry
  both `seq` and `random`, and `seq` makes the inflection a pure cache-size effect), ascending
  footprint. Inflection = the first footprint whose value exceeds 2x the smallest's; band =
  (footprint strictly before the inflection, footprint at the inflection). With a declared L2
  (CUDA): `constants.l2_bytes.agreement` passes iff the band strictly contains `l2_bytes`.
  Without one (Vulkan): a note with the band, data for the future, not a check.
- **clock per-SM coverage**: the distinct `sm` tags of the run's per-SM clock measures, a second
  source for `constants.sm_count` (exact equality). A missing or wrong run dir is a note.

## Parameters and records

`--against RUN_DIR` (optional) on both backends; `--device N` on Vulkan (0/1), CUDA runs
device 0. A `feature` `constants.<key>.sources` (effect_observed null) for every key the spec
lists fewer than two sources for, or with fewer present sources than listed; a `note` per
failed or skipped source read. Exit 0 with the honest record set: the closest thing to a pure
oracle in the domain, and its value is the finding when sources disagree.
