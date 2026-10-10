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

## What is implemented

- `backends/vulkan/memory.cpp`: the memory probe. Device-local and host-visible buffers, read
  dispatches timed by `VK_QUERY_TYPE_TIMESTAMP`, every dispatch's XOR checked against the host's
  expectation with a per-dispatch nonce folded into the result slot (an unwritten slot cannot
  pass), copies verified by a read of the destination, and a CPU/GPU contention section where
  every CPU pass is checked.
- `backends/vulkan/shaders/read_fold.comp`: the XOR-fold compute shader.
- `backends/vulkan/common/echolock_vk.h`: the same JSONL record contract as the CUDA header,
  without any CUDA dependency.
- Vulkan headers vendored (`include/`, pinned to `vulkan-sdk-1.4.363`): no Vulkan SDK install
  needed, only the system loader (`libvulkan.so`).

## A subtlety worth knowing: HOST_CACHED

RADV's host-visible memory type for system RAM is HOST_VISIBLE|HOST_COHERENT but **not**
HOST_CACHED, and maps write-combined. Writes through it are fast; CPU *reads* are
pathologically slow. The first version of the CPU contention section read that mapping and
measured nothing (zero passes in 500 ms, while the same loop on ordinary memory does ~4800).
The probe now asks for a HOST_CACHED type for buffers the CPU streams (two-pass selection:
cached first, then relaxed) and records the chosen memory type per buffer. A GPU-side-only
measurement would never notice this; a CPU/GPU shared one cannot avoid it.

## Not yet implemented

The latency map (pointer chase through a storage buffer — dependent loads work in a compute
shader; file-backed pages have no Vulkan equivalent) and the sync probe (queue/fence completion
lag). On NVIDIA, comparing this backend against CUDA on the same machine is the design's
cross-check: where the two disagree, that points at a driver.
