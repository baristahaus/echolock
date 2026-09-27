# memory

What the GPU and the CPU can read from each kind of memory an integrated GPU sees, how they share
one memory budget, and whether requested compression does anything.

CUDA: `backends/cuda/memory.cu`. Origin: ninfer-gb10 `tools/gb10/memory_probe.cu`.

## Stimulus

| Part | What runs |
|---|---|
| Compression | Two VMM allocations, one plain and one requesting generic compression. Each is filled with zeros, then with a random pattern, and read in full by a streaming kernel. |
| Compression, real bytes | `--sample LABEL=PATH`: the file's bytes tiled over both allocations, then read. |
| Memory kinds | Streaming GPU reads of device memory, pinned mapped host memory and pageable `malloc` memory (the last only when pageable access is declared). |
| Copies | Host-to-device copies from pinned and from pageable memory. |
| Contention | CPU threads read pageable memory, then the GPU reads device memory alone and again while the CPU keeps reading. |

## Oracle

Every read folds all loaded words into a 128-bit XOR. The expected XOR comes from the generator
on the host (random fill: a counter-based xorshift per 16-byte word; zeros: 0; tiled samples:
whole periods cancel, the remainder is computed). Host-filled buffers are XORed on the host. Every launch
writes its own result slot, so every timed launch is checked. After each copy, an untimed
device read of the destination is checked. Each CPU read pass is checked against its slice XOR.
A launch whose XOR is wrong is a failure, not a bandwidth.

## Parameters

`--gib` buffer size (1), `--seconds` per series (0.5), `--cpu-threads` (all), `--sample`.

## Metrics

- `gpu_read` GB/s, tagged `source` (vmm_plain, vmm_compression_requested, cudaMalloc,
  cudaHostAlloc_mapped, malloc_pageable), `fill` (zero, random, sample:LABEL), `contention`.
  One record per launch, plus a summary.
- `copy_payload` GB/s per copy. The payload rate; DRAM traffic is about twice that.
- `cpu_read` GB/s, alone and while the GPU reads.
- `shared_sum`, `gpu_read_contention_ratio`.
- `compression_gain` per sample.

## Silent fallbacks detected

`feature generic_compression`: granted by `cuMemGetAllocationPropertiesFromHandle` and effective
only if zero-filled data reads at least 1.2× faster from the compressed allocation than from the
plain one.

## GB10 results so far (ninfer-gb10, driver 580.173.02, CUDA 13.0.88)

Device read 246 GB/s (spec 273). The GPU reads pinned host memory at 174–216 GB/s and pageable
at 161 GB/s; host-to-device copies run at 59 GB/s. CPU and GPU share one ~241 GB/s budget.
Compression reads zero-filled data 8.4× faster but real weights only 1.00–1.01×.
