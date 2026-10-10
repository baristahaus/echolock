# pointer_chase

Memory latency as a dependent-load chain sees it: one load per hop, each depending on the
previous, through each kind of memory an integrated GPU can read, at footprints that fit and
overflow the cache. The cold variant chases through evicted file pages, so a page fault's cost is
measured by the GPU's own clock at pointer-chase granularity.

CUDA: `backends/cuda/pointer_chase.cu`.

## Stimulus

| Part | What runs |
|---|---|
| Chains | An array of u32 indices where `chain[i]` is the next position. A single thread walks `hops` nodes: `i = chain[i]`, accumulating the loaded value. Every hop is a dependent load; nothing can be prefetched. |
| Footprints | Chain arrays sized from a few KiB (fits L1/L2) to hundreds of MiB (DRAM): the same walk measures cache-hit and DRAM latency. |
| Patterns | Sequential stride (predictable addresses) and a splitmix64 permutation (uniformly random): prefetching cannot help a dependent chain, but the memory system's row-buffer behaviour can. |
| Memory kinds | `cudaMalloc` (device), `cudaHostAlloc` mapped (pinned host), `malloc` pageable (when the device declares pageable access), and a memory-mapped file (host pages, resident and evicted). |
| Evicted pages | The file-backed chain is walked once after `posix_fadvise(DONTNEED)` drops its pages: each first-touch hop pays a page fault, timed by the GPU. |

## Oracle

The chain's contents are a known function of position, so the host knows the XOR of the visited
nodes. The walk's accumulated XOR must match, and the number of nodes visited must equal `hops` --
a load that returned stale or wrong data fails instead of posting a fast latency. The final index
is stored to memory, so the chain cannot be optimized away. Chains are validated on the host
after construction (each node reachable, exactly one cycle).

## Parameters

`--footprints` MiB list (e.g. `0.0625,1,64`), `--hops` per measurement (4096), `--kinds`
(device,pinned,pageable,file), `--file PATH`, `--bytes` file size (512M), `--reps` per config (5),
`--pattern` (seq, random).

## Metrics

- `chase_hop_latency_ns` per hop (walk time / hops), tagged `kind`, `footprint_bytes`, `pattern`.
  One record per rep plus a summary.
- `cold_fault_hop_ns`: the evicted-file walk, where a hop can include a 4 KiB page fault. The
  hop rate (faults/s) is derived from unique pages on the chain, since two hops in one page pay
  one fault.
- `l2_vs_dram_delta`: the latency difference between the smallest and largest device footprint,
  which is the cache-latency contribution the bandwidth probes cannot show.

## Relation to the other probes

`memory` measures how many bytes per second; this measures how long one dependent load takes. The
two together distinguish bandwidth-bound from latency-bound behaviour on the same hardware.
`file_page` measures cold faults at 2.5 KiB row granularity; this measures them at hop
granularity, where a single fault is not amortized by anything.
