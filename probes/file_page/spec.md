# file_page

What a GPU read of a memory-mapped file costs, with its pages resident in the page cache and
after they have been evicted (so every access goes through the fault path and storage).

CUDA: `backends/cuda/file_page.cu`. Origin: ninfer-gb10 `tools/gb10/file_page_probe.cu`.

## Stimulus

A scratch file (512 MiB default) holding a counter-based pattern: u32 word *w* = fmix32(*w* ⊕
seed). Rows are 2,560 B, the size of one ninfer PLE row, each at a 4 KiB-aligned page chosen by
splitmix64 with a fixed seed, so every phase and run reads the same rows. Phases run as separate
processes, so a hung cold phase is killed on its own timeout.

| Phase | What runs |
|---|---|
| prep | write + fsync the file |
| warm | host reads the whole mapping (populates the page cache) |
| res-seq | GPU streams the whole mapping |
| res-rows | GPU reads 100k random rows, one block per row |
| evict | `posix_fadvise(DONTNEED)` on the file (default), or read a >RAM file (`--evict PATH`, the original method) |
| cold-serial | one GPU thread reads 512 rows in order: serialized fault latency |
| cold-rows | GPU reads the 100k rows cold: parallel fault throughput |
| res-rows-again | same rows, resident again; must agree with res-rows |

The scratch file must be on a page-cache-backed filesystem, not tmpfs.

## Oracle

The expected XOR of every row, and of the whole file, is computed from the pattern function
without touching the file. After the warm-up launch the results are cleared, so only the timed
launch is checked. `mincore()` records the resident fraction before and after every phase, so a
"cold" phase that started warm is visible.

## Metrics

`gpu_seq_read` GB/s; `gpu_rows` µs/row, `gpu_rows_bw` MB/s; `gpu_serial_rows` µs/row; wall
times; `page_cache_resident` fraction; `host_warm_read`, `prep_write` GB/s; `evict_duration`.

## Silent fallbacks detected

`feature page_cache_eviction`: eviction was requested; the effect is observed only if under 1%
of the pages are still resident. A cold phase that starts with more than 1% of its pages resident
records the same feature as not observed.

## GB10 results so far

Resident: 162–169 GB/s. Cold: a single fault takes 96 µs; faults in parallel cost about 1 µs per
row. No hangs.
