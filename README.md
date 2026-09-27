# echolock

echolock measures what an edge GPU actually does rather than what it declares. Every probe sends
a known stimulus, checks the result against an independent oracle, and times the response. It
records the environment around each run: clocks, temperature and power every second, and kernel
Xid/OOM/hang lines. Results are time series, and a fast wrong answer counts as a failure.

GB10 (DGX Spark class, `sm_121a`) is the first target. Probe definitions are backend-neutral;
CUDA is the first backend and Vulkan compute the second. Design and principles:
[docs/brief.md](docs/brief.md).

## Status

| Milestone | State |
|---|---|
| 1. Skeleton, environment capture, Xid watcher, 1 s sampler | done; needs a first run on GB10 |
| 2. Port memory, file_page, sync with JSONL output | done (compiles for sm_121a; not yet run on GB10) |
| 3. Derived SM clock, pointer-chase latency map, per-format MMA throughput | not started |
| 4. Load generators and grid runner | not started ([load/](load/README.md)) |
| 5. Vulkan backend | not started ([backends/vulkan/](backends/vulkan/README.md)) |

## Quick start (on the GPU machine)

```sh
./echolock build                      # nvcc from PATH or /usr/local/cuda; ARCH defaults to sm_121a
./echolock run memory                 # -> runs/<utc>-memory-cuda-idle/ with report.md
./echolock run sync --pin-cpus 0,10   # also blocking-schedule runs pinned to CPU 0 and CPU 10
./echolock run file_page --scratch-dir /path/on/nvme --drop-caches
./echolock run memory -- --gib 2 --seconds 1          # after --: arguments for the probe
./echolock report runs/A runs/B       # several runs side by side (e.g. idle vs loaded)
```

Utilities: `./echolock env` prints the environment snapshot, `./echolock sample` streams 1 s
samples, and `./echolock kmsg` streams kernel GPU/OOM/hang events. Reading `/dev/kmsg` needs root
or `kernel.dmesg_restrict=0`; without it the run still happens, and the manifest records that the
kernel log was not watched. `--drop-caches` needs root or passwordless sudo.

Requirements: CUDA 13 toolkit (nvcc) and Python 3.11+ with the standard library only. The
probes link the CUDA runtime statically and reach the driver API through entry points, so they
need no `-lcuda` and no vendor libraries.

## Layout

```
probes/<name>/spec.md     stimulus, oracle, parameters, metrics (backend-neutral)
backends/cuda/<name>.cu   CUDA implementation; common/echolock.cuh holds the JSONL, oracle and clock helpers
backends/vulkan/          later
load/                     background load generators (milestone 4)
env/                      capture.py (versions, firmware, limits), sampler.py (1 s), kmsg.py (Xid watcher), nvml.py
run/                      plans.py (probe → process steps), runner.py (steps + env → run directory)
report/                   markdown.py: declared vs measured, one column per run label
docs/                     brief.md (design), record-format.md (JSONL schema)
tests/                    python3 -m unittest discover -s tests -t .
```

Each run writes `runs/<run_id>/` containing `manifest.json`, `env.json`, `probe.jsonl`,
`samples.jsonl`, `kernel.jsonl`, `stderr.log` and `report.md`. Every stream uses CLOCK_MONOTONIC
nanoseconds, so a kernel Xid lines up with the probe sample and the 1 s sample around it. Format:
[docs/record-format.md](docs/record-format.md).

## What changed from the ninfer-gb10 originals

- **Oracles.** Every timed read or copy is checked against an XOR the host derives from the
  generator. The file_page pattern is now counter-based, so the host knows every row's XOR
  without reading the file.
- **Per-launch series.** Every launch is timed on its own, where the originals reported a mean or
  a percentile.
- **Silent fallbacks.** Compression granted but not faster, and eviction requested but pages
  still resident (checked with `mincore`), are recorded as `feature` records.
- **Eviction.** file_page evicts with `posix_fadvise(DONTNEED)` by default instead of reading a
  file larger than RAM; the original method is still available as `--evict-file`.
- **Graph check.** In the sync probe each graph node increments a counter, which checks that all
  nodes ran.
