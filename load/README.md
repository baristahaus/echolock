# load

Background load generators: the grid's second axis (milestone 4). Each is its own process,
started by the grid runner before a probe run and stopped after it. A load records what it
intended and what it achieved in JSONL on stdout, so "50%" can be checked instead of assumed.

Record protocol (all records `src: "load"`, on the load's stdout, archived per run as
`load.jsonl`):

| type | Fields |
|---|---|
| `load_start` | `load`, `intended`, `args`, `deadline_min` |
| `load_ready` | `achieved` — steady state reached; the grid waits for this before starting the probe |
| `heartbeat` | `achieved` — every few seconds while holding |
| `load_end` | `achieved` — on SIGTERM, deadline, or dead runner |
| `load_error` | `message` |

## Generators

| Name | Condition | What it does |
|---|---|---|
| `cpu` | `cpu50` | N threads doing GIL-releasing sha256 work at a duty cycle; achieved duty from /proc/self/stat, so 16 threads at 50% really occupies 8 cores. |
| `mem` | `mem80` | Touch a fraction of physical RAM (one byte per 4 KiB page) and hold it; ready only when the touch pass finished, resident size from /proc/self/statm. |
| `io` | `io500` | Sequential file reads throttled to a target MB/s; whether the reads hit disk depends on file size vs page cache, and the record says which case it is. |
| `gpu` | `gpu50` | `backends/cuda/gpuload`: duty-cycled streaming reads, every busy cycle XOR-checked, achieved duty and bandwidth measured from the kernel's own timing. |

## Safety

A load must never outlive the runner that started it. Three mechanisms, in order:

1. The runner terminates the load after the probe run (SIGTERM, then SIGKILL).
2. The load watches the runner's pid (`--parent-pid`, `/proc/<pid>` liveness) and exits on its
   own. `getppid()` alone is not enough: a load that starts after its parent died captures the
   reaper's pid and never notices.
3. Every load has a deadline (`--deadline-min`, default 120).

## Use

The grid runner (`run/grid.py`, `./echolock grid`) picks the generator from the condition name:

```sh
./echolock grid memory --conditions idle,cpu50,mem80   # one run per condition, side-by-side report
./echolock grid file_page --conditions idle,gpu50 --scratch-dir /path/on/nvme
```

Each run directory gets `load.jsonl`; the report's "During the run" section shows what the load
intended and what it achieved.
