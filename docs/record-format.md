# Record format (v1)

Every stream is JSONL: one JSON object per line. Every record carries:

| Field | Meaning |
|---|---|
| `v` | format version, 1 |
| `t_ns` | CLOCK_MONOTONIC, ns. Probes, runner, sampler and kernel log all use it, so their streams merge on it. |
| `src` | `probe`, `runner`, `sampler` or `kernel` |
| `type` | record type, below |

The runner adds `run_id` and `step` to every probe record it stores.

## Probe records (`src: probe`, plus `probe` and `backend`)

| type | Fields | Notes |
|---|---|---|
| `probe_start` | `params` | first record |
| `declared` | `key`, `value`, `source` | what an API reports; chooses probes, never replaces a measurement |
| `measure` | `name`, `value`, `unit`, `tags`, `verified`, `rep`? | `rep` present: one element of a series. Absent: the series summary (`tags.reps`, `tags.per_rep` {min,p50,p90,max,mean}). `verified` false means the oracle rejected this work, so the value is not a result. |
| `check` | `name`, `pass`, `detail` | an oracle verdict; any failure makes the probe exit 3 |
| `feature` | `feature`, `requested`, `granted` (bool or null), `effect_observed`, `evidence` | requested + granted + not observed = silent fallback |
| `note` | `message` | |
| `error` | `message` | followed by `probe_end` with status `error` |
| `probe_end` | `status` (`ok`, `verify_failed`, `error`), `failed_checks` | last record |

Exit codes: 0 ok, 1 runtime error, 2 usage, 3 verification failed.

## Runner records (`src: runner`, in probe.jsonl)

`step_start` {step, argv, drop_caches}, `step_end` {step, status, returncode, duration_s,
timed_out, unkillable, interrupted}, `abort`, `stdout_text` {text}: a probe line that wasn't JSON.

Step status is `ok`, `verify_failed`, `error`, `timeout` or `incomplete` (exit 0 with no
`probe_end`). A run's status is its worst step's.

**Interrupted runs.** Ctrl-C or `kill` (SIGINT, SIGTERM, SIGHUP) stops the step's process group --
the step runs in its own session, so the signal would otherwise not reach it and it would go on
holding the GPU after the runner exited -- writes an `abort` record, and still writes
`manifest.json`, whose `status_note` names the signal and how many steps ran. The interrupted step
is listed with status `error` and `interrupted` true. `./echolock run` then exits 130 (Ctrl-C) or
143 (`kill`). Everything recorded up to that point is intact.

## Sampler records (`src: sampler`, `type: sample`, samples.jsonl)

`gpu_source` is `nvml`, `amdgpu_sysfs` (the amdgpu driver's own sysfs files, no library
involved) or `drm_sysfs`. `gpu` has one entry per GPU. From NVML: `sm_clock_mhz`,
`graphics_clock_mhz`, `mem_clock_mhz`, `temp_c`, `power_w`, `energy_mj`, `util_gpu_pct`,
`util_mem_pct`, `pstate`, `clock_event_reasons` (raw mask) and `clock_event_names`. From
`amdgpu_sysfs`: `sm_clock_mhz`, `mem_clock_mhz`, `temp_c`, `power_w`, `energy_mj`,
`util_gpu_pct`, `util_mem_pct`, `pstate` (`D0`/`D3`), `vram_used_bytes`, `gtt_used_bytes`,
`card`, and `clock_event_reasons` as the raw `gpu_metrics` throttle word
(`clock_event_names` is null there: the bit meanings are asic-specific). From DRM, whatever the
driver exposes. `host`: `cpu_khz` (per CPU), `thermal_c`, `hwmon`, `mem` (/proc/meminfo, KiB),
`psi_memory`, `psi_cpu`, `loadavg`. `sample_cost_us` is how long the sample took to take.

## Kernel records (`src: kernel`, `type: kernel_event`, kernel.jsonl)

`category` (`xid`, `sxid`, `nvrm`, `gpu`, `reset`, `oom`, `hang`, `iommu`, `mce`), `priority`,
`seq`, `message`, and for Xid lines `xid` and `device`. `t_ns` comes from the kernel's own
timestamp. `reset` covers a driver resetting the device, which means work was discarded: on AMD
`gpu reset begin`, on NVIDIA a bus reset. Only events logged after the watcher started are
written. Earlier ones are counted in `manifest.kernel_log.before_run`.

## Run directory

`manifest.json`, `env.json`, `probe.jsonl`, `samples.jsonl`, `kernel.jsonl`, `stderr.log`,
`report.md`. A directory missing or broken in any of these still reports: the run's status cell
says `unreadable` plus what could not be read, and whatever streams survived are laid out as
usual. Only `probe.jsonl` carries the measurements, so that is the file worth recovering.
