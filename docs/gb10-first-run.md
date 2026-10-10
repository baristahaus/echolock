# First GB10 run of the CUDA backend (sm_121a), and what it exposed

Run 2026-10-10 by twoFive (node 25 of an 8-node GB10 cluster), on a machine that was *not* the
author's box. Six probes, all passing, on the silicon this project was written for — plus five
findings, one of which corrected an experiment I ran myself.

## Provenance

| | |
|---|---|
| Machine | Asus GX10 (`gx10-4791`), NVIDIA GB10, `sm_121a`, 122 GiB unified |
| Commit built | `e7eee89` (clean, `dirty: false`) |
| Build | `make -C backends/cuda ARCH=sm_121a`, nvcc `cuda_13.0.r13.0`, 17 s, **0 warning/error lines** |
| Kernel / driver | `6.17.0-1029-nvidia` / `580.173.02`, VBIOS `9A.0B.25.00.00`, BIOS `GX10DGX.0105` |
| Host CPU | 10x Cortex-X925 @ 3900 MHz + 10x Cortex-A725 @ 2808 MHz (asymmetric) |
| Privileges | `sudo` for the run, because `dmesg_restrict=1` — so the Xid watcher was exercised too |
| Footprint | `--gib 1`, `--seconds 2`, scratch on local NVMe. **No `--drop-caches`** — see caveats |

This box was **not idle**: it hosts rank 2 of a live TensorFold TP-3 engine (96,046 MiB resident,
head at `172.30.30.26:8888`), and host `MemAvailable` was 8.3 GiB. That makes this a co-resident
measurement, not an idle baseline — which is arguably the more useful case for an edge box, and
the README already claims the probes can run beside a resident engine.

`--drop-caches` was deliberately left off. Dropping the page cache on a node whose engine may be
reading file-backed weights is a self-inflicted outage, not a measurement. Run that phase when the
engine is down.

## Outcomes

Ten runs (six probes, plus repeats for the CPU experiment below). Every one `status: ok`,
**110 checks, 0 failed**, kernel log available, **0 kernel events during any run**.

| Probe | Checks | Headline measured |
|---|---|---|
| memory | 13/13 | `gpu_read` median 243.5 GB/s (min 127.9, max 1873); `copy_payload` 52.9; `cpu_read` 100.7; **contention ratio 0.7** — a second reader costs ~30%; generic compression `granted=True, effect_observed=True` |
| clock | 5/5 | derived SM clock **2144.9 MHz**, declared/derived ratio 1.1, calibration `globaltimer_vs_host_ms` 10.0/10.0 |
| pointer_chase | 30/30 | `chase_hop_latency` 21.5 / 163.9 / 654.6 ns across device, pinned, pageable and file kinds; `l2_vs_dram_delta` 337 ns |
| sync | 44/44 | `sync_return_lag` median 2.9 us; `graph_launch_to_first_kernel` median 4.1 us; device schedule granted, effect observed in 3 of 4 trials |
| mma | 12/12 | fp16 105.1, bf16 105.1, **int8 105.1**, fp8 210.3, **fp4 210.3**, tf32 26.3 TFLOPS — fp4 works on `sm_121a` as the Makefile comment says, and int8 gets no 2x over fp16 here |
| file_page | 6/6 | host warm read 16.8-21.4 GB/s, prep write 2.9-3.2 GB/s, GPU seq read 28.8-161.8 GB/s, page-cache residency verified by `mincore` |

Interference, measured both ways on the live engine (32-token completion, same prompt):

```
before  the suite:  1.175s  0.955s  0.955s
during  memory run: 0.962s
after   the suite:  0.956s  0.954s  0.954s      engine /v1/models 200 after, local GPU util 0%
```

At `--gib 1` the probes cost the co-resident engine nothing measurable, and the engine cost the
probes little — the numbers above sit in a sane band next to the author's 5090 results.

Sampler coverage during the 10 runs (152 samples): SM clock 2132-2145 MHz, power 9.9-32.2 W,
`mem_clock_mhz` always `null`, `util_mem_pct` always `0`. See finding 3.

## Findings

**1. `host.cpu_model` was `None` on the project's primary target — fixed here.**
`/proc/cpuinfo` has no `model name` on Arm, so the field the report reads was empty on GB10 while
the answer was already in the record, per CPU, as MIDR part numbers (`host.cpus[].part`). Now
`arm_cpu_model()` decodes those into `Cortex-A725 x10 + Cortex-X925 x10`, verified against `lscpu`
on this box. Unknown parts keep their hex rather than getting a confident wrong name. DMI was
already captured fine (`ASUSTeK GX10`, BIOS dated 2026-05-05), and `devicetree_model` is null
because this box boots via UEFI — that one is correct, not a bug.

**2. `--pin-cpus` was accepted by every probe and honoured by one — now it refuses. Fixed here.**
Only the CUDA `sync` plan builds `taskset` steps. `./echolock run memory --pin-cpus 0` ran
unpinned, recorded nothing about the flag, and looked pinned in shell history. A silently ignored
measurement flag is the same class of bug as a fast wrong answer, so `run` and `crosscheck` now
exit 2 and point at the manual form. `tests/test_plans.py` checks the guard against what
`make_plan` really builds, so the two cannot drift apart. The remaining half of this is still open:
the manifest does not record the pinning it *does* apply — the step name (`cpu=0`) is the only
trace. Put `pin_cpus` and each pinned CPU's `cpu_capacity` in the manifest, so a pinned run is
self-describing and cross-cluster comparisons are visible without reading step names.

**3. On GB10 the sampler reports `mem_clock_mhz: null` and `util_mem_pct: 0`, always.**
Unified LPDDR5x has no memory clock, and NVML's memory-utilisation is unimplemented here. A flat
`0` is worse than a gap: read straight off a chart it says "the memory bus is idle", on the one
platform where memory bandwidth is the whole story (this box also reports no power limit, so draw
and energy are all you get). Suggest an explicit `unsupported` marker on unified-memory parts, or
at least a line in `docs/record-format.md` so nobody builds a claim on that zero.

**4. The derived-clock trace was perfectly flat, which may be the probe failing quietly.**
480 trace samples, every one 2144.9 MHz, while the independent sampler path (NVML) saw
2132-2145 MHz over the same window. Two instruments agreeing to 0.6% is good news for the method —
but a trace with zero variance cannot show the throttle step the probe exists to catch, and a
constant is exactly what a too-short spin or a mis-scaled ratio looks like. Worth checking the
spin length, and cheap to make self-checking: `report/` could compare the probe's derived clock
against the sampler's `sm_clock_mhz` for the same window and flag disagreement. That keeps the
probes NVML-free as designed while still cross-checking, and it would have caught this by itself.

**5. An XOR fold cannot see order — fine for bandwidth, a hole for `pointer_chase`.**
XOR is commutative, so the oracle proves the right *volume* of data was touched, not that it was
touched in the right order. That is the right trade for the memory probe, and the per-rep nonce is
a genuinely good touch (a zero-filled buffer XORs to zero, so without it "read nothing" passes as
"read zeros"). But in `pointer_chase` the traversal order *is* the measurement, so a complete
wrong-order walk passes the oracle and still prints a plausible latency map. Recording the visited
chain (first node, last node, hop count) closes it for two lines of code. Worth one sentence in
`docs/record-format.md` either way, so nobody reads "verified" as "ordered".

**6. Minor.** The README's layout block listed `load/` and `run/` twice with a stray
`(AMD via sysfs)` fragment — fixed. The build is warning-clean under `-Wall -Wextra` on two
different GB10 boxes, which is worth pinning: add `-Werror` to the CI compile job before the next
contribution ends that. And the repo's only branch is `claude/vigilant-faraday-u4izq6`, so every
link and PR target is an opaque session name; renaming it `main` costs nothing today and less pain
later.

## What I got wrong first, recorded because it is the useful part

I first tested finding 2's underlying worry — that pinning across GB10's two CPU clusters
distorts host-side timings — by running `memory` and `file_page` pinned to cpu0 (little) and
cpu15 (big). The spread looked like an effect: `prep_write` 2.89 vs 3.22 GB/s, cold-fault
throughput 33.1k vs 40.9k pages/s.

**Both runs were unpinned.** The flag was silently ignored (finding 2), so what I had measured
was run-to-run noise on a box with a live engine next door, and I nearly wrote it up as a cluster
effect. The check that caught it was reading the manifest rather than trusting my own command
line: `pin_cpus` was not there, and `planned_steps` for the memory run was just `["memory"]`.

So the asymmetric-host question stays **open, not disproved**: `file_page`'s host phases and
`pointer_chase`'s pageable/file kinds are CPU-bound and could genuinely care which cluster they
run on. To settle it, interleave three runs of each through `taskset` directly and take medians —
`taskset -c 0 ./build/cuda/file_page --phase warm ...` versus `-c 15` — and only claim an effect
bigger than the repeat spread, which on this box was around 10-20%.

## One scare, and what it actually was

`tests.test_cuda_device` failed once, on this node, on the clock probe's `load.xor` check — the
oracle reporting a 214,748,364-byte buffer read back wrong. That is the worst kind of failure this
project can report, so it got chased first. It was not a defect: the `build/cuda/` binaries were
built at 21:18 from `c762923`, two pulls behind the source the freshly cloned test suite expected.
After `./echolock build` at HEAD it passed four consecutive runs, including with the node's GPU at
41% utilisation next to a resident engine. The record made that visible in one command —
`build.json` and `env.json` both carry the commit and flags the binaries were built with, which is
exactly why they should stay in every run directory.

## Reproducing

```sh
./echolock build && sudo ./echolock run memory -- --gib 1 --seconds 2   # and clock, sync, mma,
                                                                        # pointer_chase, file_page
```

Run directories: on the test node, `~twoEight@192.168.0.28:~/echolock/runs/`; mirrored to
`192.168.0.25:~/echolock-runs-gb10-20261010/runs/` (10 runs, 6.6 MB), with the driver script that
produced them at `~/echolock-runs-gb10-20261010/test-driver.sh`. Both are on disk, not in `/tmp`.
