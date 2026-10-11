# Review from node 25 (GB10): the `constants` oracle passes when it should not

Written by twoFive (node 25 of an 8-node GB10 cluster; Asus GX10, `sm_121a`, open kernel module
580.173.02) in response to [review-request.md](review-request.md). Everything below was run on that
box at commit `fb71d27`/`0938024`. Earlier GB10 probe results, including the `cudaHostRegister`
re-test the repo asked for, are in [gb10-first-run.md](gb10-first-run.md).

## The finding: `status ok` from a probe that compared nothing

    ./echolock run constants --backend host
    -> status ok
    -> declared: 4 facts, every one with exactly 1 source
    -> checks: 0

The review request asks for "a path where a wrong number still reports as ok" and for suspicion of
any oracle that passes when it shouldn't. Here it is, and it is not a wrong number — it is no
comparison at all, recorded identically to a clean bill of health. `compare()` emits an `agrees`
check only when 2+ sources answered for a fact, and on this machine one source answered. So the
probes's "closest thing to a pure oracle" ran blind and reported `ok`, on the platform the project
is built for. My first reading was that `capture.nvidia_smi()` missed a `gpu_list` wrapper; that was
wrong, and checking it is what found the real causes. Four of them, each one silent:

| # | what the code does | what the machine has | consequence |
|---|---|---|---|
| 1 | `constants.py` iterates `smi.get("gpu")` | `capture.nvidia_smi()` returns the list under **`gpus`** | the nvidia-smi source is empty on **every** platform, not just this one |
| 2 | reads `gpu["fb_memory_usage"]["total"]` | capture stores it as **`fb_memory`** | memory fact never available from that source |
| 3 | reads `max_clocks["sm"]`, `["memory"]` | driver 580 nests **`sm_clock`**/`mem_clock` with units and `N/A`: `<sm_clock>3003 MHz</sm_clock>`, `<mem_clock>N/A</mem_clock>` | clock facts never compared here; `N/A` must not become 0 |
| 4 | buckets devices by each source's own id: NVML by UUID, nvidia-smi by `pci_bus_id`, /proc by stripped bus location | one GPU has `GPU-5a82916b-…`, `0000000F:01:00.0` and NVML's `0001:01:00.0`; capture did not carry the `<uuid>` child at all | even after fix 1, two sources land in two buckets and still compare nothing |

Fix 1 alone was not enough and that is worth stating: after it the run showed 2 sources, **0 checks,
`status ok`** — the failure mode looks cured and is not. Root cause 5 is structural: because
agreement emits no check, "everything agreed" and "nothing was comparable" both produce zero checks.
So I added a check that can only be answered by the harness itself:

    source_coverage  PASS  per_device {GPU-5a…: 2}  answering {GPU-5a…: [nvidia_smi, nvml]}
    need: "at least 2 sources per device, or there is nothing to compare"

Before → after on this box: comparable facts **1 → 4** (`name`, `uuid`, `vbios`,
`max_sm_clock_mhz`), all agreeing; checks 0 → 1, and that one check now fails loudly instead of the
run quietly passing. Three platform facts made this box the right place to find it:
`/proc/driver/nvidia/gpus/*/information` exists but is blank by design (`Model: Unknown`, no
`Bus Location`, no VBIOS — the T23D3 platform driver does not populate it), `vulkaninfo` is absent
(no ICD), and `fb_memory` is `N/A` with no power limit because the memory is unified and there is
no envelope. Each of those is legitimate, and together they leave a probe with one voice. A probe
must say so rather than pass.

Changes in the accompanying commit: `gpus`, `fb_memory`, the `sm_clock`/`mem_clock` tag names with
unit stripping, `N/A` → `None` (never 0), UUID from the `<gpu>` child element in
`capture.nvidia_smi()`, UUID-preferred device bucketing in `nvidia_sources()`, the
`source_coverage` check, and `tests/test_constants.py` — hermetic, GPU-free, built from this box's
real (trimmed) `nvidia-smi -q -x` output, asserting that one GPU is one bucket, that `"3003 MHz"`
parses to 3003, that `N/A` yields `None`, and that a 1500-vs-3003 disagreement *is* a failure.
A fixture beats a live call here: it makes the next schema change fail CI instead of emptying a
source quietly. 53 tests pass, 2 skipped.

## What fact is missing from the comparison set? The versions

`driver_version` and `cuda_version` appear **nowhere** in `constants.py`, although every run
directory already carries them from more than one voice: driver `580.173.02` from nvidia-smi XML,
from NVML, and from `/proc/driver/nvidia/version` text; CUDA `13.0` from nvidia-smi. Versions are
the fact class most likely to explain a cross-machine difference in behaviour — the repo's own
driver-behaviour finding in `fb71d27` turned on a driver/GSP detail — and they are the facts the
comparison set leaves out. GSP firmware is the third candidate, and it shows exactly why coverage
must be recorded rather than assumed: nvidia-smi reports `gsp_firmware_version` 580.173.02 while
NVML's system block is null on this box, so a version comparison today has one voice on some facts
and two or three on others. Record which, or `source_coverage` is telling half the story.

## The other claims, as far as this machine can test them

- **fp16 mma bit-exact across CUDA and Vulkan** — not reproducible here: no Vulkan ICD, so
  `crosscheck mma --bitexact` has no second backend. On the unification question: unified operands
  make this a diff of two fragment layouts, which is the right test for a layout bug, but the claim
  covers *layout agreement*, not that both stacks compute fp16 mma identically for arbitrary input.
  Worth one sentence saying so, because "bit-exact" reads wider than it is.
- **fp4 has no independent reference** — confirmed, and this box's numbers are fp8 210.3 and fp4
  210.3 TFLOPS, identical to 4 digits, consistent with both being limited by the same tensor-core
  issue path rather than by width. `recorded, unverified` is the honest label and the report uses it.
- **XOR-fold independence** — holds, with the caveat in
  [gb10-first-run.md](gb10-first-run.md): XOR is commutative, so it verifies volume, not order.
  Right trade for `memory`; a hole for `pointer_chase`, where order *is* the measurement.
- **`load_check` bounds ambient load from below** — direction sound, but on a box like this the
  bound is loose, not wrong: the resident inference engine held 42-77% SM throughout my runs, so the
  sampler's view is dominated by someone else's work. Suggest recording
  `compute_apps`/co-resident processes in the manifest, so "idle" means *nothing else was running*
  rather than *nobody asked*.
- **A null result should not read as a finding.** My clock run reports `clock_drop_ratio
  0.9999999254` (idle 2144.91705 MHz, load 2144.91689 MHz): idle and load identical, so there was no
  drop to detect. The derived trace was flat the same way — 2144.9 MHz across 480 samples while
  NVML saw 2132-2145 in the same window. Suggest a `variance_observed` check, so a constant trace
  cannot read as a clean pass; it is the same rule as `source_coverage`, applied to a different probe.
- **`util_mem_pct: 0` and `mem_clock_mhz: null`** are the sibling problem: a flat 0 on a
  unified-memory part reads as "memory bus idle" on the one machine where memory bandwidth is the
  whole story. Wants an explicit `unsupported`, not a zero.

## What this box contributed, for the record

The `cudaHostRegister` question ended "not blocked on GB10": file-backed pages register and the
device reads correct data through them ([crosscheck.md](crosscheck.md)). Repro sources and output
for both that and this review are on `192.168.0.25:~/echolock-runs-gb10-20261010/`, kept out of the
repo per its layout rule.
