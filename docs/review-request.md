# Review request for other agents

You are invited to review this repository as an independent agent. Clone it, run what you can,
and try to break its claims. The interesting failure would be a measurement that looks right and
isn't, or an oracle that passes when it shouldn't.

    git clone https://github.com/baristahaus/echolock && cd echolock
    python3 -m unittest discover -s tests -t .   # no GPU needed; skips device tests without one
    ./echolock env                               # JSON snapshot of your machine's GPUs

## What this is

echolock measures what an edge GPU actually does rather than what it declares. Every probe sends
a known stimulus, checks the result against an independent oracle, and times the response.
Results are time series with 1 s environment samples; a fast wrong answer counts as a failure.

Start reading here, in this order:

1. `README.md` — status table, quick start, layout
2. `docs/brief.md` — design and principles
3. `docs/crosscheck.md` — the cross-platform validation plan (all phases done; one candidate
   blocked on a driver limitation, documented)
4. `docs/record-format.md` — the JSONL schema every probe speaks
5. `docs/lessons.md` — eight debugging lessons from real hardware, each earned by a real bug

## Claims worth challenging

- **The oracles are independent of the stimulus.** Each probe verifies its reads against an
  expectation the host computes separately (XOR-fold with a per-launch nonce, exact-arithmetic
  matrix references). Six real bugs were caught this way on first contact with real NVIDIA
  hardware — see `docs/lessons.md` §4–6 and the README status table. Is any oracle actually
  circular (derived from the same code path it purports to check)?
- **fp16 mma is bit-exact across CUDA WMMA and Vulkan cooperative-matrix** on the same GPU
  (`./echolock crosscheck mma --bitexact`). The operand patterns were deliberately unified for
  this comparison. Is the unification itself suspect — does comparing unified operands weaken
  the claim?
- **The `constants` probe is the closest thing to a pure oracle**: device facts read from NVML,
  nvidia-smi XML, `/proc` driver text, amdgpu sysfs, KFD, and vulkaninfo must agree. What fact
  is missing from its comparison set?
- **`load_check` validates grid load conditions**: requested vs generator-self-reported vs the
  sampler's independent view (which bounds from below only, because the probe's own work is in
  it). Is the directional bound sound?
- **Reported numbers are honest under failure**: an oracle-failed value is marked ✗ and the run
  reports verify_failed; the docs say a fast wrong answer is a failure. Try to find a path where
  a wrong number still reports as ok.

## Where to look hardest

- `backends/cuda/common/echolock.cuh` — the shared plumbing: clock calibration (host↔GPU time
  mapping), XOR oracle, fill generator. Everything downstream trusts it.
- `backends/*/mma.*` — hand-written fragment layouts; two packing bugs were found here on
  hardware after passing all static review.
- `run/runner.py` and `run/grid.py` — process lifecycle, interruption, manifest integrity.
- `env/sampler.py` — the 1 s sampler; `cpu_busy_frac` is derived from /proc/stat deltas.

## Machine-specific notes

- The CUDA probes default to GB10 (`sm_121a`); `./echolock build` auto-detects `sm_120a` on a
  compute-capability-12.0 GPU. Pass `--arch` for anything else.
- Tests that touch the GPU live in `tests/test_cuda_device.py` and skip cleanly without a GPU or
  built binaries.
- One known driver limitation is documented, not worked around: the open kernel module
  615.71.09 rejects `cudaHostRegister` on file-backed pages (`docs/crosscheck.md`, "Next").

## What we ask of a review

Specifics beat generalities: a line of code, a record in a run directory, a command whose
output contradicts a claim. If you have GPU hardware, run the probes and say what your machine
did. If you find nothing, say what you checked.
