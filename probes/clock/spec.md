# clock

The SM clock the GPU actually runs at, measured from counters the kernel itself reads -- not the
number the driver reports. On an edge box the declared clock is a maximum; the sustained clock
under load is what every other measurement was actually taken at.

CUDA: `backends/cuda/clock.cu`.

## Stimulus

| Part | What runs |
|---|---|
| Ratio | One block per SM reads `%globaltimer` (GPU-wide nanoseconds) and `clock64()` (SM clock ticks) around a spin, and derives the SM clock as the ratio of the two deltas. |
| Idle | The same measurement with nothing else running: the baseline clock. |
| Under load | The ratio kernel runs while a streaming-read kernel keeps the memory system busy: the clock under real work, where power and thermal caps bite. |
| Trace | A series of ratio samples a few hundred ms apart while the load runs, so a throttle event shows as a step in the series rather than an average. |
| Calibration | A kernel that spins for a fixed number of `%globaltimer` nanoseconds: its wall time, measured on the host clock, must match -- the same host/GPU clock mapping the sync probe depends on. |

## Oracle

- The spin kernel's wall time matches its requested GPU time within tolerance, so `%globaltimer`
  is trusted before it is used to derive anything.
- The per-SM derived clocks agree within a few percent: SMs run in lockstep, and one SM
  disagreeing means the measurement is wrong, not the GPU.
- The derived clock is within a sane band of the API's declared `clock_khz` (a ratio outside
  0.1x-3x means one of the two is not measuring what it claims).

## Parameters

`--seconds` per series (2), `--load-gib` load buffer size (1), `--trace-ms` between trace samples
(200).

## Metrics

- `derived_sm_clock_mhz`, tagged `condition` (idle, load), one record per SM block plus a summary
  with per-SM min/p50/max.
- `clock_trace` MHz per trace sample while loaded, with rep indices so drops are visible.
- `declared_over_derived_ratio`, measured against `cudaDeviceGetAttribute(ClockRate)`.

## Why this probe exists

Every bandwidth and latency number the tool records is divided by seconds measured somewhere. If
the clock is not what the driver says, the GB/s and the latency are still real, but attributing
them to "the GPU at 1.0 GHz" is wrong. The sampler records the driver's clock every second; this
probe measures it.
