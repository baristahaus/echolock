# sync

How late the host learns that GPU work has finished, per CUDA device schedule, and what a graph
launch costs against its node count.

CUDA: `backends/cuda/sync.cu`. Origin: ninfer-gb10 `tools/gb10/sync_probe.cu`.

## Stimulus

1. **Completion lag.** A one-thread kernel spins on `%globaltimer` for 50 µs, 1 ms, 10 ms and
   60 ms, so the host thread sleeps about as long as it would in a real round. Then
   `cudaStreamSynchronize` runs. Lag is the host time when the synchronize returns minus the
   kernel's end time, mapped to the host clock.
2. **Graph launch.** Chains of 100, 1,000 and 4,000 one-block kernels are captured into a graph.
   Two times are recorded: how long the `cudaGraphLaunch` call takes, and the time from the
   call's start to the first kernel's start.

The schedule (blocking, yield, spin, auto) is fixed per process, so the plan runs one process per
schedule. `--pin-cpus` adds blocking runs pinned to given CPUs, to compare core types.

## Clock mapping (principle 3)

`%globaltimer` is mapped to the host's CLOCK_MONOTONIC without calling a synchronize. A kernel
writes the GPU time and a sequence number to mapped host memory; the host spins until the
sequence number appears. Of 200 such round trips, the one with the smallest round trip gives the
offset, which is uncertain by ± half that round trip.

## Oracle

- The spin kernel records its own start and end, and each iteration checks that it spun at least
  the requested time.
- A synchronize cannot return before the kernel ended. A lag more negative than half the
  calibration round trip fails the `clock_mapping` check.
- Every graph node increments a counter (one atomic from one thread). A launch whose count isn't
  the node count fails `graph_nodes_executed`. The nodes aren't empty kernels, as they were in
  the original.

## Metrics

`sync_return_lag` µs per iteration, tagged schedule and kernel_us. `graph_launch_call` and
`graph_launch_to_first_kernel` µs per iteration, tagged nodes. Each has a p50 summary.

## Silent fallbacks detected

The requested schedule is recorded as `cuda.device_flags` from `cudaGetDeviceFlags`. Its effect is
the lag distribution: if blocking and spin produce the same lag, the schedule was not applied.

## GB10 results so far

Blocking sync returns 0.4–1.2 ms late, worse after longer waits. Yield and spin return in about
3 µs. Graph launch cost doesn't scale with node count: 4,000 nodes launch in 14 µs.
