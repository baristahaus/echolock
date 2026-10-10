# mma

What each tensor-core math format actually delivers, measured with the hardware's own MMA
instructions and verified numerically -- not the peak number in a datasheet.

CUDA: `backends/cuda/mma.cu`.

## Stimulus

| Part | What runs |
|---|---|
| MMA loop | A warp issues `mma.sync` fragments back-to-back: each result accumulates into the next, so the pipeline cannot be emptied by the compiler, and the final accumulator is written out. |
| Formats | fp16, bf16, tf32 (PTX `mma.sync` shapes native to the architecture); fp8 e4m3 and int8 where the architecture supports them; nvfp4/mxfp4 on parts that have fourth-generation tensor cores. |
| Shapes | The smallest native MMA fragment per format (m16n8k16 fp16/bf16, m16n8k8 tf32, m16n8k32 fp8/int8, and the architecture's fp4 shape), repeated many times per thread block to saturate the tensor cores. |

## Oracle

Every format's MMA result is checked numerically. Operands use exactly representable values
(small integers in fp16/bf16/tf32; powers of two in fp8; the fp4 quantization grid), so the host
computes the exact expected accumulator without floating-point ambiguity. A fragment whose result
is wrong fails the check rather than posting a throughput -- this is what catches a mis-encoded
operand layout, which is the classic failure mode of hand-written MMA code and would otherwise
look like a working probe with plausible numbers.

## Parameters

`--seconds` per series (0.5), `--formats` subset (fp16,bf16,tf32,fp8,int8,fp4), `--blocks` grid
size (SM count x 8).

## Metrics

- `mma_throughput` in TFLOPS (TOPS for int8), tagged `format` and `shape`. One record per rep
  plus a summary.
- `format_ratio` fp16-relative throughput per format, so fp8-at-half-of-fp16 reads as 2.0 and a
  format the compiler emulated reads as a small number.

## What this probe is for

An edge inference box is bought for its tensor throughput in one or two formats (on GB10-class
parts: fp8 and fp4). The declared peak is a roofline under conditions nobody's workload meets.
The measured number, with a verified result and the SM clock it ran at (the `clock` probe
supplies it), is what a model's latency budget can actually be built from.
