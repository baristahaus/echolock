# mma

What each tensor-core math format actually delivers, measured with the hardware's own MMA
instructions and verified numerically -- not the peak number in a datasheet.

CUDA: `backends/cuda/mma.cu` (WMMA 16x16x16 fp16/int8, 16x16x8 tf32; hand-written PTX
m16n8k16 bf16 and m16n8k32 fp8 e4m3/fp4 e2m1).
Vulkan: `backends/vulkan/mma.cpp` (VK_KHR_cooperative_matrix 16x16x16, subgroup scope: fp16,
bf16, int8; the driver's supported configuration list is declared first).

## Stimulus

| Part | What runs |
|---|---|
| Verify | One block/warp computes a single MMA of the operand pattern and stores the full tile: the layout check, run before anything is timed. |
| MMA loop | Many independent accumulators per warp (CUDA) or per workgroup (Vulkan) issue MMAs back-to-back: the pipeline cannot be emptied by the compiler, and every accumulator is read back. |
| Slots | Each result slot carries a per-dispatch nonce, so an unwritten slot cannot match. |

## Oracle

Every format's MMA result is checked numerically, in exact arithmetic. Operands use the exact
shared pattern: `A[m][k] = (13m+7k) mod 5 - 2`, `B[k][n] = (11k+3n) mod 5 - 2` -- small
integers [-2, 2], the same values in both backends, exactly representable in every format, a
documented constant (no time-seeding), so the host's expected tile is identical on every
backend, every run. CUDA's tf32/fp8/fp4 keep a fractional 0.5-step grid (the fp4 e2m1 grid
needs the 0.5 steps) and are excluded from the checksum. A fragment whose result is wrong
fails the check rather than posting a throughput -- this is what catches a mis-encoded
operand layout, the classic failure mode of hand-written MMA code.

## ref_xor

One additional verified record per format: a `<format>.ref_xor` check whose detail is the
string `ref_xor=<hex64>` -- the 64-bit XOR, over row-major (i, j) with i < 16, j < 8 (the
left half of a 16-wide tile, the whole 8-wide tile), of the expected tile's values as int64
bit patterns, emitted after the format's matrix check and passing only when it passes. A
pure function of the operand pattern: identical across backends and stable across runs on
the same hardware, so "fp16/bf16/int8 are bit-exact across backends" is a comparison of
these detail strings (the crosscheck does exactly that).

## Parameters

CUDA: `--seconds` per series (0.5), `--formats` subset (fp16,bf16,tf32,int8,fp8,fp4),
`--blocks-per-sm` (8).
Vulkan: `--seconds` per series (0.5), `--formats` subset (fp16,bf16,int8), `--workgroups` (512).

## Metrics

- Throughput in TFLOPS (TOPS for int8): `mma_throughput` (CUDA, tagged `format`/`shape`) and
  `<format>_tflops` (Vulkan), one record per rep plus a summary, only verified reps count.
- `format_ratio` (CUDA): fp16-relative throughput per format.
- `<format>.ref_xor` check records (fp16/bf16/int8 only), the cross-backend checksum.
