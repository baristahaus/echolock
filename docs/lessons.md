# Lessons

Mistakes and misdiagnoses this project actually made, recorded so they are not repeated. Each
entry names the trap, how it was found, and the rule to carry forward. Read this before
debugging a "driver bug".

## 1. Suspect your own translation before the driver (2026-10, mma/Vulkan)

**What happened.** The Vulkan mma probe's tensor-multiply verified false on every format and
shape. The disassembly showed the compiled kernel doing nothing but storing zeros, and the
working diagnosis — written into the README and the probe's own notes — was "RADV lowers
OpCooperativeMatrixMulAddKHR wrong on RDNA4". Hours went into ACO opcode tables and Mesa
source before the real cause surfaced: `coopMatMulAdd(x, y, z)` in GL_KHR_cooperative_matrix
**returns** the new accumulator and does not modify `z`, unlike CUDA's `mma_sync`. The shader
discarded the return value; the compiler correctly dead-code-eliminated the multiply and
stored the untouched zero initializer. Two further probe bugs hid behind it (a transposed
column-major B fill, missing signedness operand bits in SPIR-V), each producing a different
wrong number that looked like compiler brokenness. The driver was never wrong.

**Why it fooled us.** The failure was real, the disassembly looked damning (loads eliminated,
zero-stores), and the vendor line "RADV is young on RDNA4" made a driver bug plausible. Only
the fact that rocBLAS computed correctly on the same silicon kept the "silicon" hypothesis
out — but nothing kept the "my code" hypothesis in.

**Rules.**
- When a ported probe verifies false, first diff the *semantics* of the source API against
  the target API (return value vs in-place, argument order, units, inclusive/exclusive), and
  read the target's spec section, not just the signature. The disassembly tells you what the
  compiler did with your code, not whether your code asked for what you think it asked for.
- A driver-bug claim requires either a second independent consumer failing on the same
  hardware path (rocBLAS worked ⇒ silicon fine), or a hand-verified ISA/opcode path. Neither
  alone is proof; "it looks wrong in the disassembly" is neither.
- Never write a driver-bug conclusion into persistent documentation (README, probe headers)
  from a debugging session alone. This project had to retract one. Record the *observation*
  (verified=false, evidence) persistently; record the *attribution* only when proven.
- The oracle is what caught all three bugs, including the two hiding behind the first. Keep
  verifying every result even while a known failure is being explained — the second bug's
  signature (got=4 where a wrong-but-different value was expected) is only visible if you
  still look.

## 2. A stale build artifact can impersonate a driver fix (2026-10, mma/Vulkan)

Changing a Makefile's compile flags does not rebuild targets whose *prerequisites* are
unchanged — the int8 shader kept its old, unsigned-operand SPIR-V through one "fix verified in
isolation but not in the probe" cycle. When a fix lives in the build recipe, `rm` the outputs
or touch the sources before concluding anything.

## 3. Environment differences masquerade as code differences (2026-10, mma/Vulkan)

The same SPIR-V gave 19 in a test harness and 4 in the probe. The harness differed from the
probe in buffer placement (host vs BAR) and flags, and the real difference was one fill loop.
When two runs of the same binary disagree, enumerate *every* difference — allocation type,
binding layout, push contents, grid size — before theorizing.

---

Add new lessons at the bottom with a date and a one-line title. A lesson earns its place by
having cost real debugging time here.

## 4. Feed fp16 bits into a bf16 MMA and the oracle teaches you bit layouts (2026-10, mma/CUDA)

First real-GPU run of the bf16 PTX path failed the element-wise oracle. Debugging went fastest
by re-deriving the fragment layout from the hardware itself: WMMA's `load_matrix_sync` into a
raw-register dump gives ground truth for A, and an identity-A times unique-valued-B run reads
the B mapping straight out of D. One detour worth recording: fp16 bits reinterpreted as bf16
produce astronomically wrong values that look like hardware breakage but are a test bug.

## 5. Verify the whole pipeline, not just the fill (2026-10, mma/CUDA)

The bf16 and fp8 failures had two independent causes: `store_d` had the D fragment's d1/d2
registers swapped (masked so far because WMMA handles its own store on the fp16 path), and
`pack4` added `row_bytes` to a base that already included the row offset — every A and B
fragment read one row too far. A fill verified in isolation passes while the probe fails; test
fill, instruction, and store together against an exact reference, and bisect by replacing one
stage at a time with known-good indexing.

## 6. A probe can invalidate its own stimulus (2026-10, memory/CUDA)

The memory probe's copy phase used the device buffer as copy destination, destroying the random
fill the later contention phase expected; the XOR oracle flagged it with the host pattern's
fingerprint, which pinpointed the overwrite immediately. Give each stimulus buffer one owner,
or refill before reuse — and be suspicious when a failure's value matches another buffer's
expected XOR exactly.

## 7. Read the packing section of the PTX spec before guessing nibble layouts (2026-10, mma/CUDA fp4)

The fp4 path failed on hardware twice before the spec was read end to end. The mma instruction
pages say each .b32 register holds four e2m1 elements — but the packing note (under the
kind::f8f6f4 description) says a 4-bit element lives in the *central* four bits of its 8-bit
container, two padding bits below and above. Two independent layout guesses (2 values per byte;
low nibble) both produced negative sums that looked like fragment mis-mapping. Also: `-arch=sm_120a`
silently drops the family suffix just like sm_121a does — always use explicit
`-gencode arch=compute_120a,code=sm_120a`.

## 8. Control-test the rejection before designing around it (2026-10, file-backed GPU reads)

The file-backed GPU-read probe died at `cudaHostRegister` -> cudaErrorInvalidArgument on
file-backed mmap pages. One control test (anonymous malloc, anonymous mmap) showed registration
itself works and the rejection is specific to file-backed pages -- a driver limitation, not a
probe bug, and not something to design around until it is re-tested on other drivers. Also: a
segfaulting feasibility test had an unbuffered-diagnosis detour worth keeping -- printf to a
crashed process loses buffered output, so feasibility scaffolds should `vdprintf(2, ...)`.
