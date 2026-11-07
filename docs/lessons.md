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

## 7. A build stamp that is not a prerequisite can lie about the build (2026-11, CUDA Makefile)

**What happened.** `make -C backends/cuda ARCH=sm_120a` after an sm_121a build did nothing: the
binary rules depended only on `%.cu` and the header, so ARCH was invisible to make, while the
PHONY stamp target rewrote `build.json` to *claim* sm_120a. The next run's `env.json` would have
recorded the wrong architecture for the binaries that actually executed. A first fix made the
binaries depend on the stamp *file* while the rule still targeted the literal name `stamp` —
make then never re-ran the check at all ("Nothing to be done"), which took a controlled
121a→120a→90→121a sequence (counting actual nvcc lines) to expose.

**Why it fooled us.** Each `make` exited 0, and the stamp — the one artifact that says what the
binaries were built for — agreed with the requested ARCH while lying about the binaries.

**Rules.**
- The stamp must be the real target the binaries depend on, re-evaluated on every invocation
  (`FORCE`), and rewritten only when its *content* changes; a changed stamp deletes the stale
  binaries. An ARCH or flag switch that does not produce compile lines is a stale build, not a
  no-op.
- Verify build-system claims with a sequence, not a single build: switch, expect N compiles;
  repeat, expect 0; switch back, expect N.

## 8. Read the packing section of the PTX spec before guessing nibble layouts (2026-10, mma/CUDA fp4)

The fp4 path failed on hardware twice before the spec was read end to end. The mma instruction
pages say each .b32 register holds four e2m1 elements — but the packing note (under the
kind::f8f6f4 description) says a 4-bit element lives in the *central* four bits of its 8-bit
container, two padding bits below and above. Two independent layout guesses (2 values per byte;
low nibble) both produced negative sums that looked like fragment mis-mapping. Also: `-arch=sm_120a`
silently drops the family suffix just like sm_121a does — always use explicit
`-gencode arch=compute_120a,code=sm_120a`.

## 9. Control-test the rejection before designing around it (2026-10, file-backed GPU reads)

The file-backed GPU-read probe died at `cudaHostRegister` -> cudaErrorInvalidArgument on
file-backed mmap pages. One control test (anonymous malloc, anonymous mmap) showed registration
itself works and the rejection is specific to file-backed pages -- a driver limitation, not a
probe bug, and not something to design around until it is re-tested on other drivers. Also: a
segfaulting feasibility test had an unbuffered-diagnosis detour worth keeping -- printf to a
crashed process loses buffered output, so feasibility scaffolds should `vdprintf(2, ...)`.

**The re-test happened, and the limitation moved.** On GB10 with open kernel module 580.173.02 the
same call accepts file-backed pages in every variant tried (MAP_SHARED and MAP_PRIVATE, read-only
and read-write, flags 0 and `Mapped`), and the device reads correct data through the registered
pointer -- a checksum over 8 MiB matched the host-written pattern. "The driver rejects it" was
true of one driver/platform pair and is not a property of CUDA, and the workaround that would have
changed what is measured (a staging copy) turned out not to be needed on the project's primary
target. Two things this kind of re-test must not overclaim: register returning `cudaSuccess` is not
evidence the mapping works, so checksum from the device; and `mincore` reports page-cache residency,
not a mapping's fault state, so an "unfaulted" file mapping that measures 16384/16384 resident was
never unfaulted -- the fault-state question is still open (docs/crosscheck.md). The instruction in
the original note -- re-test before designing around it -- is what kept a wrong platform-wide
conclusion from becoming a design; keep it as procedure.

## 10. A library rewrite breaks consumers that were never re-run (2026-11, host constants / env.amdgpu)

**What happened.** The host constants probe (written October) called the `env/amdgpu` API the way
it existed before the November S3 rewrite. After the rewrite, `dpm_table()` returns
`{"states": {level: MHz}, "active": ..., "forced_level": ...}` where it used to return
`{level: MHz}`, and `_kfd_nodes()` yields node id + `mem_banks` where it used to yield
`node_id`/`name`/`mem`. The probe did not fail loudly: the exception was swallowed by the
source-collection try/except and resurfaced as "no GPU sources found (no NVIDIA, no amdgpu)"
on a machine with two AMD GPUs. The merge that brought the probe in passed 111 tests — every
test that existed still passed — and the defect only surfaced by *running the probe on real
hardware* after the merge.

**Why it fooled us.** The API change and the probe came from different months and different
machines; nothing at merge time knew the probe's expectations were stale. Unit tests built
against the old shape were not written (the probe had no AMD fixture test, unlike the NVIDIA
side, which had been forced to grow one by a GB10 review), and the failure mode — a swallowed
exception becoming a *plausible* "no GPU here" note — reads like an environment fact, not a
crash.

**Rules.**
- When a shared library's data shapes change (not just signatures), re-run every consumer on
  real hardware the same day. A green test suite is not evidence the consumers are intact:
  the suite only covers the code its tests touch.
- A source collector that fails must not report "source absent" — collection failure and
  source absence are different facts. The probe now records the exception as a `note` with
  the type and message before concluding anything.
- Fixture tests are what make API drift fail *in CI instead of on hardware*: the NVIDIA side
  of this probe had one (built from real `nvidia-smi` output) and its schema mismatch was
  caught by design; the AMD side did not, and its mismatch cost a real run to find. New
  parsing code gets a fixture test from real output, not a synthetic one.
- The fix itself had to be written against the *current* library (the tested one), never the
  other way round: adapt the probe to `env/amdgpu`, don't regress `env/amdgpu` to what the
  probe expects.

## 11. The probe CLI is a per-backend contract, and it drifts (2026-11, split crosscheck)

**What happened.** The split crosscheck ran `memory` on both machines "with the same
arguments" — `--gib 0.5` on CUDA, which the Vulkan build rejected with `unknown argument
--gib` (it takes `--mib`, default 256). The Vulkan run exited 2 and wrote a run directory
whose manifest said `error` while its `probe.jsonl` held two records and no error record to
explain why; the explanation was in `stderr.log` (`[memory] usage: memory [--mib N (256)]
...`), a place the record-stream contract does not require the reader to look.

**Why it fooled us.** The two backends were written in different months with different
defaults (CUDA: GiB granularity; Vulkan: MiB), and nothing enforces a shared argument
surface — `crosscheck`'s `--a-args`/`--b-args` exists precisely because per-backend
arguments are expected to differ. Assuming one CLI for "the same probe" is assuming the
contract instead of reading it.

**Rules.**
- Before a cross-backend (or cross-machine) batch run, read both `--help`/usage lines and
  translate the arguments; "same probe" guarantees the same *stimulus contract* (the
  spec.md), not the same flags.
- A run directory with status `error` and no `error` record is not self-explanatory: the
  contract says the explanation may be in `stderr.log` alone. When a batch produces such a
  directory, read `stderr.log` before re-running — the usage line was the whole diagnosis.
- Broken run directories are not just clutter: `test_every_real_run_dir_validates` lints
  every directory under `runs/`, so a half-written or argument-error run fails the suite
  until it is removed or fixed.
