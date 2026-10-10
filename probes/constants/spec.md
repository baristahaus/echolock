# constants

Device facts required to agree across independent sources: the one probe that needs no GPU API
and no backend implementation, because the stimulus is the machine's own reporting and the
oracle is agreement between code paths that share nothing. A disagreement means something is
misreporting -- the closest thing to a pure oracle in this repo.

Host (all vendors): `backends/host/constants.py`.

## Stimulus

| Source | What it is | Why it is independent |
|---|---|---|
| NVML | the driver's management library | its own code path |
| nvidia-smi `-q -x` | the driver's CLI over a different interface | parses its own XML |
| `/proc/driver/nvidia/gpus/*/information` | kernel-module text | written by the kernel module |
| amdgpu sysfs | `amdgpu` hwmon + device files | written by the kernel driver |
| KFD nodes | the compute-kernel driver's view | a different subsystem of the same driver |
| `vulkaninfo --summary` | the vendor-agnostic Vulkan stack | user-mode driver, no shared code with the above |

Facts compared per device: name, UUID, VBIOS version, max SM clock, max memory clock, power
limit, total memory.

## Oracle

- A fact reported by two or more sources must agree: strings exactly, numerics within 2%
  (clocks may round; memory totals may differ by reserved-VRAM bookkeeping).
- A fact only one source reports is declared, never checked -- absence is not failure.
- The Vulkan stack's name and memory are matched fuzzily against the device (names differ in
  branding across stacks; memory is compared within tolerance). A mismatch is a finding: it
  means the user-mode driver is describing a different device or a different carve-out.

## Parameters

None. Every source is best-effort; a missing source becomes a note, never a failure. A machine
with no GPU sources at all is a run error, not an oracle failure.

## Metrics

None by design: this probe produces no timing measurements. Its value is the checks and the
declared table, which the report renders like any other probe's.
