# Vulkan compute backend (milestone 5)

This backend will implement the same probe specs (`probes/*/spec.md`) as compute shaders, first
memory and the latency map. It will give one design across NVIDIA, AMD, Intel, Mali and Adreno.
On NVIDIA the proprietary Vulkan driver is a second closed view of the hardware; where it
disagrees with CUDA, that points at the driver. NVK (Mesa) would be an independent open path if
it supports GB10; that has to be checked before relying on it.
