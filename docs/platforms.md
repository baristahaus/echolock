# echolock platforms

Generated from run directories (`./echolock platforms`); the newest run of each
(platform, probe) pair wins each cell. Regenerated, never hand-written.

| Platform | clock clock drop | clock declared/derived | constants source agreement | file_page status | memory copy h2d | memory device read | memory status | mma bf16 | mma fp16 | mma fp4 | mma fp8 | mma int8 | mma status | mma tf32 | pointer_chase L1 hop seq | pointer_chase random worst | sync return lag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA GeForce RTX 5090 [cuda] | 1.00 x | 0.87 x | — | ok | — | — | error | 256.6 TFLOPS | 256.7 TFLOPS | 513.4 TFLOPS | 513.4 TFLOPS | 513.0 TFLOPS | — | 64.2 TFLOPS | 15.9 ns/hop | 65.9 ns/hop | — |
| NVIDIA GeForce RTX 5090 [host] | — | — | 1/1 checks | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| NVIDIA GeForce RTX 5090 [vulkan] | — | — | — | — | 19.6 GB/s | 719.0 GB/s | — | — | — | — | — | — | ok | — | 19.3 ns/hop | — | — |
