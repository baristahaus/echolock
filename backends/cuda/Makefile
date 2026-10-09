# CUDA backend. No -lcuda and no vendor libraries: the runtime is linked statically and driver-API
# calls go through cudaGetDriverEntryPointByVersion.
#
#   make -C backends/cuda                 # GB10 (sm_121a)
#   make -C backends/cuda ARCH=sm_90      # another target
#   make -C backends/cuda NVCC=/path/nvcc

NVCC    ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
ARCH    ?= sm_121a
OUT     ?= ../../build/cuda
FLAGS   ?= -O3 -std=c++17 -lineinfo -Xcompiler -Wall,-Wextra
PROBES  := memory file_page sync
STAMP   := $(OUT)/build.json

all: $(addprefix $(OUT)/,$(PROBES)) stamp

# What the probes were built with; env/capture.py copies it into every run's env.json. PHONY so it
# is refreshed with the current ARCH/NVCC even when the binaries themselves are up to date.
stamp:
	@mkdir -p $(OUT)
	@printf '{"arch": "%s", "nvcc": "%s", "flags": "%s", "built_utc": "%s"}\n' \
	    "$(ARCH)" "$(NVCC)" "$(FLAGS)" "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > $(STAMP)

$(OUT)/%: %.cu common/echolock.cuh
	@mkdir -p $(OUT)
	$(NVCC) $(FLAGS) -arch=$(ARCH) $(EXTRA) $< -o $@

clean:
	rm -f $(addprefix $(OUT)/,$(PROBES))

.PHONY: all clean
