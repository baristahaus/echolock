# CUDA backend. No -lcuda and no vendor libraries: the runtime is linked statically and driver-API
# calls go through cudaGetDriverEntryPointByVersion.
#
#   make -C backends/cuda                 # GB10 (sm_121a, family-specific target)
#   make -C backends/cuda ARCH=sm_90      # another target
#   make -C backends/cuda NVCC=/path/nvcc
#
# NB: `-arch=sm_121a` is NOT the family-specific build: nvcc turns it into ptxas -arch=compute_121
# (no 'a'), which rejects the fp4/f8f6f4 MMA instructions. The explicit gencode below keeps
# compute_121a, which is what lets the mma probe use fp4.

NVCC    ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
ARCH    ?= sm_121a
OUT     ?= ../../build/cuda
FLAGS   ?= -O3 -std=c++17 -lineinfo -Xcompiler -Wall,-Wextra
PROBES  := memory file_page sync clock pointer_chase mma
TOOLS   := gpuload
STAMP   := $(OUT)/build.json

ifeq ($(ARCH),sm_121a)
GENCODE := -gencode arch=compute_121a,code=sm_121a
else
GENCODE := -arch=$(ARCH)
endif

all: $(addprefix $(OUT)/,$(PROBES)) $(addprefix $(OUT)/,$(TOOLS)) stamp

# What the probes were built with; env/capture.py copies it into every run's env.json. PHONY so it
# is refreshed with the current ARCH/NVCC even when the binaries themselves are up to date.
stamp:
	@mkdir -p $(OUT)
	@printf '{"arch": "%s", "nvcc": "%s", "flags": "%s", "built_utc": "%s"}\n' \
	    "$(ARCH)" "$(NVCC)" "$(GENCODE) $(FLAGS)" "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > $(STAMP)

$(OUT)/%: %.cu common/echolock.cuh
	@mkdir -p $(OUT)
	$(NVCC) $(FLAGS) $(GENCODE) $(EXTRA) $< -o $@

clean:
	rm -f $(addprefix $(OUT)/,$(PROBES)) $(addprefix $(OUT)/,$(TOOLS))

.PHONY: all clean stamp
