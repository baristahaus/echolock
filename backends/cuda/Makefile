# CUDA backend. No -lcuda and no vendor libraries: the runtime is linked statically and driver-API
# calls go through cudaGetDriverEntryPointByVersion.
#
#   make -C backends/cuda                 # GB10 (sm_121a, family-specific target)
#   make -C backends/cuda ARCH=sm_120a    # consumer Blackwell (RTX 5090); fp4 MMA included
#   make -C backends/cuda ARCH=sm_90      # another target
#   make -C backends/cuda NVCC=/path/nvcc
#
# NB: `-arch=sm_121a` is NOT the family-specific build: nvcc turns it into ptxas -arch=compute_121
# (no 'a'), which rejects the fp4/f8f6f4 MMA instructions. The explicit gencode below keeps
# compute_121a, which is what lets the mma probe use fp4. Same for sm_120a on consumer Blackwell.

NVCC    ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
ARCH    ?= sm_121a
OUT     ?= ../../build/cuda
FLAGS   ?= -O3 -std=c++17 -lineinfo -Xcompiler -Wall,-Wextra
PROBES  := memory file_page sync clock constants pointer_chase mma
TOOLS   := gpuload
STAMP   := $(OUT)/build.json

ifeq ($(ARCH),$(filter sm_121a sm_120a,$(ARCH)))
GENCODE := -gencode arch=compute_$(patsubst sm_%,%,$(ARCH)),code=$(ARCH)
else
GENCODE := -arch=$(ARCH)
endif

all: $(addprefix $(OUT)/,$(PROBES)) $(addprefix $(OUT)/,$(TOOLS))

# What the probes were built with; env/capture.py copies it into every run's env.json. The rule
# always runs (FORCE) but only rewrites the stamp when arch/nvcc/flags actually change; a changed
# stamp deletes the old binaries, so building for one ARCH can never silently leave another
# ARCH's artifacts in place (a stale binary impersonating a new build is the trap in
# docs/lessons.md #2). When nothing changed the stamp keeps its mtime, so up-to-date binaries
# stay up to date.
$(STAMP): FORCE
	@mkdir -p $(OUT)
	@if [ -f $(STAMP) ] \
	   && [ "$$(sed -n 's/.*"arch": "\([^"]*\)".*/\1/p' $(STAMP))" = "$(ARCH)" ] \
	   && [ "$$(sed -n 's/.*"nvcc": "\([^"]*\)".*/\1/p' $(STAMP))" = "$(NVCC)" ] \
	   && [ "$$(sed -n 's/.*"flags": "\([^"]*\)".*/\1/p' $(STAMP))" = "$(GENCODE) $(FLAGS)" ]; then \
	    :; \
	  else rm -f $(addprefix $(OUT)/,$(PROBES) $(TOOLS)); \
	   printf '{"arch": "%s", "nvcc": "%s", "flags": "%s", "built_utc": "%s"}\n' \
	    "$(ARCH)" "$(NVCC)" "$(GENCODE) $(FLAGS)" "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" > $(STAMP); \
	  fi

$(OUT)/%: %.cu common/echolock.cuh $(STAMP)
	@mkdir -p $(OUT)
	$(NVCC) $(FLAGS) $(GENCODE) $(EXTRA) $< -o $@

clean:
	rm -f $(addprefix $(OUT)/,$(PROBES)) $(addprefix $(OUT)/,$(TOOLS))

# Compatibility name.
stamp: $(STAMP)

FORCE:

.PHONY: all clean stamp FORCE
