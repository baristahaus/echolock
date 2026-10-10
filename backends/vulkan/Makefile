# Vulkan backend. One probe design across NVIDIA, AMD, Intel, Mali and Adreno; on this
# machine it runs on AMD through RADV (Mesa), an open-source driver -- the whole path from the
# probe to the hardware is open, unlike the NVIDIA proprietary Vulkan driver.
#
#   make -C backends/vulkan            # glslc (shader) + g++ (probe), -lvulkan (the loader)
#
# The Vulkan headers are vendored (include/vulkan/, pinned to vulkan-sdk-1.4.363) so the build
# needs no Vulkan SDK install; only the system loader library (libvulkan.so) is required.

CXX     ?= g++
GLSLC   ?= glslc
OUT     ?= ../../build/vulkan
FLAGS   ?= -O2 -std=c++17 -Wall -Wextra
PROBES  := memory

SHADERS := read_fold

all: $(addprefix $(OUT)/,$(PROBES))

$(OUT)/%.spv: shaders/%.comp
	@mkdir -p $(OUT)
	$(GLSLC) -O $< -o $@

$(OUT)/memory: memory.cpp common/echolock_vk.h $(OUT)/read_fold.spv
	@mkdir -p $(OUT)
	$(CXX) $(FLAGS) -I include -I . memory.cpp -lvulkan -o $@

clean:
	rm -f $(addprefix $(OUT)/,$(PROBES)) $(addsuffix .spv,$(addprefix $(OUT)/,$(SHADERS)))

.PHONY: all clean
