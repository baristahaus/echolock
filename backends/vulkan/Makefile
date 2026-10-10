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
PROBES  := memory pointer_chase mma sync clock file_page

SHADERS := read_fold chase spin_until clock_dual row_read
# The cooperative-matrix shader is compiled once per math format (-D selects the operand types).
COOP_FORMATS := fp16 bf16 int8

all: $(addprefix $(OUT)/,$(PROBES)) $(addprefix $(OUT)/coopmul_,$(addsuffix .spv,$(COOP_FORMATS)))

$(OUT)/%.spv: shaders/%.comp
	@mkdir -p $(OUT)
	$(GLSLC) -O $< -o $@

$(OUT)/memory: memory.cpp common/echolock_vk.h $(OUT)/read_fold.spv
	@mkdir -p $(OUT)
	$(CXX) $(FLAGS) -I include -I . memory.cpp -lvulkan -o $@

$(OUT)/pointer_chase: pointer_chase.cpp common/echolock_vk.h $(OUT)/chase.spv
	@mkdir -p $(OUT)
	$(CXX) $(FLAGS) -I include -I . pointer_chase.cpp -lvulkan -o $@

$(OUT)/coopmul_%.spv: shaders/coopmul.comp common/echolock_vk.h
	@mkdir -p $(OUT)
	$(GLSLC) --target-env=vulkan1.4 -O -DFORMAT_$(subst fp16,FP16,$(subst bf16,BF16,$(subst int8,INT8,$*))) $< -o $@

$(OUT)/mma: mma.cpp common/echolock_vk.h common/vk_ctx.h $(OUT)/coopmul_fp16.spv \
            $(OUT)/coopmul_bf16.spv $(OUT)/coopmul_int8.spv
	@mkdir -p $(OUT)
	$(CXX) $(FLAGS) -I include -I . mma.cpp -lvulkan -o $@

$(OUT)/sync: sync.cpp common/echolock_vk.h common/vk_ctx.h $(OUT)/spin_until.spv
	@mkdir -p $(OUT)
	$(CXX) $(FLAGS) -I include -I . sync.cpp -lvulkan -o $@

$(OUT)/clock: clock.cpp common/echolock_vk.h common/vk_ctx.h $(OUT)/clock_dual.spv

	@mkdir -p $(OUT)
	$(CXX) $(FLAGS) -I include -I . clock.cpp -lvulkan -o $@

$(OUT)/file_page: file_page.cpp common/echolock_vk.h common/vk_ctx.h $(OUT)/row_read.spv
	@mkdir -p $(OUT)
	$(CXX) $(FLAGS) -I include -I . file_page.cpp -lvulkan -o $@

clean:
	rm -f $(addprefix $(OUT)/,$(PROBES)) $(addsuffix .spv,$(addprefix $(OUT)/,$(SHADERS))) \
	      $(addprefix $(OUT)/coopmul_,$(addsuffix .spv,$(COOP_FORMATS)))

.PHONY: all clean
