BUILD_FLAVOR        := release

PLATFORM            := WINDOWS
ifeq ($(shell uname -s),Linux)
    # Check for Jetson by file
    ifneq ($(wildcard /etc/nv_tegra_release),)
        PLATFORM    := JETSON
    else
        PLATFORM    := DESKTOP_LINUX
    endif
endif

ifeq ($(PLATFORM),WINDOWS)
    VULKAN_SDK      := C://VulkanSDK/1.4.313.2
    VULKAN_FLAGS    := -I$(VULKAN_SDK)/Include -L$(VULKAN_SDK)/Lib -lvulkan-1
    COMPILER        := $(VULKAN_SDK)/Bin/glslangValidator.exe
    COMPILER_FLAGS  :=

    GLFW3_PATH      := dean/3rdparty/glfw-3.3.8.bin.WIN64
    GLFW3_DLL       := $(GLFW3_PATH)/glfw3.dll
    GLFW3_FLAGS     := -I$(GLFW3_PATH)/include -L$(GLFW3_PATH) -lglfw3
else ifeq ($(PLATFORM),JETSON)
    # Jetson AGX Xavier 上全部都有了, 不用自己装
    # 头文件: grep "#define VK_HEADER_VERSION" /usr/include/vulkan/vulkan_core.h
    # loader libvulkan.so.1: ldconfig -p | grep vulkan
    # icd: cat /etc/vulkan/icd.d/nvidia_icd.json
    # vulkaninfo / vkcube
    #
    # jetson上没有validation layer, 要自己build, 用release build跳过validation layer
    #BUILD_FLAVER    := -DNDEBUG
    VULKAN_FLAGS    := -lvulkan
    COMPILER        := dean/3rdparty/glslc/glslc
    COMPILER_FLAGS  :=

    GLFW3_PATH      := dean/3rdparty/glfw-3.3.2.LINUX.aarch64
    # compile time needs glfw.so, runtime needs glfw.so.3
    GLFW3_DLL       := $(GLFW3_PATH)/libglfw.so.3
    GLFW3_FLAGS     := -I$(GLFW3_PATH)/include -L$(GLFW3_PATH) -lglfw
else
    # DESKTOP_LINUX
    # 临时用一下

    # 安装好了sdk后, 应该默认都source了.
    #VULKAN_SDK      := ~/vk/1.4.313.0
    #VULKAN_FLAGS    := -I$(VULKAN_SDK)/Include -L$(VULKAN_SDK)/Lib -lvulkan-1
    VULKAN_FLAGS    := -lvulkan

    # glslangValidator -V shader.vert -o shader.spv
    #COMPILER        := $(VULKAN_SDK)/bin/glslc
    COMPILER        := $(VULKAN_SDK)/bin/glslangValidator
    COMPILER_FLAGS  := -gVS # for ngfx profiling.
    
    # 直接安装了: sudo apt install libglfw3-dev
    GLFW3_FLAGS     := -lglfw
endif

CXX             := g++

COMMON_HEADERS  := -Idean/3rdparty
CXXFLAGS        := -std=c++17 \
                   -Wall \
                   $(VULKAN_FLAGS) \
                   $(GLFW3_FLAGS) \
                   $(COMMON_HEADERS)

# nsight graphics profiling shader时候disable validation layer
ifeq ($(BUILD_FLAVOR),release)
    CXXFLAGS    += -DNDEBUG
else
    CXXFLAGS    += -g
endif

# all
# 30开始给cs rt用
# todo, move 31.1 to 32_cs_rt_basic, move 31.2 to 33_cs_rt_bvh
# 40开始给vk rt用
#IDS := 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 31 31.1 31.2 40
# common used
#IDS := 15 27 31 31.1 31.2 40
# debug
IDS := 40

all:
	mkdir -p out
ifneq ($(PLATFORM),DESKTOP_LINUX)
	cp $(GLFW3_DLL) out/
endif
	@$(foreach id,$(IDS),echo "Building $(id)..." && $(CXX) dean/*/$(id)_*.cpp $(CXXFLAGS) -o out/$(id).exe;)

# make resources.
# used by 15
15:
	mkdir -p out/shaders
	$(COMPILER) dean/09_shader_base.vert -o out/shaders/vert.spv
	$(COMPILER) dean/09_shader_base.frag -o out/shaders/frag.spv

# used by 27
27:
	mkdir -p out/shaders out/textures
	cp images/texture.jpg out/textures/
	$(COMPILER) dean/27_shader_depth.vert -o out/shaders/vert.spv
	$(COMPILER) dean/27_shader_depth.frag -o out/shaders/frag.spv

# used by 31
31:
	mkdir -p out/shaders
	$(COMPILER) dean/31_shader_compute.comp -o out/shaders/comp.spv
	$(COMPILER) dean/31_shader_compute.vert -o out/shaders/vert.spv
	$(COMPILER) dean/31_shader_compute.frag -o out/shaders/frag.spv

# used by 31.1
31.1:
	mkdir -p out/shaders
	$(COMPILER) dean/31.1_shader_compute_rt_basic.comp -o out/shaders/comp.spv
	$(COMPILER) dean/31.1_shader_compute_rt_basic.vert -o out/shaders/vert.spv
	$(COMPILER) dean/31.1_shader_compute_rt_basic.frag -o out/shaders/frag.spv

# used by 31.2
31.2:
	mkdir -p out/shaders
	$(COMPILER) $(COMPILER_FLAGS) -V dean/31.2_shader_compute_rt_bvh_copy.comp -o out/shaders/comp.spv
	$(COMPILER) $(COMPILER_FLAGS) -V dean/31.2_shader_compute_rt_bvh.vert -o out/shaders/vert.spv
	$(COMPILER) $(COMPILER_FLAGS) -V dean/31.2_shader_compute_rt_bvh.frag -o out/shaders/frag.spv

# used by 40
40:
	mkdir -p out/shaders out/textures
	cp images/texture.jpg out/textures/
	$(COMPILER) -V dean/40_vk_rt_basic/40_shader_depth.vert -o out/shaders/vert.spv
	$(COMPILER) -V dean/40_vk_rt_basic/40_shader_depth.frag -o out/shaders/frag.spv

clean:
	@echo "===== Clean ====="
	rm -rf out

.PHONY: all clean
