####### Compiler, tools and options

OS = Linux
#OS = Windows

ifeq ($(OS),Windows)
    VULKAN_SDK          =C://VulkanSDK/1.3.239.0
    GLSLC               =$(VULKAN_SDK)/Bin/glslc.exe
    VULKAN_INCLUDE_DIR  =$(VULKAN_SDK)/Include
    VULKAN_LIB_DIR      =$(VULKAN_SDK)/Lib
    VULKAN_LIB          =vulkan-1
    GLFW_LIB            =glfw3
else
    GLSLC               =$(VULKAN_SDK)/bin/glslc
    VULKAN_INCLUDE_DIR  =$(VULKAN_SDK)/include
    VULKAN_LIB_DIR      =$(VULKAN_SDK)/lib
    VULKAN_LIB          =vulkan
    # sudo apt install libglfw3 libglfw3-dev
    GLFW_LIB            =glfw
endif

# General options for all programs.
HEADER_DIR    = -I$(VULKAN_INCLUDE_DIR) \
                -Idean_dlls/glfw-3.3.8.bin.WIN64/include
LIB_DIR       = -Ldean_dlls/glfw-3.3.8.bin.WIN64/lib-mingw-w64 \
                -L$(VULKAN_LIB_DIR)
LIBS          = -l$(GLFW_LIB) -l$(VULKAN_LIB)

#TODO
# the stb image file is in dean_dlls/glfw-3.3.8.bin.WIN64/include for 24 program, fix me.


####### Build rules

CXX      	  = g++
CXXFLAGS 	  = -Wall \
                -Werror \
                -Wno-unused-variable \
                -Wno-unused-but-set-variable \
                -g \
                -std=c++17 \
                $(WINDOWS_FLAGS) \
                $(HEADER_DIR) \
                $(LIB_DIR) \
                $(LIBS)

.PHONY : all

# for debug
PROGRAMS = 15.12

all : Makefile
	mkdir -p out/shaders out/textures
	cp dean_dlls/glfw-3.3.8.bin.WIN64/lib-mingw-w64/glfw3.dll out/
	cp images/texture.jpg out/textures/
    # used by 09 ~ 17
    #$(GLSLC) code/09_shader_base.vert -o out/shaders/vert.spv
    #$(GLSLC) code/09_shader_base.frag -o out/shaders/frag.spv
    # 15.1
	$(GLSLC) code/15.1_shader_base.vert -o out/shaders/vert.spv
	$(GLSLC) code/15.1_shader_base.frag -o out/shaders/frag.spv
    # 15.12
	$(GLSLC) code/15.12_shader_write.vert -o out/shaders/15.12_shader_write_vert.spv
	$(GLSLC) code/15.12_shader_write.frag -o out/shaders/15.12_shader_write_frag.spv
	$(GLSLC) code/15.12_shader_read.vert -o out/shaders/15.12_shader_read_vert.spv
	$(GLSLC) code/15.12_shader_read.frag -o out/shaders/15.12_shader_read_frag.spv
    # used by 18 ~ 21
    #$(GLSLC) code/18_shader_vertexbuffer.vert -o out/shaders/vert.spv
    #$(GLSLC) code/18_shader_vertexbuffer.frag -o out/shaders/frag.spv
    # used by 22 ~ 25
    #$(GLSLC) code/22_shader_ubo.vert -o out/shaders/vert.spv
    #$(GLSLC) code/22_shader_ubo.frag -o out/shaders/frag.spv
    # used by 26 ~
    #$(GLSLC) code/26_shader_textures.vert -o out/shaders/vert.spv
    #$(GLSLC) code/26_shader_textures.frag -o out/shaders/frag.spv
	$(foreach id,$(PROGRAMS),$(CXX) code/$(id)_*.cpp $(CXXFLAGS) -o out/$(id).exe;)

clean :
	@echo "===== Clean ====="
	rm -rf out