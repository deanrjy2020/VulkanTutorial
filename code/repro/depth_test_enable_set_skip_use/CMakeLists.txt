# 只给Android(NDK toolchain)用, 和gtb的--android-executable一样编一个adb shell binary.
# 保留源码中的 Windows/Linux 分支, 本目录只配置独立 Android build.
# 用法见同目录readme.txt.
cmake_minimum_required (VERSION 3.10)

project (depth_test_enable_set_skip_use CXX)

if (NOT ANDROID)
  message (FATAL_ERROR "This CMakeLists.txt is Android only, configure with -DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake")
endif ()

# Vulkan 1.3 core commands are exported by the Android NDK stub from API 33.
if (ANDROID_PLATFORM_LEVEL LESS 33)
  message (FATAL_ERROR "depth_test_enable_set_skip_use requires -DANDROID_PLATFORM=android-33 or later")
endif ()

set (CHAPTER_NAME depth_test_enable_set_skip_use)

add_executable (${CHAPTER_NAME} ${CHAPTER_NAME}.cpp)
set_target_properties (${CHAPTER_NAME} PROPERTIES CXX_STANDARD 17)
# NDK sysroot里有libvulkan.so (stub, 运行时用device上的/system/lib64/libvulkan.so).
target_link_libraries (${CHAPTER_NAME} vulkan)
# GLM, stb_image.h and stb_image_write.h are provided by ../../3rdparty.
target_include_directories (${CHAPTER_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty)

# shader用NDK自带的glslc编, 不依赖VULKAN_SDK和AOSP.
find_program (GLSLC glslc HINTS "${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}" NO_DEFAULT_PATH REQUIRED)

set (SHADERS_DIR ${CMAKE_BINARY_DIR}/shaders)
set (SHADER_BASE ${CMAKE_CURRENT_SOURCE_DIR}/${CHAPTER_NAME})
add_custom_command (
  OUTPUT ${SHADERS_DIR}/vert.spv ${SHADERS_DIR}/frag.spv
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  COMMAND ${GLSLC} --target-env=vulkan1.0 ${SHADER_BASE}.vert -o ${SHADERS_DIR}/vert.spv
  COMMAND ${GLSLC} --target-env=vulkan1.0 ${SHADER_BASE}.frag -o ${SHADERS_DIR}/frag.spv
  DEPENDS ${SHADER_BASE}.vert ${SHADER_BASE}.frag
  COMMENT "Compiling Shaders"
  VERBATIM
  )
add_custom_target (${CHAPTER_NAME}_shader DEPENDS ${SHADERS_DIR}/vert.spv ${SHADERS_DIR}/frag.spv)
add_dependencies (${CHAPTER_NAME} ${CHAPTER_NAME}_shader)

# Match the desktop chapter: shaders/ and textures/ live next to the executable.
add_custom_command (TARGET ${CHAPTER_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${CHAPTER_NAME}>/textures"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../images/texture.jpg"
    "$<TARGET_FILE_DIR:${CHAPTER_NAME}>/textures/texture.jpg"
  VERBATIM)
