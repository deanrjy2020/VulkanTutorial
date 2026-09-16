cmake_minimum_required (VERSION 3.8)

project (VulkanTutorial)

# 从仓库内的 thirdparty 目录加载 GLFW、GLM、STB 和 tinyobjloader
add_subdirectory ("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty" thirdparty)
find_package (Vulkan REQUIRED)

add_executable (glslang::validator IMPORTED)
find_program (GLSLANG_VALIDATOR "glslangValidator" HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
set_property (TARGET glslang::validator PROPERTY IMPORTED_LOCATION "${GLSLANG_VALIDATOR}")

function (add_shaders_target TARGET)
  cmake_parse_arguments ("SHADER" "" "CHAPTER_NAME" "SOURCES" ${ARGN})
  # RUNTIME_OUTPUT_DIRECTORY 已将可执行文件放在章节目录中；shader 也必须
  # 放在同一目录下，程序从资源管理器双击启动时才能找到 shaders/*.spv。
  set (SHADERS_DIR ${CMAKE_BINARY_DIR}/${SHADER_CHAPTER_NAME}/shaders)
  set (SHADERS ${SHADERS_DIR}/frag.spv ${SHADERS_DIR}/vert.spv)
  # Some chapters may have compute shaders in addition to vertex and fragment shaders,
  # so we conditionally check this and add them to the target
  string(FIND "${SHADER_SOURCES}" "${CHAPTER_SHADER}.comp" COMPUTE_SHADER_INDEX)
  if (${COMPUTE_SHADER_INDEX} GREATER -1)  
    set (SHADERS ${SHADERS} ${SHADERS_DIR}/comp.spv)
  endif()
  add_custom_command (
    OUTPUT ${SHADERS}
    # 编译着色器前确保目标目录存在
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
    COMMAND glslang::validator
    ARGS --target-env vulkan1.0 ${SHADER_SOURCES} --quiet
    WORKING_DIRECTORY ${SHADERS_DIR}
    DEPENDS ${SHADER_SOURCES}
    COMMENT "Compiling Shaders"
    VERBATIM
    )
  add_custom_target (${TARGET} DEPENDS ${SHADERS})
endfunction ()

function (add_chapter CHAPTER_NAME)
  cmake_parse_arguments (CHAPTER "" "SOURCE;SHADER" "LIBS;TEXTURES;MODELS" ${ARGN})

  if (NOT DEFINED CHAPTER_SOURCE)
    set (CHAPTER_SOURCE ${CHAPTER_NAME}.cpp)
  endif ()
  add_executable (${CHAPTER_NAME} ${CHAPTER_SOURCE})
  set_target_properties (${CHAPTER_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CHAPTER_NAME})
  set_target_properties (${CHAPTER_NAME} PROPERTIES CXX_STANDARD 17)
  target_link_libraries (${CHAPTER_NAME} Vulkan::Vulkan glfw)
  target_include_directories (${CHAPTER_NAME} PRIVATE ${STB_INCLUDEDIR})

  # 将 GLFW 动态库复制到可执行文件旁，方便直接运行程序
  if (WIN32 AND TARGET glfw)
    add_custom_command (TARGET ${CHAPTER_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:glfw> $<TARGET_FILE_DIR:${CHAPTER_NAME}>)
  endif ()

  if (DEFINED CHAPTER_SHADER)
    set (CHAPTER_SHADER_TARGET ${CHAPTER_NAME}_shader)
    file (GLOB SHADER_SOURCES ${CHAPTER_SHADER}.frag ${CHAPTER_SHADER}.vert ${CHAPTER_SHADER}.comp)
    add_shaders_target (${CHAPTER_SHADER_TARGET} CHAPTER_NAME ${CHAPTER_NAME} SOURCES ${SHADER_SOURCES})
    add_dependencies (${CHAPTER_NAME} ${CHAPTER_SHADER_TARGET})
  endif ()
  if (DEFINED CHAPTER_LIBS)
    target_link_libraries (${CHAPTER_NAME} ${CHAPTER_LIBS})
  endif ()
  if (DEFINED CHAPTER_MODELS)
    # 将模型复制到可执行文件同级的 models 目录
    add_custom_command (TARGET ${CHAPTER_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:${CHAPTER_NAME}>/models"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CHAPTER_MODELS} "$<TARGET_FILE_DIR:${CHAPTER_NAME}>/models"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      VERBATIM)
  endif ()
  if (DEFINED CHAPTER_TEXTURES)
    # 将纹理复制到可执行文件同级的 textures 目录
    add_custom_command (TARGET ${CHAPTER_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:${CHAPTER_NAME}>/textures"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CHAPTER_TEXTURES} "$<TARGET_FILE_DIR:${CHAPTER_NAME}>/textures"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      VERBATIM)
  endif ()
endfunction ()

add_chapter (00_base_code)

add_chapter (01_instance_creation)

add_chapter (02_validation_layers)

add_chapter (03_physical_device_selection)

add_chapter (04_logical_device)

add_chapter (05_window_surface)

add_chapter (06_swap_chain_creation)

add_chapter (07_image_views)

add_chapter (08_graphics_pipeline)

add_chapter (09_shader_modules
  SHADER 09_shader_base)

add_chapter (10_fixed_functions
  SHADER 09_shader_base)

add_chapter (11_render_passes
  SHADER 09_shader_base)

add_chapter (12_graphics_pipeline_complete
  SHADER 09_shader_base)

add_chapter (13_framebuffers
  SHADER 09_shader_base)

add_chapter (14_command_buffers
  SHADER 09_shader_base)

add_chapter (15_hello_triangle
  SHADER 09_shader_base)

add_chapter (16_frames_in_flight
  SHADER 09_shader_base)

add_chapter (17_swap_chain_recreation
  SHADER 09_shader_base)

add_chapter (18_vertex_input
  SHADER 18_shader_vertexbuffer
  LIBS glm::glm)

add_chapter (19_vertex_buffer
  SHADER 18_shader_vertexbuffer
  LIBS glm::glm)

add_chapter (20_staging_buffer
  SHADER 18_shader_vertexbuffer
  LIBS glm::glm)

add_chapter (21_index_buffer
  SHADER 18_shader_vertexbuffer
  LIBS glm::glm)

add_chapter (22_descriptor_set_layout
  SHADER 22_shader_ubo
  LIBS glm::glm)

add_chapter (23_descriptor_sets
  SHADER 22_shader_ubo
  LIBS glm::glm)

add_chapter (24_texture_image
  SHADER 22_shader_ubo
  TEXTURES ../images/texture.jpg
  LIBS glm::glm)

add_chapter (25_sampler
  SHADER 22_shader_ubo
  TEXTURES ../images/texture.jpg
  LIBS glm::glm)

add_chapter (26_texture_mapping
  SHADER 26_shader_textures
  TEXTURES ../images/texture.jpg
  LIBS glm::glm)

add_chapter (27_depth_buffering
  SHADER 27_shader_depth
  TEXTURES ../images/texture.jpg
  LIBS glm::glm)

add_chapter (27.1_depth_buffering_fix
  SOURCE 27.1_depth_buffering_fix/27.1_depth_buffering_fix.cpp
  SHADER 27.1_depth_buffering_fix/27.1_depth_buffering_fix
  TEXTURES ../images/texture.jpg
  LIBS glm::glm)

add_chapter (28_model_loading
  SHADER 27_shader_depth
  MODELS ../resources/viking_room.obj
  TEXTURES ../resources/viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (29_mipmapping
  SHADER 27_shader_depth
  MODELS ../resources/viking_room.obj
  TEXTURES ../resources/viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (30_multisampling
  SHADER 27_shader_depth
  MODELS ../resources/viking_room.obj
  TEXTURES ../resources/viking_room.png
  LIBS glm::glm tinyobjloader::tinyobjloader)

add_chapter (31_compute_shader
  SHADER 31_shader_compute
  LIBS glm::glm)

add_chapter (31.1_cs_rt_basic
  SOURCE 31.1_cs_rt_basic/31.1_cs_rt_basic.cpp
  SHADER 31.1_cs_rt_basic/31.1_cs_rt_basic
  LIBS glm::glm)
target_include_directories (31.1_cs_rt_basic PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/includes")

add_chapter (31.2_cs_rt_bvh
  SOURCE 31.2_cs_rt_bvh/31.2_cs_rt_bvh.cpp
  SHADER 31.2_cs_rt_bvh/31.2_cs_rt_bvh
  LIBS glm::glm)
target_include_directories (31.2_cs_rt_bvh PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/includes")

add_chapter (40_vk_rt_basic
  SOURCE 40_vk_rt_basic/40_vk_rt_basic.cpp
  SHADER 40_vk_rt_basic/40_vk_rt_basic
  TEXTURES ../images/texture.jpg
  LIBS glm::glm)
