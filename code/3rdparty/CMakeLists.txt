set (GLFW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/glfw-3.3.8.bin.WIN64")

set (REQUIRED_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/glm/glm.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/stb_image.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/tiny_obj_loader.h")
if (WIN32)
  list (APPEND REQUIRED_FILES
    "${GLFW_DIR}/include/GLFW/glfw3.h"
    "${GLFW_DIR}/glfw3dll.lib"
    "${GLFW_DIR}/glfw3.dll")
endif ()

foreach (REQUIRED_FILE ${REQUIRED_FILES})
  if (NOT EXISTS "${REQUIRED_FILE}")
    message (FATAL_ERROR "Required third-party file not found: ${REQUIRED_FILE}")
  endif ()
endforeach ()

if (WIN32)
  add_library (glfw SHARED IMPORTED GLOBAL)
  set_target_properties (glfw PROPERTIES
    IMPORTED_IMPLIB "${GLFW_DIR}/glfw3dll.lib"
    IMPORTED_LOCATION "${GLFW_DIR}/glfw3.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${GLFW_DIR}/include")
else ()
  # 非 Windows 使用系统安装的 GLFW (例如 libglfw3-dev)
  find_package (glfw3 REQUIRED)
  # find_package 创建的 imported target 仅在当前目录可见, 提升为 GLOBAL 供上层使用
  set_target_properties (glfw PROPERTIES IMPORTED_GLOBAL TRUE)
endif ()

add_library (glm::glm INTERFACE IMPORTED GLOBAL)
set_property (TARGET glm::glm PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

add_library (tinyobjloader::tinyobjloader INTERFACE IMPORTED GLOBAL)
set_property (TARGET tinyobjloader::tinyobjloader PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

set (STB_INCLUDEDIR "${CMAKE_CURRENT_SOURCE_DIR}" PARENT_SCOPE)
