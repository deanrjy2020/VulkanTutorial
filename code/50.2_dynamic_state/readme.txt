50.2_dynamic_state

从 27.2_zbuf_android_offscreen 增量加入通用 dynamic state, 要求 Vulkan 1.3.
保留一个普通 VkPipeline, 传统 VkRenderPass/VkFramebuffer, 每帧一个 vkCmdDrawIndexed().
保留 texture, UBO, vertex/index buffers, depth attachment, Android offscreen PNG 和 desktop GLFW.
Shaders 与 27.2 相同, 不加入 stencil, MSAA, line/tessellation draw 或 static/dynamic 对照格子.

Dynamic state 不依赖 dynamic rendering.
创建 VkPipeline 时通过 pDynamicStates 声明哪些 state 是 dynamic, 在 command buffer 中设置它们的值.
这些字段在 pipeline create info 中的值被忽略, 不是 draw 时可用的默认值.
本例没有 enable dynamicRendering feature, 也没有 vkCmdBeginRendering().

查看增量:
diff -u ../27.2_zbuf_android_offscreen/27.2_zbuf_android_offscreen.cpp 50.2_dynamic_state.cpp

阅读顺序:
1. queryDynamicStateSupport(): 查询 extensions 和 feature bits, 只选择本章需要的功能.
2. createLogicalDevice(): enable 支持的 optional features/extensions.
3. loadDynamicStateFunctions(): 加载 optional EXT commands, 缺失时明确报错.
4. createGraphicsPipeline(): 声明 dynamic state, 同时保留 optional 项的 static fallback.
5. setDynamicStates(): bind pipeline 后设置状态, 随后唯一一次 indexed draw 消费这组配置.

固定 15 项, Vulkan 1.3 core:
  VIEWPORT_WITH_COUNT, SCISSOR_WITH_COUNT
  VERTEX_INPUT_BINDING_STRIDE
  PRIMITIVE_TOPOLOGY, PRIMITIVE_RESTART_ENABLE
  CULL_MODE, FRONT_FACE, RASTERIZER_DISCARD_ENABLE
  DEPTH_BIAS_ENABLE, DEPTH_BIAS
  DEPTH_TEST_ENABLE, DEPTH_WRITE_ENABLE, DEPTH_COMPARE_OP
  DEPTH_BOUNDS_TEST_ENABLE, BLEND_CONSTANTS

最多再加 8 项, 每项单独检查:
  DEPTH_BOUNDS: core depthBounds feature, 支持时 enable test 并设置 [0, 1].
  VERTEX_INPUT_EXT: VK_EXT_vertex_input_dynamic_state / vertexInputDynamicState.
  COLOR_WRITE_ENABLE_EXT: VK_EXT_color_write_enable / colorWriteEnable.
  DEPTH_CLAMP_ENABLE_EXT: EDS3 / extendedDynamicState3DepthClampEnable.
  POLYGON_MODE_EXT: EDS3 / extendedDynamicState3PolygonMode.
  COLOR_BLEND_ENABLE_EXT: EDS3 / extendedDynamicState3ColorBlendEnable.
  COLOR_BLEND_EQUATION_EXT: EDS3 / extendedDynamicState3ColorBlendEquation.
  COLOR_WRITE_MASK_EXT: EDS3 / extendedDynamicState3ColorWriteMask.

EDS3 = VK_EXT_extended_dynamic_state3.
启动时打印 GPU, 每个目标 state 是否使用 dynamic, 以及实际数量, 最多 23 项.
不支持的 optional 项保留 static 值, 不影响其他 supported state.
这不是枚举 Vulkan 的全部 dynamic state, 也不是每个 state 所有取值的 conformance test.

几个便于单步观察的细节:
- Viewport/scissor 改成 WITH_COUNT, pipeline 中的 count 必须为 0, command 设置 count=1.
- Vertex input 支持时设置现有 3 个 attributes, 然后 vkCmdBindVertexBuffers2 设置 buffer 和 stride.
  两种 commands 都能设置 stride, 最后一次设置生效. 没有 vertex input extension 时 attributes 保持 static.
- Topology 仍为 TRIANGLE_LIST, 与 pipeline 的 topology class 相同, 不要求 unrestricted topology.
- Depth test/write 使用 TRUE, compare 使用 LESS. 没有 depthBounds feature 时动态关闭 bounds test.
- Depth bias 动态开启, factors 全为 0, 保持原画面.
- Blend 开启, srcColor=CONSTANT_COLOR, srcAlpha=CONSTANT_ALPHA, dst=ZERO, op=ADD.
  Blend constants 动态设成 {1, 1, 1, 1}, 所以输出仍为 src, 与 27.2 的画面相同.
  Blend enable/equation 的 EDS3 support 分开处理, 任一不支持时使用相同的 static 值.
- Color write mask 为 RGBA, color write enable 为 TRUE.
- Polygon mode 设为 FILL, 不需要 fillModeNonSolid; depth clamp 设为 FALSE, 不需要 core depthClamp.
- Rasterizer discard 和 primitive restart 动态设为 FALSE. 动态关闭功能也是使用 dynamic state.
- Stencil test 保持 static FALSE, sample count 保持 static 1, 不声明相关 dynamic state.

常用的 dynamic state:
Viewport/scissor 是最常见起点; shadow pass 常用 depth bias; stencil rendering 常用 stencil reference/masks,
但本章不演示 stencil. 减少 pipeline variants 时也常用 cull/front face, depth test/write/compare,
blend enable/equation 和 color write mask. Vertex input/stride 是否需要 dynamic 取决于 vertex layout 的变化.
Dynamic state 更多不代表性能一定更好, 需要结合具体 app 和 driver 测量.

build and run on Android:
在 50.2_dynamic_state 目录下执行:
./run_android.sh

自动编译并 push 到 Android, 运行 60 frames, 输出保存为本目录的 last.png.
多个 device 时使用 ANDROID_SERIAL=<serial> ./run_android.sh.
NDK 默认使用 gtb/external/dependencies/ndk, 可用 NDK=<path> ./run_android.sh 覆盖.

Debug build 尝试启用 VK_LAYER_KHRONOS_validation, Android 上找不到时打印说明并继续.
有限帧数使用与 27.2 相同的固定 animation timestep, 可对比相同帧数的 PNG.

# Desktop
cmake -S code -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target 50.2_dynamic_state
cd build/50.2_dynamic_state
./50.2_dynamic_state

References:
https://docs.vulkan.org/guide/latest/dynamic_state.html
https://docs.vulkan.org/guide/latest/dynamic_state_map.html
https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetVertexInputEXT.html
