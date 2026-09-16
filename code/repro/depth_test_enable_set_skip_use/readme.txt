depth_test_enable_set_skip_use reproducer

基于 CTS 3f781da.diff, 复用 50.2 的 build, launch 和 Android PNG 路径.
在本目录执行 ./run_android.sh, 默认运行 1 frame, 输出 last.png.
Binary 参数仍为 [frameCount [outputFile]], 0 表示持续运行.

要验证的问题:
验证 dynamic depthTestEnable 的值能否跨过一个没有 DS state 的 pipeline bind/draw 而保持不变.
这里 DS 指 depth/stencil state, 不是 descriptor set.

对应 CTS test:
dEQP-VK.pipeline.monolithic.extended_dynamic_state.misc.set_skip_use_depth_test_enable
来源: https://gerrit.khronos.org/c/vk-gl-cts/+/20908
Patch: 3f781da8cb19672ac589039c1362c9e75b31cf68, VK-GL-CTS issue 6800.
本 app 提取传统 render pass + monolithic pipeline 路径, 用两个重叠 quad 的颜色代替 depth readback 判断.

触发条件和测试预期:
1. App 先调用 vkCmdSetDepthTestEnable(TRUE), 把 dynamic depth test 设置为开启.
2. App bind Skip pipeline. 对应 subpass 没有 depth attachment, pDepthStencilState 为 nullptr.
   Skip 不提供 DS state, 也没有把 depthTestEnable 声明为 dynamic.
3. 按此 CTS patch 的测试预期, Skip bind/draw 不应破坏此前设置的 dynamic depthTestEnable.
   这里测试的是缺少 DS state 的 pipeline, 不是提供 static depthTestEnable=FALSE 的 pipeline.
4. App bind Final pipeline. Final 使用 dynamic depthTestEnable, 应继续使用此前设置的 TRUE.
   Final create info 中的 depthTestEnable=FALSE 是被忽略的字段, 不应成为 dynamic state 的新值.
5. App 不再调用 depth test setter. 如果在 Final draw 前重新设置 TRUE, 就会掩盖要调查的问题.

要调查的 driver 行为:
Skip pipeline 不需要 depth test. Driver 可能为这次 draw 使用关闭 depth test 的内部 state.
需要检查这个内部 state 是否错误覆盖了 command buffer 保存的 dynamic 值,
或者 Final bind/draw 是否没有重新使用保存的 TRUE.
这是待排查的方向, 不是已经确认的 SUMD root cause.

结果如何解释:
- Depth test 保持 TRUE: near quad 先写入 depth=0.25, far quad 的 0.75 不满足 LESS, 中心保持绿色.
- Depth test 变成 FALSE: 后画的 far quad 覆盖 near quad, 中心变红.
- 红色中心能表明可见结果不符合测试预期, 但不能单凭 PNG 确定哪个 driver state 被覆盖.
  还需在下面列出的断点处检查 setter 保存的值, Skip bind 的修改, 以及 Final draw 使用的值.
- 绿色中心只说明当前运行通过这条复现路径, 不代表所有 CTS variants 都通过.

Render pass:
- Subpass 0: color only. Skip pipeline 的 pDepthStencilState=nullptr, dynamic list 无 DS state.
- Subpass 1: color + depth. Final pipeline 的 depthTestEnable 为 dynamic.
  depthWriteEnable=TRUE 和 depthCompareOp=LESS 为 static. Depth 首次使用时 clear 为 1.0.

Command 顺序:
1. Begin render pass, vkCmdSetDepthTestEnable(TRUE), 仅设置一次.
2. Bind Skip, draw far quad.
3. Next subpass, bind Final.
4. Draw near green quad, z=0.25; draw far red quad, z=0.75.
5. End render pass, 复用 color readback 保存 PNG, 不读取 depth buffer.

正确结果: 黑色背景, 红色外围, 绿色中心.
Depth test 丢失并变成 FALSE 时: 中心被后画的 far quad 覆盖, 整个 quad 为红色.
Shader 使用固定 clip-space coordinates 和纯色, 不读取 texture 或 UBO.
保留原例子的 texture/UBO 初始化和清理, 以复用原资源流程.

每个 frame 录制 1 次 depth test setter, 2 次 pipeline bind, 3 次 draw.
Driver 断点: SetDepthTestEnableEXT, ProgramDepthStencil, BindStaticPipelineStateObjects, ValidateGraphicsStates.
实际命中次数由 driver 内部路径决定.
单步建议 frameCount=1. 复用现有 launch 配置时, 将 executable 和 device 工作目录改成下面的独立路径.
Host binary: build/android/repro/depth_test_enable_set_skip_use/depth_test_enable_set_skip_use
Device 工作目录: /data/local/tmp/vt/repro/depth_test_enable_set_skip_use

本 reproducer 不加入学习 demo 的默认 build. 保留 Windows/Linux 源码分支, 当前只验证独立 Android build.

本次验证:
- Android build 和 desktop C++ 编译通过.
- Host lavapipe offscreen: 绿色中心; 临时 FALSE 对照版本: 红色中心.
- 当前连接 Android device: 绿色中心, 本次没有复现红色覆盖.
- Host SDK 1.4.335 和 device validation layer 均报告 VUID-vkCmdDrawIndexed-None-07843.
  它们认为 Skip bind 使此前的 setter 失效. 为保留 CTS 测试序列, 没有在 Final draw 前补 setter.
  这不是 validation-clean 的验证结果, 也不能据此证明 SUMD fix 是否已包含在当前 device build 中.

Build only, 在 repo root 执行:
cmake -S code/repro/depth_test_enable_set_skip_use -B build/android/repro/depth_test_enable_set_skip_use -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$HOME/code/workspaceLocal/gtb/external/dependencies/ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33 -DCMAKE_BUILD_TYPE=Debug
cmake --build build/android/repro/depth_test_enable_set_skip_use

也可在本目录执行 ./run_android.sh, 自动 build, push, 运行 1 frame 并拉回 last.png.
NDK=<path> 可覆盖 NDK 路径, ANDROID_SERIAL=<serial> 可选择 device.
