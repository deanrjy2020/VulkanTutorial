快速运行:
在本章节目录执行 ./run_android.sh.
自动 build, push 到本章节独立 device 目录, 运行 60 frames, 将 last.png 拉回本章节目录.
默认 NDK: ~/code/workspaceLocal/gtb/external/dependencies/ndk, 可用 NDK=<path> ./run_android.sh 覆盖.
多个 device 时使用 ANDROID_SERIAL=<serial> ./run_android.sh.

以下保留手动操作说明:

27.2_zbuf_android_offscreen

从 27.1_depth_buffering_fix 增量加入 Android offscreen, 方式与 15.2 相同.
保留 Vulkan 1.0, 传统 VkRenderPass/VkFramebuffer, texture, UBO, vertex/index buffers 和 depth test.
每帧仍然只有一个 vkCmdDrawIndexed(), 绘制两个前后错开的 textured quads.
没有引入 dynamic rendering, stencil test 或 MSAA.

Android 使用 NDK 编译 adb shell binary, 不需要 AOSP tree, surface 或 swapchain.
一张 RGBA8 SRGB color image 和一张 depth image, 一个 frame slot, 用 fence 串行复用.
Render pass 的 color finalLayout 是 TRANSFER_SRC_OPTIMAL, 显式 subpass dependency 连接 color write 和 transfer read.
Copy 到 HOST_VISIBLE | HOST_COHERENT buffer 后, 用 transfer-to-host barrier, 等 GPU 完成再写 PNG.
Android 每帧使用 1/60 s 的固定 animation timestep, 首帧角度为 0, 便于重复比较截图.
Desktop 保留 27.1 的 GLFW window, swapchain 和 resize 路径.

查看增量:
diff -u ../27.1_depth_buffering_fix/27.1_depth_buffering_fix.cpp 27.2_zbuf_android_offscreen.cpp

# 0. 环境
cd ~/code/workspaceLocal/VulkanTutorial
NDK=~/code/workspaceLocal/gtb/external/dependencies/ndk
B=build/android/27.2_zbuf_android_offscreen

# 1. 编译 binary 和 shaders, 同时复制 texture
cmake -G Ninja -S code/27.2_zbuf_android_offscreen -B "$B" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Debug
cmake --build "$B"

# 2. Push 到独立目录, 避免与其他 chapter 的 shaders 混用
adb shell mkdir -p /data/local/tmp/vt/27.2/shaders /data/local/tmp/vt/27.2/textures
adb push "$B/27.2_zbuf_android_offscreen" /data/local/tmp/vt/27.2/
adb push "$B/shaders/." /data/local/tmp/vt/27.2/shaders/
adb push "$B/textures/." /data/local/tmp/vt/27.2/textures/

# 3. 默认画 1 帧并输出 out.png
adb shell "cd /data/local/tmp/vt/27.2 && ./27.2_zbuf_android_offscreen"
adb pull /data/local/tmp/vt/27.2/out.png .

# 参数: [frameCount [outputFile]]
# 60: 画 60 帧, 将最后一帧写到指定文件
adb shell "cd /data/local/tmp/vt/27.2 && ./27.2_zbuf_android_offscreen 60 last.png"
adb pull /data/local/tmp/vt/27.2/last.png .

# 0: 一直画, 适合 driver debugging; Ctrl+C 终止时不会输出 PNG
adb shell "cd /data/local/tmp/vt/27.2 && ./27.2_zbuf_android_offscreen 0"

Debug build 尝试启用 VK_LAYER_KHRONOS_validation, Android 上找不到时打印说明并继续.
运行目录必须有 shaders/vert.spv, shaders/frag.spv 和 textures/texture.jpg.

# Desktop, 与其他 chapter 相同
cmake -S code -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target 27.2_zbuf_android_offscreen
cd build/27.2_zbuf_android_offscreen
./27.2_zbuf_android_offscreen
