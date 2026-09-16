快速运行:
在本章节目录执行 ./run_android.sh.
自动 build, push 到本章节独立 device 目录, 运行 60 frames, 将 last.png 拉回本章节目录.
默认 NDK: ~/code/workspaceLocal/gtb/external/dependencies/ndk, 可用 NDK=<path> ./run_android.sh 覆盖.
多个 device 时使用 ANDROID_SERIAL=<serial> ./run_android.sh.

以下保留手动操作说明:

50.1 dynamic rendering: 从15.2 copy, 画的一样, render pass/framebuffer换成vkCmdBeginRendering + 自己写的sync2 barrier.
和15.2 diff一下就是全部要学的东西: diff ../15.2_hello_triangle_android_offscreen/15.2_hello_triangle_android_offscreen.cpp 50.1_dynamic_rendering.cpp
需要Vulkan 1.3. Android上NDK要android-33以上(stub才导出1.3的函数), 其他和15.2一样:
和gtb的--android-executable一样: NDK + CMake编一个adb shell binary, 不需要AOSP tree.
拿不到ANativeWindow, 所以不上屏. 画到自己创建的VkImage上, copy到host visible buffer, 写成out.png, adb pull回来看.
Windows/GLFW还是走#else分支, 在../CMakeLists.txt里编, 开窗口显示.

和GTB一样, 编译了全速跑, 单步在sumd里面配置launch.json

#0 环境
NDK用gtb下载的那个(r26), 其他NDK也行.
cd ~/code/workspaceLocal/VulkanTutorial
NDK=~/code/workspaceLocal/gtb/external/dependencies/ndk
B=build/android/50.1_dynamic_rendering

# 1. 编译 binary + shader (shader用NDK自带的glslc, 输出到$B/shaders)
cmake -G Ninja -S code/50.1_dynamic_rendering -B $B \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33 -DCMAKE_BUILD_TYPE=Debug
cmake --build $B

# 2. push
adb root
adb shell mkdir -p /data/local/tmp/vt/shaders
adb push $B/50.1_dynamic_rendering /data/local/tmp/vt/
adb push $B/shaders/. /data/local/tmp/vt/shaders/

# 3. 运行, 默认画1帧写到out.png. 参数: [frameCount [outputFile]], frameCount 0 = 一直画(不写文件)
adb shell "cd /data/local/tmp/vt && ./50.1_dynamic_rendering"

# 4. 看结果
adb pull /data/local/tmp/vt/out.png .
Debug build没有NDEBUG, 会去找VK_LAYER_KHRONOS_validation, device上没有就打印一行warning然后关掉validation继续跑.
