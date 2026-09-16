#!/usr/bin/env bash
set -euo pipefail

# 按脚本位置定位, 在 15.2_hello_triangle_android_offscreen 目录下直接 ./run_android.sh 即可.
chapter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$chapter_dir/../.." && pwd)"
ndk_dir="${NDK:-$HOME/code/workspaceLocal/gtb/external/dependencies/ndk}"
build_dir="$repo_dir/build/android/15.2_hello_triangle_android_offscreen"
device_dir=/data/local/tmp/vt/15.2

# 1. 编译 Android binary 和 shaders.
# 本章使用 Vulkan 1.0, Android API level 从 24 起.
cmake -G Ninja -S "$chapter_dir" -B "$build_dir" \
    -DCMAKE_TOOLCHAIN_FILE="$ndk_dir/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DCMAKE_BUILD_TYPE=Debug
cmake --build "$build_dir"

# 2. Push 到本章独立目录. 多个 device 时可用 ANDROID_SERIAL 选择目标.
adb shell mkdir -p "$device_dir/shaders"
adb push "$build_dir/15.2_hello_triangle_android_offscreen" "$device_dir/"
adb push "$build_dir/shaders/." "$device_dir/shaders/"

# 3. 运行 60 frames, 将 PNG 保存到本脚本所在目录.
echo "running..."
adb shell "cd '$device_dir' && ./15.2_hello_triangle_android_offscreen 60 last.png"
adb pull "$device_dir/last.png" "$chapter_dir/last.png"
printf 'PNG: %s\n' "$chapter_dir/last.png"

# 持续运行用于 driver debugging: 把上面的运行命令改为 ./15.2_hello_triangle_android_offscreen 0.
# 这种模式在 Ctrl+C 时不输出 PNG, 同时需要去掉 adb pull.
