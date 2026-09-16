#!/usr/bin/env bash
set -euo pipefail

# 按脚本位置定位, 在 50.2_dynamic_state 目录下直接 ./run_android.sh 即可.
chapter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$chapter_dir/../.." && pwd)"
ndk_dir="${NDK:-$HOME/code/workspaceLocal/gtb/external/dependencies/ndk}"
build_dir="$repo_dir/build/android/50.2_dynamic_state"
device_dir=/data/local/tmp/vt/50.2

# 1. 编译 Android binary, shaders 和 texture.
# Vulkan 1.3 core commands 需要 android-33 或更新的 NDK stub.
cmake -G Ninja -S "$chapter_dir" -B "$build_dir" \
    -DCMAKE_TOOLCHAIN_FILE="$ndk_dir/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33 -DCMAKE_BUILD_TYPE=Debug
cmake --build "$build_dir"

# 2. Push 到本章独立目录. 多个 device 时可用 ANDROID_SERIAL 选择目标.
adb shell mkdir -p "$device_dir/shaders" "$device_dir/textures"
adb push "$build_dir/50.2_dynamic_state" "$device_dir/"
adb push "$build_dir/shaders/." "$device_dir/shaders/"
adb push "$build_dir/textures/." "$device_dir/textures/"

# 3. 运行 60 frames, 将 PNG 保存到本脚本所在目录.
echo "running..."
adb shell "cd '$device_dir' && ./50.2_dynamic_state 60 last.png"
adb pull "$device_dir/last.png" "$chapter_dir/last.png"
printf 'PNG: %s\n' "$chapter_dir/last.png"

# 持续运行用于 driver debugging: 把上面的运行命令改为 ./50.2_dynamic_state 0.
# 这种模式在 Ctrl+C 时不输出 PNG, 同时需要去掉 adb pull.
