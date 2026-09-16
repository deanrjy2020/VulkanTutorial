#!/bin/bash

# 用AOSP里的glslangValidator编这个chapter的shader, 输出到本目录的 shaders/{vert,frag}.spv (方便和binary一起push).
# Android.bp里不编shader(soong不允许引用module目录外的文件), 所以单独用这个脚本.
#
# usage: compile_shaders.sh   (不用参数, 在哪个目录下跑都行)
#
# 默认用 $ANDROID_BUILD_TOP (lunch之后有), 没有就按这个脚本在
# frameworks/native/vulkan/VulkanTutorial/code/15.1_hello_triangle_android/ 下往上推. 也可以用 GLSLANG=<path> 直接指定.

set -e

chapter_dir=$(cd "$(dirname "$0")" && pwd)
chapter=$(basename "$chapter_dir")
aosp_top=${ANDROID_BUILD_TOP:-$(cd "$chapter_dir/../../../../../.." && pwd)}
glslang=${GLSLANG:-$aosp_top/prebuilts/android-emulator/linux-x86_64/lib64/vulkan/glslangValidator}

if [ ! -f "$glslang" ]; then
    echo "glslangValidator not found: $glslang"
    exit 1
fi

# AOSP里的这个prebuilt没有exec bit, 拷一份到临时目录再加上.
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
cp "$glslang" "$tmp_dir/glslangValidator"
chmod +x "$tmp_dir/glslangValidator"

sources=()
for ext in vert frag comp; do
    if [ -f "$chapter_dir/$chapter.$ext" ]; then
        sources+=("$chapter_dir/$chapter.$ext")
    fi
done

if [ ${#sources[@]} = 0 ]; then
    echo "no shader found: $chapter_dir/$chapter.{vert,frag,comp}"
    exit 1
fi

# 和CMake一样, 在输出目录下编, glslangValidator会按stage命名成vert.spv/frag.spv/comp.spv.
out_dir=$chapter_dir/shaders
mkdir -p "$out_dir"
cd "$out_dir"
"$tmp_dir/glslangValidator" --target-env vulkan1.0 "${sources[@]}" --quiet

echo "shaders written to $out_dir:"
ls "$out_dir"
