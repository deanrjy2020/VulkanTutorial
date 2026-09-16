#!/usr/bin/env bash
set -euo pipefail

# 默认运行 triangle, 也可指定章节: ./run.sh 50.2_dynamic_state
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
chapter="${1:-15_hello_triangle}"
if [[ $# -gt 1 || ! "$chapter" =~ ^[0-9][A-Za-z0-9_.]*$ ]]; then
    printf 'Usage: %s [chapter, e.g. 50.2_dynamic_state]\n' "$0" >&2
    exit 1
fi
if [[ ! -f "$repo_dir/code/$chapter.cpp" && ! -f "$repo_dir/code/$chapter/$chapter.cpp" ]]; then
    printf 'Chapter not found: %s\n' "$chapter" >&2
    exit 1
fi

# 编译去全部及其 shaders, 保留 compile_commands.json 供编辑器使用.
cmake -S "$repo_dir/code" -B "$repo_dir/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$repo_dir/build" --config Debug --parallel

# 程序通过相对路径读取 shaders/textures, 从对应 build 目录启动.
cd "$repo_dir/build/$chapter"
echo "running..."
exec "./$chapter"
