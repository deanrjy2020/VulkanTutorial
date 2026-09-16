编译, 用自带的cmake编译, output在build里面:
cmake -S code -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --config Debug --parallel

单步配置
安装2个
C/C++（Microsoft，扩展 ID：ms-vscode.cpptools）
CMake Tools（Microsoft，扩展 ID：ms-vscode.cmake-tools）
    在 VS Code：
    1. 按 Ctrl+Shift+P
    2. 运行 CMake: Select a Kit
    3. Scan for Kits, 选择类似以下的项：
    - Visual Studio Community 2019 Release - amd64
    - 或 Visual Studio Build Tools 2019 Release - amd64
reload window

todo:
