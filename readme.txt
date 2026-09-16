Supported OS:
Windows 全部
    Windows 上 glfw在3rdparty里面
Linux 全部
    Linux上要自己装: sudo apt install libglfw3-dev
Android 部分, 看app里面readme.txt
    15.1
    15.2
    27.2
    50.1
    50.2

Windows/Linux 编译, 用自带的cmake编译, output在build里面:
cmake -S code -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --config Debug --parallel
用run.sh

单步配置
安装2个
C/C++(Microsoft, 扩展 ID: ms-vscode.cpptools)
CMake Tools(Microsoft, 扩展 ID: ms-vscode.cmake-tools)
    在 VS Code:
    1. 按 Ctrl+Shift+P
    2. 运行 CMake: Select a Kit
    3. Scan for Kits, 选择类似以下的项:
    - Visual Studio Community 2019 Release - amd64
    - 或 Visual Studio Build Tools 2019 Release - amd64
reload window

todo:
