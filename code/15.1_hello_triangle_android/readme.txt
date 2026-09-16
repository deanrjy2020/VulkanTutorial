仿照aosp\platform\frameworks\native\opengl\tests\gl2_basic 纯cpp代码, 能在screen上显示
能跑, 能在app里面单步, 但是不能step到sumd driver里面(要额外配置, 直接在sumd vscode里面配置更方便)

#0 clone
在 ~/code/aosp/platform/frameworks/native/vulkan/下面clone VulkanTutorial
vscode打开VulkanTutorial

# 1. 环境 (新 shell 只需要一次), android16 + Samsung M41
cd ~/code/aosp/platform
export OUT_DIR=out_erd9965_b_bp2a_eng
source build/envsetup.sh
lunch full_erd9965_b-bp2a-eng

# 2. 编译 binary
m 15.1_hello_triangle_android

# 3. 编译 shader, 输出到 VulkanTutorial/code/15.1_hello_triangle_android/shaders/
VT=frameworks/native/vulkan/VulkanTutorial
$VT/code/15.1_hello_triangle_android/compile_shaders.sh

# 4. push
adb root
adb shell mkdir -p /data/local/tmp/vt/shaders
adb push $OUT_DIR/target/product/erd9965/system/bin/15.1_hello_triangle_android /data/local/tmp/vt/
adb push $VT/code/15.1_hello_triangle_android/shaders/. /data/local/tmp/vt/shaders/

# 5. 运行, 默认600 帧, 传 0 = 一直画
adb shell "cd /data/local/tmp/vt && ./15.1_hello_triangle_android"
