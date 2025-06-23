wget http://ports.ubuntu.com/ubuntu-ports/pool/universe/g/glfw3/libglfw3_3.3.2-1_arm64.deb
wget http://ports.ubuntu.com/ubuntu-ports/pool/universe/g/glfw3/libglfw3-dev_3.3.2-1_arm64.deb

mkdir glfw_runtime && dpkg-deb -x libglfw3_3.3.2-1_arm64.deb glfw_runtime
mkdir glfw_dev && dpkg-deb -x libglfw3-dev_3.3.2-1_arm64.deb glfw_dev
