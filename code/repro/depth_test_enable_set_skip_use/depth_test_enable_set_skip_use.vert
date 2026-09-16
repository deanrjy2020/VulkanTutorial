#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 color;
void main() {
    // 固定 clip-space depth, 不使用 UBO rotation 或 projection.
    gl_Position = vec4(inPosition, 1.0);
    color = inColor;
}
