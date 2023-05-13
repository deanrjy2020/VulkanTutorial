#version 450 core

const vec2 positions[4] = {
    vec2(-0.5, -0.5),
    vec2( 0.5, -0.5),
    vec2( 0.5,  0.5),
    vec2(-0.5,  0.5),
};
const vec4 colors[4] = {
    vec4(1.0, 0.0, 0.0, 1.0),
    vec4(0.0, 1.0, 0.0, 1.0),
    vec4(0.0, 0.0, 1.0, 1.0),
    vec4(0.0, 1.0, 1.0, 1.0),
};
const vec2 texCoord[4] = {
    vec2(0.0, 1.0),
    vec2(1.0, 1.0),
    vec2(1.0, 0.0),
    vec2(0.0, 0.0),
};

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vTexCoord;

void main()
{
    int i = gl_VertexIndex % 4;
    vColor = colors[i];
    vTexCoord = texCoord[i];
    gl_Position = vec4(positions[i], 0, 1);
}