#version 450 core

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vTexCoord;
layout(location = 0) out vec4 oFrag;

void main()
{
//    oFrag = vColor;
    oFrag = texture(texSampler, vTexCoord);
}