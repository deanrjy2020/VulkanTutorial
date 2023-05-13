#version 450 core

//layout(binding = 1) uniform sampler2D texSampler;
layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput inputColor;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vTexCoord; // not used
layout(location = 0) out vec4 oFrag;

void main()
{
    // draw right circle
    float radius = 50;
    vec2 center = vec2(500, 300);
    if (pow(gl_FragCoord.x-center.x, 2) + pow(gl_FragCoord.y-center.y, 2) < pow(radius, 2)) {
        oFrag = vColor;
    } else {
        // Read color from previous color input attachment
        oFrag = subpassLoad(inputColor).rgba;
    }
}