#version 450

layout(binding = 0) uniform sampler2D rayOutput;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(800.0, 600.0); 
    outColor = texture(rayOutput, uv);
    //outColor =vec4(0,1,0,0);
}
