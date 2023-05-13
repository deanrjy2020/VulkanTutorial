#version 450 core

// binding point改了, code里面两个地方改就可以了,
// 1, vkCreateDescriptorSetLayout时候VkDescriptorSetLayoutBinding.binding改成对应的
// 2, vkUpdateDescriptorSets时候VkWriteDescriptorSet.dstBinding改成对应的
// todo: set如果用1的话就比较麻烦了, 调不出来, 下次有空再说.
layout(set = 0, binding = 2) uniform sampler2D texSampler;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vTexCoord;
layout(location = 0) out vec4 oFrag;

void main()
{
    // draw left circle
    float radius = 50;
    vec2 center = vec2(300, 300);
    if (pow(gl_FragCoord.x-center.x, 2) + pow(gl_FragCoord.y-center.y, 2) < pow(radius, 2)) {
        oFrag = vColor;
    } else {
        oFrag = texture(texSampler, vTexCoord);
    }
}