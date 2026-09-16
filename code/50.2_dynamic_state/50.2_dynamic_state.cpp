// 50.2: 从 27.2 增量加入 dynamic state, 要求 Vulkan 1.3.
// Dynamic state 不依赖 dynamic rendering. 普通 VkPipeline 配合传统 VkRenderPass 也能使用.
// Pipeline 创建时用 pDynamicStates 声明哪些 state 是 dynamic, draw 前用 command buffer 设置值.
// 对这些 state, pipeline create info 中的值会被忽略, 不会成为 command buffer 的默认值.
// 一个 pipeline, 每帧一个 indexed draw, 同时使用本例支持的通用 dynamic state.
// 保留 27.2 的 texture, UBO, vertex/index buffers, depth attachment 和 Android offscreen PNG.
// 不添加 stencil, MSAA, line/tessellation draw, 不做 static/dynamic 对照.
//
// 固定 15 项: viewport/scissor WITH_COUNT, vertex binding stride, primitive topology/restart,
// cull mode, front face, rasterizer discard, depth bias enable/factors, depth test/write/compare,
// depth bounds test enable, blend constants. EDS1/EDS2 中用到的这些项已进入 Vulkan 1.3 core.
// Optional: depth bounds; EXT_vertex_input_dynamic_state; EXT_color_write_enable;
// EDS3 的 depth clamp enable, polygon mode, color blend enable/equation, color write mask.
// 所有 optional 项支持时共 23 项. 逐项查询和 enable, 不支持的保留 static 设置并打印说明.
// Blend 使用 src * 1 + dst * 0, depth bias 为 0, 尽量保持 27.2 的画面便于比较.
// 关闭某个功能的 dynamic enable state 仍然是 dynamic state; 不代表测试了所有取值或视觉效果.
// 常用起点: viewport/scissor; shadow pass 常用 depth bias; 减少 pipeline variants 时常用
// cull/front face, depth test/write/compare 和 blend/write mask. 更多 dynamic state 不保证更快.
// Spec: https://docs.vulkan.org/guide/latest/dynamic_state.html
// Map: https://docs.vulkan.org/guide/latest/dynamic_state_map.html

#if defined(__ANDROID__)
// Android: NDK编的adb shell binary, 和gtb的--android-executable一样, 拿不到ANativeWindow, 所以不上屏.
// 没有surface/swapchain, 画到自己创建的VkImage上, 最后copy到host visible buffer, 写成out.png, adb pull回来看.
#include <vulkan/vulkan.h>
// stb是single-header库, 必须在一个.cpp里define这个开关才会展开函数实现, 省不掉. 它本身只是开关,
// stbi_write_png是普通函数, 可以单步进去.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#else
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif


#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <string>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

/*
修正27中的下面问题:

drawFrame(), globalFrameId:0, imageIndex(=image slot): 0, currentFrame(=in flight slot): 0
drawFrame(), globalFrameId:1, imageIndex(=image slot): 1, currentFrame(=in flight slot): 1
drawFrame(), globalFrameId:2, imageIndex(=image slot): 2, currentFrame(=in flight slot): 0
validation layer: vkQueueSubmit(): pSubmits[0].pSignalSemaphores[0] (VkSemaphore 0x2d000000002d) is being signaled by VkQueue 0x25f326048f8, but it may still be in use by VkSwapchainKHR 0x30000000003.
Here are the most recently acquired image indices: [0], 1, 2.
(brackets mark the last use of VkSemaphore 0x2d000000002d in a presentation operation)
Swapchain image 0 was presented but was not re-acquired, so VkSemaphore 0x2d000000002d may still be in use and cannot be safely reused with image index 2.
Vulkan insight: One solution is to assign each image its own semaphore. Here are some common methods to ensure that a semaphore passed to vkQueuePresentKHR is not in use and can be safely reused:
        a) Use a separate semaphore per swapchain image. Index these semaphores using the index of the acquired image.
        b) Consider the VK_EXT_swapchain_maintenance1 extension. It allows using a VkFence with the presentation operation.
The Vulkan spec states: Each binary semaphore element of the pSignalSemaphores member of any element of pSubmits must be unsignaled when the semaphore signal operation it defines is executed on the device (https://vulkan.lunarg.com/doc/view/1.4.313.2/windows/antora/spec/latest/chapters/cmdbuffers.html#VUID-vkQueueSubmit-pSignalSemaphores-00067)

Rootcause:
frame 2 vkQueueSubmit 绘制 image 2 时, 要使用 renderFinishedSemaphores[0] 作为"绘制完成"的信号; 但 renderFinishedSemaphores[0] 可能还被 frame 0 的 image 0 presentation 使用, 因此不能再次 绑定这个 renderFinishedSemaphores[0].
即renderFinishedSemaphores跟in flight走, 只有2个不够用, 要跟imageCount=3走.

fix:
3个renderFinishedSemaphores
这也是 Khronos 官方推荐的处理方式: https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
*/


/*
"先分配 descriptor, 再组装成 set"这个模型不对.
[第 1128 行 (line 1128)](/D:/VulkanTutorial/code/27_depth_buffering.cpp:1128): vkAllocateDescriptorSets 直接从 pool 分配 set, 同时消耗相应类型的 descriptor 配额. descriptorSets.resize() 只给 CPU 侧保存 handle 的数组分配空间.
[第 1164 行 (line 1164)](/D:/VulkanTutorial/code/27_depth_buffering.cpp:1164) 也应改成: 两个 VkWriteDescriptorSet 更新同一个 set 的两个 binding, 不是更新两个 set; 写入的是资源引用等描述信息, 不是资源数据. 官方说明

DEVICE_LOCAL 不意味着 CPU 不能访问, VBO 也不是必须 staging.
[第 1055 行 (line 1055)](/D:/VulkanTutorial/code/27_depth_buffering.cpp:1055), [第 1088 行 (line 1088)](/D:/VulkanTutorial/code/27_depth_buffering.cpp:1088), [第 1187 行 (line 1187)](/D:/VulkanTutorial/code/27_depth_buffering.cpp:1187):
能否 vkMapMemory 看的是 HOST_VISIBLE. 同一种 memory type 可以同时具有 DEVICE_LOCAL | HOST_VISIBLE, 例如 UMA 上常见这种组合. staging 是常用上传方案, 不是所有 VBO 都必须走的机制; "HOST_VISIBLE 性能远低""DEVICE_LOCAL 性能最好"也不能一概而论. 官方内存说明
*/

// 共享的:
//   descriptor pool 只有一个
// 每个in flight frame有自己的:
//   fence和imageAvailable semaphore; renderFinished semaphore按swapchain image分配
//   UBO
//   descriptor set
//   descriptor set layout 两个一样的DS layout
//   command buffer
#if defined(__ANDROID__)
// One shared color/depth/readback set: wait for its fence before reusing it.
const int MAX_FRAMES_IN_FLIGHT = 1;
uint32_t frameCount = 1;
const char* outputFile = "out.png";
#else
const int MAX_FRAMES_IN_FLIGHT = 2;
#endif

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#if defined(__ANDROID__)
const std::vector<const char*> deviceExtensions = {};
#else
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
#endif

#ifdef NDEBUG
const bool enableValidationLayers = false;
#elif defined(__ANDROID__)
bool enableValidationLayers = true;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
#if defined(__ANDROID__)
        return graphicsFamily.has_value();
#else
        return graphicsFamily.has_value() && presentFamily.has_value();
#endif
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    // 这个是面向binding point的, 一个point对应一个binding description
    // VBO也是binding到这个point上的, 在vkCmdBeginRenderPass后的vkCmdBindPipeline后的vkCmdBindVertexBuffers
    // 这里的description事实上在描述某个binding point的VBO的stride.
    // 见vertex input description page的图片.
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    // 这个是面向attribute的, 一个attribute对应一个attribute description, 所以这里是3个元素的array.
    // attribute指定在哪个bingding里面
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};

// 一个ubo里面就有三个mat4.
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
#if !defined(__ANDROID__)
    GLFWwindow* window;
#endif

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

#if defined(__ANDROID__)
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkBuffer readbackBuffer;
    VkDeviceMemory readbackBufferMemory;
#endif

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    // 这些成员保存准备传给 vkCreateDevice 的 feature bits, EDS3 只 enable 本章用到的项.
    VkBool32 depthBoundsSupported = VK_FALSE;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3Features{};
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vertexInputFeatures{};
    VkPhysicalDeviceColorWriteEnableFeaturesEXT colorWriteFeatures{};
    std::vector<const char*> enabledDeviceExtensions;
    // Extension name 存在只代表可以查询对应 features, 不代表所有 feature bits 都支持.
    struct {
        bool extendedDynamicState3 = false;
        bool vertexInputDynamicState = false;
        bool colorWriteEnable = false;
    } dynamicStateExtensions{};

    PFN_vkCmdSetVertexInputEXT pfnCmdSetVertexInputEXT = nullptr;
    PFN_vkCmdSetColorWriteEnableEXT pfnCmdSetColorWriteEnableEXT = nullptr;
    PFN_vkCmdSetDepthClampEnableEXT pfnCmdSetDepthClampEnableEXT = nullptr;
    PFN_vkCmdSetPolygonModeEXT pfnCmdSetPolygonModeEXT = nullptr;
    PFN_vkCmdSetColorBlendEnableEXT pfnCmdSetColorBlendEnableEXT = nullptr;
    PFN_vkCmdSetColorBlendEquationEXT pfnCmdSetColorBlendEquationEXT = nullptr;
    PFN_vkCmdSetColorWriteMaskEXT pfnCmdSetColorWriteMaskEXT = nullptr;

    VkCommandPool commandPool;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    // map好的point, cpu直接往这个指针里写就可以了.
    std::vector<void*> uniformBuffersMapped;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    bool framebufferResized = false;

    void initWindow() {
#if !defined(__ANDROID__)
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
#endif
    }

#if !defined(__ANDROID__)
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        HelloTriangleApplication* app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
#endif

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        // 先查询选中 GPU 的 feature bits, 再创建 device 并加载已 enable 的 EXT commands.
        queryDynamicStateSupport();
        createLogicalDevice();
        loadDynamicStateFunctions();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createDepthResources();
        createFramebuffers();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
#if defined(__ANDROID__)
        createReadbackBuffer();
#endif
        createSyncObjects();
    }

    void mainLoop() {
#if defined(__ANDROID__)
        for (uint32_t frame = 0; frameCount == 0 || frame < frameCount; frame++) {
            drawFrame();
        }
#else
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
#endif

        if (vkDeviceWaitIdle(device) != VK_SUCCESS) {
            throw std::runtime_error("failed to wait for device idle!");
        }

#if defined(__ANDROID__)
        saveImage(outputFile);
#endif
    }

    void cleanupSwapChain() {
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for (VkFramebuffer framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (VkImageView imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

#if defined(__ANDROID__)
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorImageMemory, nullptr);
#else
        vkDestroySwapchainKHR(device, swapChain, nullptr);
        for (VkSemaphore semaphore : renderFinishedSemaphores) {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        renderFinishedSemaphores.clear();
#endif
    }

    void cleanup() {
        cleanupSwapChain();
#if defined(__ANDROID__)
        vkDestroyBuffer(device, readbackBuffer, nullptr);
        vkFreeMemory(device, readbackBufferMemory, nullptr);
#endif

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);

        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);

        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);

        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
#if !defined(__ANDROID__)
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
#endif
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

#if !defined(__ANDROID__)
        vkDestroySurfaceKHR(instance, surface, nullptr);
#endif
        vkDestroyInstance(instance, nullptr);

#if !defined(__ANDROID__)
        glfwDestroyWindow(window);
        glfwTerminate();
#endif
    }

#if !defined(__ANDROID__)
    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
        // 将这 renderFinishedSemaphores 放到 swapchain 的生命周期中管理
        createRenderFinishedSemaphores();
    }
#endif

    void createInstance() {
#if defined(__ANDROID__) && !defined(NDEBUG)
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            std::cerr << "validation layers unavailable, continuing without validation." << std::endl;
            enableValidationLayers = false;
        }
#else
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }
#endif

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        // 本章直接使用 Vulkan 1.3 core 的 EDS1/EDS2 commands, device 筛选也检查此版本.
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        std::vector<const char*> extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // 如果开启了validation layers, 则需要添加validation layers.
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            // 设置debug messenger的回调函数.
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
#if defined(__ANDROID__)
        // offscreen, 没有surface.
#else
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
#endif
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const VkPhysicalDevice& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("no suitable Vulkan 1.3 GPU with samplerAnisotropy found!");
        }
    }

    void queryDynamicStateSupport() {
        // EDS3 有多个 feature bits: 局部变量接收完整支持情况, 成员只保留本章需要 enable 的 5 项.
        VkPhysicalDeviceExtendedDynamicState3FeaturesEXT supportedEds3{};
        supportedEds3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
        // 这两个 extension 各只有一个 feature bit, 本章都要用, 因此直接复用查询结果来 enable.
        vertexInputFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;
        colorWriteFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT;
        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        // 只把选中 GPU 声明支持的 extension structs 接入 query chain.
        // next 指向当前末尾的 pNext, 每加入一个 struct 就向后移动, 未加入的 feature bits 保持 0.
        void** next = &features.pNext;
        if (dynamicStateExtensions.extendedDynamicState3) {
            *next = &supportedEds3;
            next = &supportedEds3.pNext;
        }
        if (dynamicStateExtensions.vertexInputDynamicState) {
            *next = &vertexInputFeatures;
            next = &vertexInputFeatures.pNext;
        }
        if (dynamicStateExtensions.colorWriteEnable) {
            *next = &colorWriteFeatures;
        }
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
        depthBoundsSupported = features.features.depthBounds;

        // 从 supportedEds3 挑选需要的 bits, 避免把支持但未使用的 EDS3 features 一起 enable.
        eds3Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
        eds3Features.extendedDynamicState3DepthClampEnable = supportedEds3.extendedDynamicState3DepthClampEnable;
        eds3Features.extendedDynamicState3PolygonMode = supportedEds3.extendedDynamicState3PolygonMode;
        eds3Features.extendedDynamicState3ColorBlendEnable = supportedEds3.extendedDynamicState3ColorBlendEnable;
        eds3Features.extendedDynamicState3ColorBlendEquation = supportedEds3.extendedDynamicState3ColorBlendEquation;
        eds3Features.extendedDynamicState3ColorWriteMask = supportedEds3.extendedDynamicState3ColorWriteMask;
        // 查询用的 pNext 连接先清掉, createLogicalDevice 会重新构建用于 enable 的 chain.
        vertexInputFeatures.pNext = nullptr;
        colorWriteFeatures.pNext = nullptr;

        // required extensions 沿用 27.2, optional extensions 根据支持情况追加.
        // Extension name 和 feature bit 分别通过 ppEnabledExtensionNames 和 pNext 交给 vkCreateDevice.
        enabledDeviceExtensions = deviceExtensions;
        if (dynamicStateExtensions.extendedDynamicState3) {
            enabledDeviceExtensions.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
        }
        if (vertexInputFeatures.vertexInputDynamicState) {
            enabledDeviceExtensions.push_back(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
        }
        if (colorWriteFeatures.colorWriteEnable) {
            enabledDeviceExtensions.push_back(VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME);
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        std::cout << "GPU: " << properties.deviceName << ", Vulkan "
                  << VK_API_VERSION_MAJOR(properties.apiVersion) << '.'
                  << VK_API_VERSION_MINOR(properties.apiVersion) << '\n';
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
#if defined(__ANDROID__)
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value()};
#else
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
#endif

        // 创建的时候指定创建q的个数, 还有从哪个queueFamilyIndex里面创建
        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        deviceFeatures.depthBounds = depthBoundsSupported;

        // 本章用到的 EDS1/EDS2 commands 已进入 Vulkan 1.3 core, 不需要 enable 对应 EXT 名称.
        // Dynamic state 和 dynamic rendering 独立, 这里仍使用传统 render pass.
        // 查询只获得支持情况, 这里把需要的 feature structs 接到 device create info 才真正 enable.
        void* featureChain = nullptr;
        if (dynamicStateExtensions.extendedDynamicState3) {
            eds3Features.pNext = featureChain;
            featureChain = &eds3Features;
        }
        if (vertexInputFeatures.vertexInputDynamicState) {
            vertexInputFeatures.pNext = featureChain;
            featureChain = &vertexInputFeatures;
        }
        if (colorWriteFeatures.colorWriteEnable) {
            colorWriteFeatures.pNext = featureChain;
            featureChain = &colorWriteFeatures;
        }

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = featureChain;

        // 创建logical dev需要queueCI信息
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledDeviceExtensions.size());
        createInfo.ppEnabledExtensionNames = enabledDeviceExtensions.data();

        // Validation layers are enabled on the instance, not the device.
        createInfo.enabledLayerCount = 0;

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        // dev创建完了, 再从里面得到queue
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
#if !defined(__ANDROID__)
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
#endif
    }

#if defined(__ANDROID__)
    // Reuse the framebuffer/image-view path with one owned offscreen image.
    void createSwapChain() {
        swapChainImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
        swapChainExtent = {WIDTH, HEIGHT};
        createImage(WIDTH, HEIGHT, swapChainImageFormat, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
        swapChainImages = {colorImage};
    }

    void createReadbackBuffer() {
        VkDeviceSize size = static_cast<VkDeviceSize>(swapChainExtent.width) * swapChainExtent.height * 4;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            readbackBuffer, readbackBufferMemory);
    }

    void saveImage(const char* filename) {
        uint32_t width = swapChainExtent.width;
        uint32_t height = swapChainExtent.height;

        void* data = nullptr;
        if (vkMapMemory(device, readbackBufferMemory, 0, VK_WHOLE_SIZE, 0, &data) != VK_SUCCESS) {
            throw std::runtime_error("failed to map readback memory!");
        }

        // readbackBuffer是tightly packed, row stride = width * 4.
        int written = stbi_write_png(filename, static_cast<int>(width), static_cast<int>(height), 4, data, static_cast<int>(width * 4));

        vkUnmapMemory(device, readbackBufferMemory);

        if (written == 0) {
            throw std::runtime_error("failed to write output image file!");
        }

        std::cout << "saved " << width << "x" << height << " image to " << filename << std::endl;
    }
#else
    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        // 最小2张, 推荐3张.
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        // 这个有点意思, 一般gfx queue family同时支持present, 如果不是同一个QF, 还要指定sharingMode.
        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }
#endif

    // 这里创建的imageView就是后面创建FB的attachments
    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
#if defined(__ANDROID__)
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
#else
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        // 每个render pass开始时把深度清为clearValues中指定的值(本例为最远的1.0).
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // 深度只用于本帧的遮挡判断, render pass结束后无需保留其内容.
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        // 不关心进入render pass之前的旧内容; render pass会自动完成所需的layout转换.
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        // attachment=1对应下面attachments数组中的depthAttachment.
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        // 将深度附件接入subpass; 只创建depth image并不会自动启用深度测试.
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        // 让外部操作与本subpass中的color输出, early/late depth test之间建立内存依赖,
        // 确保attachment在被本subpass读写前处于可安全访问的状态.
        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

#if defined(__ANDROID__)
        // Make color writes and the final layout transition visible to the copy.
        VkSubpassDependency dependencies[2] = {dependency, {}};
        dependencies[1].srcSubpass = 0;
        dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
#endif

        // 有color和depth
        std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
#if defined(__ANDROID__)
        renderPassInfo.dependencyCount = 2;
        renderPassInfo.pDependencies = dependencies;
#else
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
#endif

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    // 这个DS layout描述当前这个DS(descriptor set)一共有几个binding point, 每个point下有几个descriptor
    // 注意这里并不是shader用的resource (ds), 而是表述ds的布局layout, 类似于metadata
    // 虽然每个frame有自己的DS, 但是这个DS layout是一样的
    void createDescriptorSetLayout() {
        // 有2个binding point, 一个ubo, 一个sampler.

        // point 0是ubo, 只有一个descriptor
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        // point 1是sampler, 只有一个descriptor
        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    // Device 创建后, 按已 enable 的 feature 加载 EXT entry points.
    // 后面的 pipeline 声明和 command 调用使用相同的 feature 条件, 避免调用未启用的功能.
    void loadDynamicStateFunctions() {
        if (vertexInputFeatures.vertexInputDynamicState) {
            pfnCmdSetVertexInputEXT = reinterpret_cast<PFN_vkCmdSetVertexInputEXT>(vkGetDeviceProcAddr(device, "vkCmdSetVertexInputEXT"));
            if (pfnCmdSetVertexInputEXT == nullptr) {
                throw std::runtime_error("enabled feature is missing vkCmdSetVertexInputEXT!");
            }
        }
        if (colorWriteFeatures.colorWriteEnable) {
            pfnCmdSetColorWriteEnableEXT = reinterpret_cast<PFN_vkCmdSetColorWriteEnableEXT>(vkGetDeviceProcAddr(device, "vkCmdSetColorWriteEnableEXT"));
            if (pfnCmdSetColorWriteEnableEXT == nullptr) {
                throw std::runtime_error("enabled feature is missing vkCmdSetColorWriteEnableEXT!");
            }
        }
        if (eds3Features.extendedDynamicState3DepthClampEnable) {
            pfnCmdSetDepthClampEnableEXT = reinterpret_cast<PFN_vkCmdSetDepthClampEnableEXT>(vkGetDeviceProcAddr(device, "vkCmdSetDepthClampEnableEXT"));
            if (pfnCmdSetDepthClampEnableEXT == nullptr) {
                throw std::runtime_error("enabled feature is missing vkCmdSetDepthClampEnableEXT!");
            }
        }
        if (eds3Features.extendedDynamicState3PolygonMode) {
            pfnCmdSetPolygonModeEXT = reinterpret_cast<PFN_vkCmdSetPolygonModeEXT>(vkGetDeviceProcAddr(device, "vkCmdSetPolygonModeEXT"));
            if (pfnCmdSetPolygonModeEXT == nullptr) {
                throw std::runtime_error("enabled feature is missing vkCmdSetPolygonModeEXT!");
            }
        }
        if (eds3Features.extendedDynamicState3ColorBlendEnable) {
            pfnCmdSetColorBlendEnableEXT = reinterpret_cast<PFN_vkCmdSetColorBlendEnableEXT>(vkGetDeviceProcAddr(device, "vkCmdSetColorBlendEnableEXT"));
            if (pfnCmdSetColorBlendEnableEXT == nullptr) {
                throw std::runtime_error("enabled feature is missing vkCmdSetColorBlendEnableEXT!");
            }
        }
        if (eds3Features.extendedDynamicState3ColorBlendEquation) {
            pfnCmdSetColorBlendEquationEXT = reinterpret_cast<PFN_vkCmdSetColorBlendEquationEXT>(vkGetDeviceProcAddr(device, "vkCmdSetColorBlendEquationEXT"));
            if (pfnCmdSetColorBlendEquationEXT == nullptr) {
                throw std::runtime_error("enabled feature is missing vkCmdSetColorBlendEquationEXT!");
            }
        }
        if (eds3Features.extendedDynamicState3ColorWriteMask) {
            pfnCmdSetColorWriteMaskEXT = reinterpret_cast<PFN_vkCmdSetColorWriteMaskEXT>(vkGetDeviceProcAddr(device, "vkCmdSetColorWriteMaskEXT"));
            if (pfnCmdSetColorWriteMaskEXT == nullptr) {
                throw std::runtime_error("enabled feature is missing vkCmdSetColorWriteMaskEXT!");
            }
        }
    }

    void createGraphicsPipeline() {
        std::vector<char> vertShaderCode = readFile("shaders/vert.spv");
        std::vector<char> fragShaderCode = readFile("shaders/frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkVertexInputBindingDescription bindingDescription = Vertex::getBindingDescription();
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = Vertex::getAttributeDescriptions();

        // 保留 27.2 的描述作为 static fallback; enable VERTEX_INPUT_EXT 时改由 command 提供.
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        // 27.2 只动态设置 viewport/scissor 的值, count 固定为 1.
        // WITH_COUNT 连 count 也由 command 提供, 所以这里两个 count 都设为 0.
        viewportState.viewportCount = 0;
        viewportState.scissorCount = 0;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_TRUE;
        // 声明为 dynamic 的字段不会初始化 command buffer state, 实际值在 setDynamicStates 中设置.

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        // 开启深度比较, 并让通过测试的片元把新深度写回depth attachment.
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        // Vulkan的标准深度范围是[0, 1]; LESS表示深度值更小(更靠近相机)的片元通过.
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        // 使用 CONSTANT_COLOR/ALPHA 并在 draw 前将 blend constants 设为 1, 得到 src * 1 + dst * 0.
        // 这样能实际使用 BLEND_CONSTANTS, 同时保持 27.2 的输出; 缺少对应 EDS3 bit 时沿用这些 static 值.
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 1.0f;
        colorBlending.blendConstants[1] = 1.0f;
        colorBlending.blendConstants[2] = 1.0f;
        colorBlending.blendConstants[3] = 1.0f;

        // 本章要求 Vulkan 1.3, 这 15 项固定声明为 dynamic.
        // 这里声明哪些 state 由 command 设置; 具体值见 setDynamicStates, 两处需要对应.
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT,
            VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT,
            VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE,
            VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
            VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
            VK_DYNAMIC_STATE_CULL_MODE,
            VK_DYNAMIC_STATE_FRONT_FACE,
            VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
            VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
            VK_DYNAMIC_STATE_DEPTH_BIAS,
            VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
            VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
            VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
            VK_DYNAMIC_STATE_BLEND_CONSTANTS
        };
        std::cout << "Dynamic states for the single indexed draw:\n";
        const char* const coreStateNames[] = {
            "VIEWPORT_WITH_COUNT",
            "SCISSOR_WITH_COUNT",
            "VERTEX_INPUT_BINDING_STRIDE",
            "PRIMITIVE_TOPOLOGY",
            "PRIMITIVE_RESTART_ENABLE",
            "CULL_MODE",
            "FRONT_FACE",
            "RASTERIZER_DISCARD_ENABLE",
            "DEPTH_BIAS_ENABLE",
            "DEPTH_BIAS",
            "DEPTH_TEST_ENABLE",
            "DEPTH_WRITE_ENABLE",
            "DEPTH_COMPARE_OP",
            "DEPTH_BOUNDS_TEST_ENABLE",
            "BLEND_CONSTANTS"
        };
        for (const char* stateName : coreStateNames) {
            std::cout << "  " << stateName << " [core]\n";
        }
        // Optional 项逐个按 feature bit 加入; 未加入的项继续使用 pipeline 中的 static 设置.
        if (depthBoundsSupported) {
            dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
            std::cout << "  DEPTH_BOUNDS [enabled]\n";
        } else {
            std::cout << "  DEPTH_BOUNDS [unsupported, static fallback]\n";
        }
        if (vertexInputFeatures.vertexInputDynamicState) {
            dynamicStates.push_back(VK_DYNAMIC_STATE_VERTEX_INPUT_EXT);
            std::cout << "  VERTEX_INPUT_EXT [enabled]\n";
        } else {
            std::cout << "  VERTEX_INPUT_EXT [unsupported, static fallback]\n";
        }
        if (colorWriteFeatures.colorWriteEnable) {
            dynamicStates.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_ENABLE_EXT);
            std::cout << "  COLOR_WRITE_ENABLE_EXT [enabled]\n";
        } else {
            std::cout << "  COLOR_WRITE_ENABLE_EXT [unsupported, static fallback]\n";
        }
        if (eds3Features.extendedDynamicState3DepthClampEnable) {
            dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT);
            std::cout << "  DEPTH_CLAMP_ENABLE_EXT [enabled]\n";
        } else {
            std::cout << "  DEPTH_CLAMP_ENABLE_EXT [unsupported, static fallback]\n";
        }
        if (eds3Features.extendedDynamicState3PolygonMode) {
            dynamicStates.push_back(VK_DYNAMIC_STATE_POLYGON_MODE_EXT);
            std::cout << "  POLYGON_MODE_EXT [enabled]\n";
        } else {
            std::cout << "  POLYGON_MODE_EXT [unsupported, static fallback]\n";
        }
        if (eds3Features.extendedDynamicState3ColorBlendEnable) {
            dynamicStates.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT);
            std::cout << "  COLOR_BLEND_ENABLE_EXT [enabled]\n";
        } else {
            std::cout << "  COLOR_BLEND_ENABLE_EXT [unsupported, static fallback]\n";
        }
        if (eds3Features.extendedDynamicState3ColorBlendEquation) {
            dynamicStates.push_back(VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT);
            std::cout << "  COLOR_BLEND_EQUATION_EXT [enabled]\n";
        } else {
            std::cout << "  COLOR_BLEND_EQUATION_EXT [unsupported, static fallback]\n";
        }
        if (eds3Features.extendedDynamicState3ColorWriteMask) {
            dynamicStates.push_back(VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT);
            std::cout << "  COLOR_WRITE_MASK_EXT [enabled]\n";
        } else {
            std::cout << "  COLOR_WRITE_MASK_EXT [unsupported, static fallback]\n";
        }
        std::cout << "Total: " << dynamicStates.size() << " dynamic states, 1 indexed draw/frame" << std::endl;
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        // pipeline layout里面要知道descriptor set layout.
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());

        // swap chain是3 buffer, 里面只有color, depth是app自己创建传进FB的attachment的
        // 这里的顺序必须匹配render pass中的attachment索引: 0是color, 1是depth.
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 2> attachments = {
                swapChainImageViews[i],
                depthImageView
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics command pool!");
        }
    }

    // depth是gpu在走pipeline的时候生成的, 不是cpu传过去的, 没有像tex一样用staging buffer.
    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();

        // depth image的尺寸必须与swapchain一致, 所以recreateSwapChain()时也要重新创建.
        // 本例不手动transition layout: render pass会依据attachment的initial/final layout自动转换.
        // depth image是在gpu mem里面的DEVICE_LOCAL
        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // 按候选顺序查找同时满足指定图像布局方式和功能要求的格式.
    // candidates: 待检测的格式列表; tiling: 图像数据的排列方式; features: 必须支持的格式特性.
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
        for (VkFormat format : candidates) {
            VkFormatProperties props;
            // 不同物理设备对同一格式的支持可能不同, 因此需要查询当前 GPU 的格式能力.
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            // features 是位掩码. 按位与后仍等于 features, 表示请求的所有特性均被支持.
            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        // 所有候选格式都不符合要求, 无法安全地创建对应图像资源.
        throw std::runtime_error("failed to find supported format!");
    }

    // 按优先级选择深度格式: 优先纯32位浮点深度, 其次选择带8位stencil的格式.
    // 返回的格式还必须支持optimal tiling, 并能作为depth/stencil attachment使用.
    VkFormat findDepthFormat() {
        return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        // tex是cpu传到gpu用的, 也是用staging buffer.
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
            memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);

        stbi_image_free(pixels);

        // tex也是gpu mem, DEVICE_LOCAL
        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void createTextureSampler() {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        // subresourceRange 是用于选择mipmap levels和array layers.
        // aspectMask 用于选择颜色附件还是深度附件.
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image view!");
        }

        return imageView;
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {
            width,
            height,
            1
        };

        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }

    // Vulkan 中一个非常经典的 两阶段拷贝(two-stage copy) 模式
    // 第一次 copy: 从 vertices 拷贝到 CPU 可见的 staging buffer(memcpy).
    // 第二次 copy: 从 staging buffer 通过 GPU command(vkCmdCopyBuffer)拷贝到真正的 vertex buffer(位于 GPU local memory).
    //     为什么第二次从staging buffer, 而不是从vertices copy?
    //     GPU 无法直接访问 App 的 malloc/new 指针(即 CPU 普通内存), 因为这些内存并不处于 GPU 可访问的设备映射区域中
    //     是你用 std::vector, new 或 malloc 分配出来的一段用户空间的虚拟地址, 是属于当前进程的 私有 CPU 内存空间. GPU 是无法看到这块内存的
    //     所有 GPU 可访问的 buffer / memory, 必须是你通过 Vulkan 显式创建 + 绑定的:
    //     vkAllocateMemory(...) -> 返回 VkDeviceMemory + vkBindBufferMemory(buffer, memory, offset);
    // 在 OpenGL 中确实也发生了类似的"两次拷贝"行为, 只不过这些操作被 OpenGL 驱动自动处理, 隐藏了起来, 开发者看不到而已
    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        // 创建一个临时的buf, HOST_VISIBLE | HOST_COHERENT, 用于拷贝把vertices copy到这里
        // CPU 可以 vkMapMemory, 用 memcpy 写入;
        // 性能较低, 不适合频繁用于渲染
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // vertexBuffer 是 DEVICE_LOCAL 的内存 ,, 这是 GPU 本地的显存, 性能最好, 适合频繁渲染时使用
        // CPU 无法直接访问 DEVICE_LOCAL 内存(不能 vkMapMemory)
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        // 然后用gpu cmd copy到VBO上.
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // 和上面一样, 只是拷贝的是 indices.
    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
            memcpy(data, indices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    // 做动态更新的小数据(如 uniform buffer), 用 HOST_VISIBLE | HOST_COHERENT 的缓冲区直接写.
    // 但像 VBO 这样的大块静态顶点数据, 为了渲染效率, 必须走这套 staging + device copy 机制.
    // UBO 的典型特征:
    //      数据量通常较小(几个 bytes 到几 KB);
    //      更新频率较高(如每帧更新 camera matrix, 灯光参数等);
    //      GPU 使用的是只读方式;
    //      不要求最高性能(不像 VBO 那种批量访问);
    //      写完后马上要被 GPU 用.
    void createUniformBuffers() {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            // HOST_VISIBLE_BIT | HOST_COHERENT 属性的 buffer, 写入后不需要显式 vkFlushMappedMemoryRanges(), GPU 可以立刻看到更新(coherent 的定义)
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        // 2 ubo decriptors, 每个frame一个
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        // 2 sampler decriptors, 每个frame一个
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        // 这个pool总共可以分配 2 个 VkDescriptorSet(每帧一个)
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        // 只创建了一个descriptor pool, 这个pool里面有两种类型的 descriptor (uniform buffer 和 texture image)
        // 里面有4个descriptors.
        // 注意这个是descriptor的pool, 不是descriptor set的pool.
        // 从pool里面分配descriptor后, 再组装成descriptor set 来使用它. 每个frame要用的就是DS: vkCmdBindDescriptorSets
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }

    void createDescriptorSets() {
        // 两个一样的DS layout, 每个frame一个.
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        // 从descriptor pool里面分配 descriptor.
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        // 给descriptor set分配空间, descriptor是从pool里面分配的, 但是descriptor set还是要空间的.
        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        // 每个frame一个descriptor set.
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            // 这个descriptor是用来做sampler的, 要知道sampler信息.
            imageInfo.sampler = textureSampler;

            // 用两个VkWriteDescriptorSet创建(更新)两个DS, 即把资源写进descriptor set.
            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    // VkMemoryPropertyFlags指定了是怎样的mem, 在CPU端访问还是在GPU端访问.
    // DEVICE_LOCAL, gpu访问最高效, 就是gpu mem
    // HOST_VISIBLE, 在cpu端的cpu mem, 映射给 GPU 用的系统内存, 性能远低于 GPU mem, 两边都能看到
    //     这并不意味着 GPU 一定能"马上"看到你写的数据!因为还没有保证 cache 的一致性.
    //     必须手动调用: vkFlushMappedMemoryRanges(); 这是 Vulkan 的要求, 用于保证:
    //     CPU 写入的内容刷新到了内存中, 确保 GPU 能看到最新数据(就像你 flush cache 一样).
    //     反过来, 如果 GPU 写了数据, 你要让 CPU 看到, 也要调用: vkInvalidateMappedMemoryRanges();
    //     单单HOST_VISIBLE就是还在写的这边的cache里面, 要手动flash到mem.
    // HOST_VISIBLE | HOST_COHERENT, CPU 写入后, GPU 自动能看到更新(你不需要手动 flush)
    // 用于更新数据, 不需要显式 vkFlushMappedMemoryRanges
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    // 返回一个一次性的cmdbuf, 用于单次提交.
    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    // 每次录制 command buffer, 在 bind pipeline 后设置本次 draw 需要的 dynamic state.
    // Pipeline create info 中的 static 值不会成为 dynamic state 的默认值, 不能省略对应 command.
    void setDynamicStates(VkCommandBuffer commandBuffer) {
        VkViewport viewport{};
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewportWithCount(commandBuffer, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = swapChainExtent;
        vkCmdSetScissorWithCount(commandBuffer, 1, &scissor);

        // 复用 27.2 的 Vertex layout, 转成 EXT command 使用的 Description2EXT, 在录制时提供.
        if (vertexInputFeatures.vertexInputDynamicState) {
            VkVertexInputBindingDescription binding = Vertex::getBindingDescription();
            VkVertexInputBindingDescription2EXT dynamicBinding{};
            dynamicBinding.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT;
            dynamicBinding.binding = binding.binding;
            dynamicBinding.stride = binding.stride;
            dynamicBinding.inputRate = binding.inputRate;
            dynamicBinding.divisor = 1;

            std::array<VkVertexInputAttributeDescription, 3> attributes = Vertex::getAttributeDescriptions();
            std::array<VkVertexInputAttributeDescription2EXT, 3> dynamicAttributes{};
            for (size_t i = 0; i < attributes.size(); i++) {
                dynamicAttributes[i].sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT;
                dynamicAttributes[i].location = attributes[i].location;
                dynamicAttributes[i].binding = attributes[i].binding;
                dynamicAttributes[i].format = attributes[i].format;
                dynamicAttributes[i].offset = attributes[i].offset;
            }
            pfnCmdSetVertexInputEXT(commandBuffer, 1, &dynamicBinding,
                static_cast<uint32_t>(dynamicAttributes.size()), dynamicAttributes.data());
        }

        VkDeviceSize offset = 0;
        VkDeviceSize size = sizeof(vertices[0]) * vertices.size();
        VkDeviceSize stride = sizeof(Vertex);
        // vkCmdSetVertexInputEXT 和 vkCmdBindVertexBuffers2 都能设置 stride, 后设置的值生效.
        // 即使没有 EXT_vertex_input_dynamic_state, attributes 仍为 static, stride 也可以单独 dynamic.
        vkCmdBindVertexBuffers2(commandBuffer, 0, 1, &vertexBuffer, &offset, &size, &stride);
        vkCmdSetPrimitiveTopology(commandBuffer, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        // 本例 TRIANGLE_LIST 不使用 restart indices, 所以显式关闭 primitive restart.
        vkCmdSetPrimitiveRestartEnable(commandBuffer, VK_FALSE);
        vkCmdSetCullMode(commandBuffer, VK_CULL_MODE_BACK_BIT);
        vkCmdSetFrontFace(commandBuffer, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        vkCmdSetRasterizerDiscardEnable(commandBuffer, VK_FALSE);
        // Enable 和 factors 是两个独立的 dynamic state; factors 全为 0, 保持原来的 depth 结果.
        vkCmdSetDepthBiasEnable(commandBuffer, VK_TRUE);
        vkCmdSetDepthBias(commandBuffer, 0.0f, 0.0f, 0.0f);
        vkCmdSetDepthTestEnable(commandBuffer, VK_TRUE);
        vkCmdSetDepthWriteEnable(commandBuffer, VK_TRUE);
        vkCmdSetDepthCompareOp(commandBuffer, VK_COMPARE_OP_LESS);
        // 动态设置 test enable 不等于支持 depthBounds feature; 不支持时只能把 test 关闭.
        vkCmdSetDepthBoundsTestEnable(commandBuffer, depthBoundsSupported);
        if (depthBoundsSupported) {
            vkCmdSetDepthBounds(commandBuffer, 0.0f, 1.0f);
        }

        const float blendConstants[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        vkCmdSetBlendConstants(commandBuffer, blendConstants);
        if (eds3Features.extendedDynamicState3DepthClampEnable) {
            // 动态设置 enable 的能力与 core depthClamp feature 独立, 此处设为 FALSE 不需要后者.
            pfnCmdSetDepthClampEnableEXT(commandBuffer, VK_FALSE);
        }
        if (eds3Features.extendedDynamicState3PolygonMode) {
            // 本例保持 FILL, 不需要额外 enable fillModeNonSolid.
            pfnCmdSetPolygonModeEXT(commandBuffer, VK_POLYGON_MODE_FILL);
        }
        if (eds3Features.extendedDynamicState3ColorBlendEnable) {
            VkBool32 blendEnable = VK_TRUE;
            pfnCmdSetColorBlendEnableEXT(commandBuffer, 0, 1, &blendEnable);
        }
        if (eds3Features.extendedDynamicState3ColorBlendEquation) {
            VkColorBlendEquationEXT equation{};
            equation.srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR;
            equation.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            equation.colorBlendOp = VK_BLEND_OP_ADD;
            equation.srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA;
            equation.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            equation.alphaBlendOp = VK_BLEND_OP_ADD;
            pfnCmdSetColorBlendEquationEXT(commandBuffer, 0, 1, &equation);
        }
        if (eds3Features.extendedDynamicState3ColorWriteMask) {
            VkColorComponentFlags mask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            pfnCmdSetColorWriteMaskEXT(commandBuffer, 0, 1, &mask);
        }
        // Color write enable 控制整个 attachment 的写入, color write mask 则选择 RGBA 分量.
        if (colorWriteFeatures.colorWriteEnable) {
            VkBool32 writeEnable = VK_TRUE;
            pfnCmdSetColorWriteEnableEXT(commandBuffer, 1, &writeEnable);
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        // clear value的顺序与render pass的attachment顺序一致: 先color, 再depth/stencil.
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        // depth=1.0表示最远处; stencil=0(本例没有开启stencil test).
        clearValues[1].depthStencil = {1.0f, 0};

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            // 集中设置新增的 dynamic state, 也包含 27.2 原有的 viewport/scissor 和 vertex buffer binding.
            setDynamicStates(commandBuffer);

            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

            vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

#if defined(__ANDROID__)
        // colorImage已经是TRANSFER_SRC_OPTIMAL了(renderpass finalLayout), 整张copy到readbackBuffer.
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {swapChainExtent.width, swapChainExtent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, swapChainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer, 1, &region);

        // Transfer writes -> host reads. HOST_COHERENT avoids invalidation; wait for completion before mapping.
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = readbackBuffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
#endif

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

#if !defined(__ANDROID__)
    void createRenderFinishedSemaphores() {
        // Present waits on these: reuse only after acquiring the same image.
        renderFinishedSemaphores.resize(swapChainImages.size());
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (VkSemaphore& semaphore : renderFinishedSemaphores) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
                throw std::runtime_error("failed to create render-finished semaphore!");
            }
        }
    }
#endif

#if defined(__ANDROID__)
    void createSyncObjects() {
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[0]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create offscreen fence!");
        }
    }
#else
    void createSyncObjects() {
        createRenderFinishedSemaphores();
        // 每一个in-flight frame有一个acquire semaphore和一个submission fence.
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }
#endif

    // runtime的时候每个frame更新.
    void updateUniformBuffer(uint32_t currentImage) {
        static std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

        std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
#if defined(__ANDROID__)
        // Fixed timestep makes screenshots reproducible across drivers and chapters.
        static uint64_t frameNumber = 0;
        time = static_cast<float>(frameNumber++) / 60.0f;
#endif

        // 每一帧都把其对应的MVP写到对应的ubo里面.
        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

#if defined(__ANDROID__)
    void drawFrame() {
        // One frame slot protects the shared color/depth images, UBO and readback buffer.
        if (vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            throw std::runtime_error("failed to wait for offscreen fence!");
        }
        updateUniformBuffer(currentFrame);
        if (vkResetCommandBuffer(commandBuffers[currentFrame], 0) != VK_SUCCESS) {
            throw std::runtime_error("failed to reset offscreen command buffer!");
        }
        recordCommandBuffer(commandBuffers[currentFrame], 0);
        if (vkResetFences(device, 1, &inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to reset offscreen fence!");
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit offscreen frame!");
        }
    }
#else
    void drawFrame() {
        // swapchain里面3个images, 但是MAX_FRAMES_IN_FLIGHT是2 (2个fence), 这里是资源的复用(资源没有image多).
        // 只有两套资源, 不能发frame 2, 直到frame 0的 GPU 执行完毕
        // frame 0: image 0 + wait(fence 0) + vkQueueSubmit(cmd, fence 0) // fence没用过不用等
        // frame 1: image 1 + wait(fence 1) + vkQueueSubmit(cmd, fence 1) // fence没用过不用等
        // frame 2: image 2 + wait(fence 0) + vkQueueSubmit(cmd, fence 0) // cpu等frame 0的fence 0, gpu做完了再画.
        // frame 3: image 0 + wait(fence 1) + vkQueueSubmit(cmd, fence 1) // cpu等frame 1的fence 1, gpu做完了再画.
        // frame 4: image 1 + wait(fence 0) + vkQueueSubmit(cmd, fence 0) // cpu等frame 2的fence 0, gpu做完了再画.
        // frame 5: image 2 + wait(fence 1) + vkQueueSubmit(cmd, fence 1) // cpu等frame 3的fence 1, gpu做完了再画.
        //
        // 如果是3个images和3个fence:
        // frame 0: image 0 + wait(fence 0) + vkQueueSubmit(cmd, fence 0) // fence没用过不用等
        // frame 1: image 1 + wait(fence 1) + vkQueueSubmit(cmd, fence 1) // fence没用过不用等
        // frame 2: image 2 + wait(fence 2) + vkQueueSubmit(cmd, fence 2) // fence没用过不用等
        // frame 3: image 0 + wait(fence 0) + vkQueueSubmit(cmd, fence 0) // cpu等frame 0的fence, gpu做完了再画.
        // frame 4: image 1 + wait(fence 1) + vkQueueSubmit(cmd, fence 1) // cpu等frame 1的fence, gpu做完了再画.
        // frame 5: image 2 + wait(fence 2) + vkQueueSubmit(cmd, fence 2) // cpu等frame 2的fence, gpu做完了再画.
        //
        // 总结: swapchain image = minCount + 1 = 3, 是推荐的做法.
        // 如果gpu太慢, cpu一直等gpu, 如果gpu太快, 等cpu, 这两种情况上面fence数量2/3没有太大区别.
        // GPU 速度"刚好卡在 2 与 3 之间", GPU 工作量中等偏上, 不慢, 但也不快. 假如 CPU 提交太慢(因为等 fence), 会造成 GPU 有短暂 idle.
        // 而如果你把 MAX_FRAMES_IN_FLIGHT=3, CPU 不等 fence, 能及时把第三帧提交给 GPU, 让 GPU 连续运行, 不 idle. perf会有小幅提升(e.g. 90% -> 98%/100%).
        // 还有一个情况是gpu workload不稳定, 忽大忽小, 应该还是有提升的.
        //
        // 问题2, fence也是3, 和image一一对应, 还需要cpu等待吗?
        // 要的, 否则cpu可能无限提交, driver就crash了.
        // 可以在vkAcquireNextImageKHR后得到imageIndex, 等待特定的imageIndex[0, 1, 2], 而不是currentFrame[0, 1, 2], 返回的imageIndex不一定是012顺序.
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // 简单打印globalFrameId for debugging, 不考虑resize window或者error这些情况.
        static uint64_t globalFrameId = 0;
        std::cout << "drawFrame(), globalFrameId:" << globalFrameId << ", imageIndex(=image slot): " << imageIndex << ", currentFrame(=in flight slot): " << currentFrame << std::endl;
        updateUniformBuffer(currentFrame);

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
        // 这里每帧都要record, 事实上可以在init的时候record完, 这里提交就可以.
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        // 当前要提交的cmd要等这个semaphore, 即只有这个swap chain image available了, 才能在上面画图.
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        // 当前要提交的cmd被gpu执行了会signal这个semaphore, 告诉其他gpu cmd, 这个cmd已经执行完了,
        // 你可以开始你的cmd了(如果你的cmd pending在我这里的话). 这个例子里面后面的QueuePresent在等这个semaphore.
        // fix: 跟image index走, root cause上面分析了.
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;

        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        globalFrameId++;
    }
#endif

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

#if !defined(__ANDROID__)
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const VkSurfaceFormatKHR& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }
#endif

#if !defined(__ANDROID__)
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const VkPresentModeKHR& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }
#endif

#if !defined(__ANDROID__)
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }
#endif

#if !defined(__ANDROID__)
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        // 得到capabilities, 里面包含了swapchain最小/最多需要的image cnt等信息(e.g. double-buffer).
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        // 看surface支持的格式, 一般都是RGBA8
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        // 看present mode有哪些. FIFO: 就是一个queue数据结构, driver必须支持的. MODE_IMMEDIATE, 内部没有queue, 马上显示, 可能看见tearing. 还有其他mode
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }
#endif

    bool isDeviceSuitable(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_3) {
            return false;
        }

        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

#if defined(__ANDROID__)
        bool swapChainAdequate = true;
#else
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

#endif
        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        // 100多个, 不打印了, 用vulkaninfo看.

        for (const VkExtensionProperties& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
            // 下面三个是 optional extensions, 只记录支持情况, 不影响 requiredExtensions 的检查结果.
            if (strcmp(extension.extensionName, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME) == 0) {
                dynamicStateExtensions.extendedDynamicState3 = true;
            } else if (strcmp(extension.extensionName, VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME) == 0) {
                dynamicStateExtensions.vertexInputDynamicState = true;
            } else if (strcmp(extension.extensionName, VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME) == 0) {
                dynamicStateExtensions.colorWriteEnable = true;
            }
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        // 我们要找既支持gfx的QF, 又支持present的QF, 找到了就return
        // 注意这里可以是同一个QF,也可以是两个QF(看hw的支持情况).
        QueueFamilyIndices indices;

        // 返回QF的个数, 每个QF在driver里面的index是固定的, 后面向driver查询特定的QF的时候直接传idx查询.
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        // 打印QF的所有信息, 学习, flags里面最重要的就是GRAPHICS_BIT/COMPUTE_BIT/TRANSFER_BIT
        static bool printOnce = true;
        for (size_t i = 0; i < queueFamilyCount && printOnce; i++) {
            const VkQueueFamilyProperties& queueFamily = queueFamilies[i];
            const VkExtent3D& granularity = queueFamily.minImageTransferGranularity;

            std::cout << "queueFamily[" << i << "]:\n"
                      << "  queueFlags: 0x" << std::hex << queueFamily.queueFlags << std::dec << '\n'
                      << "  queueCount: " << queueFamily.queueCount << '\n'
                      << "  timestampValidBits: " << queueFamily.timestampValidBits << '\n'
                      << "  minImageTransferGranularity: ["
                      << granularity.width << ", " << granularity.height << ", " << granularity.depth << "]\n";
        }
        printOnce = false;

        int i = 0;
        for (const VkQueueFamilyProperties& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

#if !defined(__ANDROID__)
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }
#endif

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
#if defined(__ANDROID__)
        // offscreen不需要VK_KHR_surface/VK_KHR_android_surface.
        std::vector<const char*> extensions;
#else
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        // 获取glfw需要的instance extensions, 这些extensions是Vulkan的instance必须支持的, glfw要使用这些extensions来创建instance.
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
#endif

        // 如果开启了validation layers, 则需要添加VK_EXT_DEBUG_UTILS_EXTENSION_NAME, 这个extension是Vulkan的instance必须支持的, 用于调试.
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        // 打印所有的instance layers
        std::cout << "instance layer count: " << layerCount << std::endl;
        for (const VkLayerProperties& layerProperties : availableLayers) {
            std::cout << "layer: " << layerProperties.layerName << '\n'
                      << "  specVersion: " << VK_API_VERSION_MAJOR(layerProperties.specVersion) << '.'
                      << VK_API_VERSION_MINOR(layerProperties.specVersion) << '.'
                      << VK_API_VERSION_PATCH(layerProperties.specVersion) << '\n'
                      << "  implementationVersion: " << layerProperties.implementationVersion << '\n'
                      << "  description: " << layerProperties.description << "\n\n";
        }

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const VkLayerProperties& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
};

#if defined(__ANDROID__)
int main(int argc, char** argv) {
#else
int main() {
#endif
    HelloTriangleApplication app;

    try {
#if defined(__ANDROID__)
        if (argc > 3) {
            throw std::runtime_error("usage: program [frameCount [outputFile]]");
        }
        if (argc > 1) {
            std::string value(argv[1]);
            size_t consumed = 0;
            unsigned long long count = std::stoull(value, &consumed);
            if (value.empty() || value[0] == '-' || consumed != value.size() || count > UINT32_MAX) {
                throw std::runtime_error("frameCount must be an unsigned 32-bit integer");
            }
            frameCount = static_cast<uint32_t>(count);
        }
        if (argc > 2) {
            outputFile = argv[2];
        }
#endif
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
