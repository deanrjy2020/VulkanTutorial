// 50.1 dynamic rendering (Vulkan 1.3 core, 原来是VK_KHR_dynamic_rendering), 从15.2 copy过来, 画的东西完全一样.
// 和15.2的区别:
//   1. 没有VkRenderPass/VkFramebuffer. pipeline创建时用VkPipelineRenderingCreateInfo告诉它attachment的format.
//   2. vkCmdBeginRenderPass/vkCmdEndRenderPass 换成 vkCmdBeginRendering/vkCmdEndRendering,
//      attachment(image view, layout, loadOp/storeOp, clear value)在VkRenderingAttachmentInfo里直接给.
//   3. render pass原来隐式做的layout transition(initialLayout -> subpass layout -> finalLayout)和subpass dependency,
//      现在要自己在recordCommandBuffer里用vkCmdPipelineBarrier2(sync2, 也是1.3 core)写.
// 需要device支持Vulkan 1.3, Android上NDK要用android-33以上的stub, 才导出1.3的函数.

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

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

#if defined(__ANDROID__)
// 画固定帧数就退出, 最后一帧写到out.png. 0表示一直画(Ctrl+C退出, 不走cleanup, 也不写out.png).
uint32_t frameCount = 1;
const char* outputFile = "out.png";
#endif

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#if defined(__ANDROID__)
// offscreen不需要VK_KHR_swapchain.
const std::vector<const char*> deviceExtensions = {};
#else
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
#endif

#ifdef NDEBUG
const bool enableValidationLayers = false;
#elif defined(__ANDROID__)
// device上不一定push了validation layer(pushvalidationlayer), 找不到就关掉继续跑, 所以不是const.
bool enableValidationLayers = true;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
#if defined(__ANDROID__)
        // 没有surface, 不需要present queue.
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

    // Android上没有swapchain, swapChainImages里只放一张自己创建的colorImage,
    // 这样后面的createImageViews/recordCommandBuffer都不用改.
    // dynamic rendering不需要VkFramebuffer, image view直接给vkCmdBeginRendering.
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;

#if defined(__ANDROID__)
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;

    // colorImage copy到这里, host visible, 最后map出来写out.png.
    VkBuffer readbackBuffer;
    VkDeviceMemory readbackBufferMemory;
#endif

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    VkFence inFlightFence;

    void initWindow() {
#if defined(__ANDROID__)
        // 没有window, 图片大小就是WIDTH x HEIGHT.
#else
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
#endif
    }

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createGraphicsPipeline();
        createCommandPool();
        createCommandBuffer();
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

        vkDeviceWaitIdle(device);

#if defined(__ANDROID__)
        saveImage(outputFile);
#endif
    }

    void cleanup() {
#if defined(__ANDROID__)
        vkDestroyBuffer(device, readbackBuffer, nullptr);
        vkFreeMemory(device, readbackBufferMemory, nullptr);
#else
        vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
#endif
        vkDestroyFence(device, inFlightFence, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

#if defined(__ANDROID__)
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorImageMemory, nullptr);
#else
        vkDestroySwapchainKHR(device, swapChain, nullptr);
#endif
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

    void createInstance() {
#if defined(__ANDROID__) && !defined(NDEBUG)
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            std::cerr << "validation layers requested, but not available, continue without them." << std::endl;
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
        // dynamic rendering和sync2都是1.3 core.
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
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

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
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

        // 1.3 core的feature也要显式打开, isDeviceSuitable里已经查过支持.
        VkPhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13Features.dynamicRendering = VK_TRUE;
        vulkan13Features.synchronization2 = VK_TRUE;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        // pEnabledFeatures可以和VkPhysicalDeviceVulkan13Features一起用, 只有VkPhysicalDeviceFeatures2不能和pEnabledFeatures同时给.
        createInfo.pNext = &vulkan13Features;

        // 创建logical dev需要queueCI信息
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

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

    // 代替swapchain: 一张WIDTH x HEIGHT的color attachment, 画完copy出去所以还要TRANSFER_SRC.
    // R8G8B8A8_SRGB做color attachment是spec要求必须支持的, 不用查format properties.
    // 用SRGB和原版chooseSwapSurfaceFormat选的B8G8R8A8_SRGB一致, 读回来的字节直接就是显示用的值, RGBA顺序直接给stbi_write_png.
    void createSwapChain() {
        swapChainImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
        swapChainExtent = {WIDTH, HEIGHT};

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = swapChainImageFormat;
        imageInfo.extent = {WIDTH, HEIGHT, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(device, &imageInfo, nullptr, &colorImage) != VK_SUCCESS) {
            throw std::runtime_error("failed to create color image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, colorImage, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &colorImageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate color image memory!");
        }

        vkBindImageMemory(device, colorImage, colorImageMemory, 0);

        swapChainImages = {colorImage};
    }

    void createReadbackBuffer() {
        // RGBA8, 4 bytes per pixel, tightly packed.
        VkDeviceSize size = static_cast<VkDeviceSize>(swapChainExtent.width) * swapChainExtent.height * 4;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &readbackBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create readback buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, readbackBuffer, &memRequirements);

        // HOST_COHERENT, map之后不用vkInvalidateMappedMemoryRanges.
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &readbackBufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate readback buffer memory!");
        }

        vkBindBufferMemory(device, readbackBuffer, readbackBufferMemory, 0);
    }

    // 用stb写成PNG, RGBA 4 channel原样写(alpha是clear color的1.0), Windows上可以直接打开.
    void saveImage(const char* filename) {
        uint32_t width = swapChainExtent.width;
        uint32_t height = swapChainExtent.height;

        void* data = nullptr;
        vkMapMemory(device, readbackBufferMemory, 0, VK_WHOLE_SIZE, 0, &data);

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

        createInfo.oldSwapchain = VK_NULL_HANDLE;

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

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("shaders/vert.spv");
        auto fragShaderCode = readFile("shaders/frag.spv");

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
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // 没有render pass, pipeline需要知道attachment的format(原来从renderPass + subpass里拿).
        // 必须和vkCmdBeginRendering给的attachment一致.
        VkPipelineRenderingCreateInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderingInfo.viewMask = 0;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &swapChainImageFormat;
        renderingInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
        renderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = &renderingInfo;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = VK_NULL_HANDLE;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkImage image = swapChainImages[imageIndex];

        // 原来render pass的initialLayout(UNDEFINED) -> subpass layout(COLOR_ATTACHMENT_OPTIMAL), 加上
        // EXTERNAL -> subpass 0的dependency, 现在都在这一个barrier里.
        // srcStage用COLOR_ATTACHMENT_OUTPUT: Windows上submit时imageAvailableSemaphore的wait stage就是它,
        // barrier的src scope从这个stage开始, 才能和semaphore wait串起来(等acquire完成再transition).
        // oldLayout = UNDEFINED: 反正要clear, 不要以前的内容.
        VkImageMemoryBarrier2 toColorAttachment{};
        toColorAttachment.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toColorAttachment.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toColorAttachment.srcAccessMask = VK_ACCESS_2_NONE;
        toColorAttachment.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toColorAttachment.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toColorAttachment.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toColorAttachment.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toColorAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toColorAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toColorAttachment.image = image;
        toColorAttachment.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        toColorAttachment.subresourceRange.baseMipLevel = 0;
        toColorAttachment.subresourceRange.levelCount = 1;
        toColorAttachment.subresourceRange.baseArrayLayer = 0;
        toColorAttachment.subresourceRange.layerCount = 1;

        VkDependencyInfo toColorAttachmentDependency{};
        toColorAttachmentDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        toColorAttachmentDependency.imageMemoryBarrierCount = 1;
        toColorAttachmentDependency.pImageMemoryBarriers = &toColorAttachment;
        vkCmdPipelineBarrier2(commandBuffer, &toColorAttachmentDependency);

        // 原来VkAttachmentDescription(loadOp/storeOp) + VkFramebuffer(image view) + VkRenderPassBeginInfo(clear value)
        // 的内容, 现在都在VkRenderingAttachmentInfo里.
        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = swapChainImageViews[imageIndex];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.resolveMode = VK_RESOLVE_MODE_NONE;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue = {{{0.0f, 0.0f, 0.0f, 1.0f}}};

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.offset = {0, 0};
        renderingInfo.renderArea.extent = swapChainExtent;
        renderingInfo.layerCount = 1;
        renderingInfo.viewMask = 0;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = nullptr;
        renderingInfo.pStencilAttachment = nullptr;

        vkCmdBeginRendering(commandBuffer, &renderingInfo);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdDraw(commandBuffer, 3, 1, 0, 0);

        vkCmdEndRendering(commandBuffer);

        // 原来render pass的finalLayout, 加上subpass 0 -> EXTERNAL的dependency.
        VkImageMemoryBarrier2 afterRendering{};
        afterRendering.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        afterRendering.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        afterRendering.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
#if defined(__ANDROID__)
        // 后面要copy到readbackBuffer.
        afterRendering.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        afterRendering.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        afterRendering.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
#else
        // present不是pipeline stage, 靠renderFinishedSemaphore同步, 这里dst不用等任何stage.
        afterRendering.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        afterRendering.dstAccessMask = VK_ACCESS_2_NONE;
        afterRendering.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        afterRendering.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        afterRendering.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        afterRendering.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        afterRendering.image = image;
        afterRendering.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        afterRendering.subresourceRange.baseMipLevel = 0;
        afterRendering.subresourceRange.levelCount = 1;
        afterRendering.subresourceRange.baseArrayLayer = 0;
        afterRendering.subresourceRange.layerCount = 1;

        VkDependencyInfo afterRenderingDependency{};
        afterRenderingDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        afterRenderingDependency.imageMemoryBarrierCount = 1;
        afterRenderingDependency.pImageMemoryBarriers = &afterRendering;
        vkCmdPipelineBarrier2(commandBuffer, &afterRenderingDependency);

#if defined(__ANDROID__)
        // 整张colorImage copy到readbackBuffer.
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
        vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer, 1, &region);

        // 让host能看到copy的结果. submit时的fence只保证执行完, 不保证host可见.
        VkBufferMemoryBarrier2 toHost{};
        toHost.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        toHost.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        toHost.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toHost.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        toHost.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        toHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toHost.buffer = readbackBuffer;
        toHost.offset = 0;
        toHost.size = VK_WHOLE_SIZE;

        VkDependencyInfo toHostDependency{};
        toHostDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        toHostDependency.bufferMemoryBarrierCount = 1;
        toHostDependency.pBufferMemoryBarriers = &toHost;
        vkCmdPipelineBarrier2(commandBuffer, &toHostDependency);
#endif

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

#if defined(__ANDROID__)
        // 没有acquire/present, 不需要semaphore, 只要fence让cpu等gpu.
        if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
#else
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
#endif

    }

#if defined(__ANDROID__)
    void drawFrame() {
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFence);

        // 只有一张colorImage.
        uint32_t imageIndex = 0;

        vkResetCommandBuffer(commandBuffer, /*VkCommandBufferResetFlagBits*/ 0);
        recordCommandBuffer(commandBuffer, imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        // 和原版的vkQueueWaitIdle(presentQueue)对应, 一帧一帧来.
        vkQueueWaitIdle(graphicsQueue);
    }
#else
    void drawFrame() {
        // 所有frame(swapchain里面3个images)共用一个fence, cpu用来等gpu, cmd里面就画一个三角形, 每次submit cmd到queue都要带一个fence,
        // 然后cpu在每次drawFrame开始的时候等这个fence, 即上一帧gpu做完了在画. (效率不高)
        vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFence);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

        vkResetCommandBuffer(commandBuffer, /*VkCommandBufferResetFlagBits*/ 0);
        recordCommandBuffer(commandBuffer, imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        // vkQueueSubmit要2个semaphore,
        // 1, 什么时候开始? 即当前要提交的cmd要等这个semaphore, 即只有这个swap chain image available了, 才能在上面画图.
        VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        // 2, 什么时候结束?
        // 当前要提交的cmd被gpu执行了会signal这个semaphore, 告诉其他gpu cmd, 这个cmd已经执行完了,
        // 你可以开始你的cmd了(如果你的cmd pending在我这里的话). 这个例子里面后面的QueuePresent在等这个semaphore.
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS) {
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

        vkQueuePresentKHR(presentQueue, &presentInfo);

        // 加上这个validation layer不会有抱怨, 这里不用太纠结同步的事情, 放到27
        vkQueueWaitIdle(presentQueue);
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
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

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

    // device要支持1.3, 并且dynamicRendering/synchronization2这两个feature要报true.
    // (1.3 spec要求这两个必须支持, 这里查一下是为了学习, 也能在driver报的版本和feature对不上时早点发现.)
    bool isVulkan13Supported(VkPhysicalDevice device) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);

        std::cout << "device: " << properties.deviceName << ", apiVersion "
                  << VK_API_VERSION_MAJOR(properties.apiVersion) << '.'
                  << VK_API_VERSION_MINOR(properties.apiVersion) << '.'
                  << VK_API_VERSION_PATCH(properties.apiVersion) << std::endl;

        if (properties.apiVersion < VK_API_VERSION_1_3) {
            return false;
        }

        VkPhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &vulkan13Features;
        vkGetPhysicalDeviceFeatures2(device, &features2);

        std::cout << "  dynamicRendering: " << vulkan13Features.dynamicRendering
                  << ", synchronization2: " << vulkan13Features.synchronization2 << std::endl;

        return vulkan13Features.dynamicRendering == VK_TRUE && vulkan13Features.synchronization2 == VK_TRUE;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        if (!isVulkan13Supported(device)) {
            return false;
        }

#if defined(__ANDROID__)
        // 没有surface, 不用查swapchain support.
        return indices.isComplete() && extensionsSupported;
#else
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
#endif
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        // 100多个, 不打印了, 用vulkaninfo看.

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
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
            const auto& queueFamily = queueFamilies[i];
            const auto& granularity = queueFamily.minImageTransferGranularity;

            std::cout << "queueFamily[" << i << "]:\n"
                      << "  queueFlags: 0x" << std::hex << queueFamily.queueFlags << std::dec << '\n'
                      << "  queueCount: " << queueFamily.queueCount << '\n'
                      << "  timestampValidBits: " << queueFamily.timestampValidBits << '\n'
                      << "  minImageTransferGranularity: ["
                      << granularity.width << ", " << granularity.height << ", " << granularity.depth << "]\n";
        }
        printOnce = false;

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
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
        for (const auto& layerProperties : availableLayers) {
            std::cout << "layer: " << layerProperties.layerName << '\n'
                      << "  specVersion: " << VK_API_VERSION_MAJOR(layerProperties.specVersion) << '.'
                      << VK_API_VERSION_MINOR(layerProperties.specVersion) << '.'
                      << VK_API_VERSION_PATCH(layerProperties.specVersion) << '\n'
                      << "  implementationVersion: " << layerProperties.implementationVersion << '\n'
                      << "  description: " << layerProperties.description << "\n\n";
        }

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
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
// usage: 50.1_dynamic_rendering [frameCount [outputFile]], 默认画1帧, 写到out.png. 0 = 一直画.
// cwd下要有shaders/vert.spv和shaders/frag.spv.
int main(int argc, char** argv) {
    if (argc > 1) {
        frameCount = static_cast<uint32_t>(std::stoul(argv[1]));
    }
    if (argc > 2) {
        outputFile = argv[2];
    }
#else
int main() {
#endif
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
