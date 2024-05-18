#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <cstdlib>
#include <exception>
#include <format>
#include <iostream>
#include <optional>

const int WIDTH = 800;
const int HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validation_layers = {
    "VK_LAYER_KHRONOS_validation",
};

#ifdef NDEBGUG
const bool enable_validation_layers = false;
#else
const bool enable_validation_layers = true;
#endif

VkResult create_debug_utils_messenger_EXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* p_create_info,
    const VkAllocationCallbacks* p_allocator,
    VkDebugUtilsMessengerEXT* p_debug_messenger
) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT
    )vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, p_create_info, p_allocator, p_debug_messenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void destroy_debug_utils_messenger_EXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debug_messenger,
    const VkAllocationCallbacks* p_allocator
) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT
    )vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

    if (func != nullptr) {
        func(instance, debug_messenger, p_allocator);
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_family;
    std::optional<uint32_t> present_family;

    bool is_complete() {
        return graphics_family.has_value() && present_family.has_value();
    }
};

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
};

class HelloTriangleApplication {
  public:
    void run() {
        init_window();
        init_vulkan();
        main_loop();
        cleanup();
    }

  private:
    GLFWwindow* window;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphics_queue;
    VkQueue present_queue;

    VkSwapchainKHR swapchain;
    std::vector<VkImage> swapchain_images;
    VkFormat swapchain_image_format;
    VkExtent2D swapchain_extent;
    std::vector<VkFramebuffer> swapchain_framebuffers;
    std::vector<VkImageView> swapchain_image_views;

    VkRenderPass render_pass;
    VkPipelineLayout pipeline_layout;
    VkPipeline graphics_pipeline;

    VkCommandPool command_pool;
    std::vector<VkCommandBuffer> command_buffers;

    std::vector<VkSemaphore> image_available_semaphores;
    std::vector<VkSemaphore> render_finished_semaphores;
    std::vector<VkFence> in_flight_fences;

    uint32_t current_frame = 0;
    bool framebuffer_resized = false;

    const std::vector<const char*> device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    void init_window() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);
    }

    static void framebuffer_resize_callback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebuffer_resized = true;
    }

    void init_vulkan() {
        create_instance();
        setup_debug_messenger();
        create_surface();
        pick_physical_device();
        create_logical_device();
        create_swapchain();
        create_image_views();
        create_render_pass();
        create_graphics_pipeline();
        create_framebuffers();
        create_command_pool();
        create_command_buffers();
        create_sync_objects();
    }

    void recreate_swapchain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanup_swapchain();

        create_swapchain();
        create_image_views();
        create_framebuffers();
    }

    void cleanup_swapchain() {
        for (auto& framebuffer : swapchain_framebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto& image_view : swapchain_image_views) {
            vkDestroyImageView(device, image_view, nullptr);
        }

        vkDestroySwapchainKHR(device, swapchain, nullptr);
    }

    void record_command_buffer(VkCommandBuffer command_buffer, uint32_t image_index) {
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        };

        VkRenderPassBeginInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = render_pass;
        render_pass_info.framebuffer = swapchain_framebuffers[image_index];

        render_pass_info.renderArea.offset = {0, 0};
        render_pass_info.renderArea.extent = swapchain_extent;

        VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        render_pass_info.clearValueCount = 1;
        render_pass_info.pClearValues = &clear_color;

        vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapchain_extent.width);
        viewport.height = static_cast<float>(swapchain_extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(command_buffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapchain_extent;
        vkCmdSetScissor(command_buffer, 0, 1, &scissor);

        vkCmdDraw(command_buffer, 3, 1, 0, 0);

        vkCmdEndRenderPass(command_buffer);
        if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffers!");
        }
    }

    void create_command_buffers() {
        command_buffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = command_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = (uint32_t)command_buffers.size();
        if (vkAllocateCommandBuffers(device, &alloc_info, command_buffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void create_command_pool() {
        QueueFamilyIndices queue_family_indices = find_queue_families(physical_device);

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = queue_family_indices.graphics_family.value();

        if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void create_framebuffers() {
        swapchain_framebuffers.resize(swapchain_image_views.size());

        for (size_t i = 0; i < swapchain_image_views.size(); i++) {
            VkImageView attachments[] = {swapchain_image_views[i]};

            VkFramebufferCreateInfo framebuffer_info{};
            framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_info.renderPass = render_pass;
            framebuffer_info.attachmentCount = 1;
            framebuffer_info.pAttachments = attachments;
            framebuffer_info.width = swapchain_extent.width;
            framebuffer_info.height = swapchain_extent.height;
            framebuffer_info.layers = 1;

            if (vkCreateFramebuffer(
                    device, &framebuffer_info, nullptr, &swapchain_framebuffers[i]
                ) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void create_render_pass() {
        VkAttachmentDescription color_attachment{};
        color_attachment.format = swapchain_image_format;
        color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;

        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_attachment_ref{};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = 1;
        render_pass_info.pAttachments = &color_attachment;
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass;

        render_pass_info.dependencyCount = 1;
        render_pass_info.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &render_pass_info, nullptr, &render_pass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void create_graphics_pipeline() {
        auto vert_shader_code = read_file("shaders/vert.spv");
        auto frag_shader_code = read_file("shaders/frag.spv");

        VkShaderModule vert_shader_module = create_shader_module(vert_shader_code);
        VkShaderModule frag_shader_module = create_shader_module(frag_shader_code);

        VkPipelineShaderStageCreateInfo vert_shader_stage_info{};
        vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vert_shader_stage_info.module = vert_shader_module;
        vert_shader_stage_info.pName = "main";

        VkPipelineShaderStageCreateInfo frag_shader_stage_info{};
        frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        frag_shader_stage_info.module = frag_shader_module;
        frag_shader_stage_info.pName = "main";

        VkPipelineShaderStageCreateInfo shader_stages[] = {
            vert_shader_stage_info, frag_shader_stage_info
        };

        VkPipelineVertexInputStateCreateInfo vertex_input_info{};
        vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_info.vertexBindingDescriptionCount = 0;
        vertex_input_info.pVertexBindingDescriptions = nullptr;
        vertex_input_info.vertexAttributeDescriptionCount = 0;
        vertex_input_info.pVertexAttributeDescriptions = nullptr;

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.scissorCount = 1;

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

        VkPipelineColorBlendAttachmentState color_blend_attachment{};
        color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                                VK_COLOR_COMPONENT_G_BIT |
                                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        color_blend_attachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo color_blending{};
        color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE;
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &color_blend_attachment;

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        std::vector<VkDynamicState> dynamic_states = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamic_state{};
        dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
        dynamic_state.pDynamicStates = dynamic_states.data();

        VkGraphicsPipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = 2;
        pipeline_info.pStages = shader_stages;
        pipeline_info.pVertexInputState = &vertex_input_info;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pColorBlendState = &color_blending;
        pipeline_info.pDynamicState = &dynamic_state;

        pipeline_info.layout = pipeline_layout;
        pipeline_info.renderPass = render_pass;
        pipeline_info.subpass = 0;

        if (vkCreateGraphicsPipelines(
                device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline
            ) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, vert_shader_module, nullptr);
        vkDestroyShaderModule(device, frag_shader_module, nullptr);
    }

    VkShaderModule create_shader_module(const std::vector<char>& code) {
        VkShaderModuleCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = code.size();
        create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shader_module;
        if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shader_module;
    }

    void create_image_views() {
        swapchain_image_views.resize(swapchain_images.size());

        for (size_t i = 0; i < swapchain_images.size(); i++) {
            VkImageViewCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            create_info.image = swapchain_images[i];

            create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            create_info.format = swapchain_image_format;

            create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            create_info.subresourceRange.baseMipLevel = 0;
            create_info.subresourceRange.levelCount = 1;
            create_info.subresourceRange.baseArrayLayer = 0;
            create_info.subresourceRange.layerCount = 1;

            if (vkCreateImageView(device, &create_info, nullptr, &swapchain_image_views[i]) !=
                VK_SUCCESS) {
                throw std::runtime_error("failed to create image views!");
            }
        }
    }

    void create_surface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void create_swapchain() {
        SwapchainSupportDetails swapchain_support = query_swapchain_support(physical_device);

        VkSurfaceFormatKHR surface_format = choose_swap_surface_format(swapchain_support.formats);
        VkPresentModeKHR present_mode = choose_swap_present_mode(swapchain_support.present_modes);
        VkExtent2D extent = choose_swap_extent(swapchain_support.capabilities);

        uint32_t image_count = swapchain_support.capabilities.minImageCount + 1;

        if (swapchain_support.capabilities.maxImageCount > 0 &&
            image_count > swapchain_support.capabilities.maxImageCount) {
            image_count = swapchain_support.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        create_info.surface = surface;

        create_info.minImageCount = image_count;
        create_info.imageFormat = surface_format.format;
        create_info.imageColorSpace = surface_format.colorSpace;
        create_info.imageExtent = extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = find_queue_families(physical_device);
        uint32_t queue_family_indices[] = {
            indices.graphics_family.value(), indices.present_family.value()
        };

        if (indices.graphics_family != indices.present_family) {
            create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = 2;
            create_info.pQueueFamilyIndices = queue_family_indices;
        } else {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        create_info.preTransform = swapchain_support.capabilities.currentTransform;
        create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

        create_info.presentMode = present_mode;
        create_info.clipped = VK_TRUE;

        create_info.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &create_info, nullptr, &swapchain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
        swapchain_images.resize(image_count);
        vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.data());

        swapchain_image_format = surface_format.format;
        swapchain_extent = extent;
    }

    VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actual_extent = {
                static_cast<uint32_t>(width), static_cast<uint32_t>(height)
            };

            actual_extent.width = std::clamp(
                actual_extent.width,
                capabilities.minImageExtent.width,
                capabilities.maxImageExtent.width
            );
            actual_extent.height = std::clamp(
                actual_extent.width,
                capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height
            );

            return actual_extent;
        }
    }

    SwapchainSupportDetails query_swapchain_support(VkPhysicalDevice device) {
        SwapchainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t format_count;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);

        if (format_count != 0) {
            details.formats.resize(format_count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(
                device, surface, &format_count, details.formats.data()
            );
        }

        uint32_t present_mode_count;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nullptr);
        if (present_mode_count != 0) {
            details.present_modes.resize(present_mode_count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                device, surface, &present_mode_count, details.present_modes.data()
            );
        }

        return details;
    }

    QueueFamilyIndices find_queue_families(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            device, &queue_family_count, queue_families.data()
        );

        for (size_t i = 0; i < queue_families.size(); i++) {
            if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphics_family = i;
            }

            VkBool32 present_support = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);

            if (present_support) {
                indices.present_family = i;
            }

            if (indices.is_complete()) {
                break;
            }
        }

        return indices;
    }

    void create_logical_device() {
        QueueFamilyIndices indices = find_queue_families(physical_device);

        std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
        std::set<uint32_t> uniqeue_queue_families = {
            indices.graphics_family.value(), indices.present_family.value()
        };

        float queue_priority = 1.0f;

        for (uint32_t queue_family : uniqeue_queue_families) {
            VkDeviceQueueCreateInfo queue_create_info{};
            queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_create_info.queueFamilyIndex = queue_family;
            queue_create_info.queueCount = 1;
            queue_create_info.pQueuePriorities = &queue_priority;
            queue_create_infos.push_back(queue_create_info);
        }

        VkPhysicalDeviceFeatures device_features{};

        VkDeviceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
        create_info.pQueueCreateInfos = queue_create_infos.data();

        create_info.pEnabledFeatures = &device_features;

        create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
        create_info.ppEnabledExtensionNames = device_extensions.data();

        if (enable_validation_layers) {
            create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
            create_info.ppEnabledLayerNames = validation_layers.data();
        } else {
            create_info.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physical_device, &create_info, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device");
        }

        vkGetDeviceQueue(device, indices.graphics_family.value(), 0, &graphics_queue);
        vkGetDeviceQueue(device, indices.present_family.value(), 0, &present_queue);
    }

    void pick_physical_device() {
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

        if (device_count == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

        std::multimap<int, VkPhysicalDevice> candidates;

        for (const auto& device : devices) {
            int score = is_device_suitable(device);
            candidates.insert(std::make_pair(score, device));
        }

        if (candidates.rbegin()->first >= 0) {
            physical_device = candidates.rbegin()->second;
        } else {
            throw std::runtime_error("failed to find suitable GPU!");
        }
    }

    int is_device_suitable(VkPhysicalDevice device) {
        int score = 0;

        VkPhysicalDeviceProperties device_properties;
        VkPhysicalDeviceFeatures device_features;
        vkGetPhysicalDeviceProperties(device, &device_properties);
        vkGetPhysicalDeviceFeatures(device, &device_features);

        if (device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1;
        }

        QueueFamilyIndices indices = find_queue_families(device);

        bool extensions_supported = check_device_extension_support(device);

        bool swapchain_adequate = false;

        if (extensions_supported) {
            SwapchainSupportDetails swapchain_support = query_swapchain_support(device);
            swapchain_adequate =
                !swapchain_support.formats.empty() && !swapchain_support.present_modes.empty();
        }

        if (!indices.is_complete() || !extensions_supported || !swapchain_adequate) {
            score = -1;
        }

        return score;
    }

    bool check_device_extension_support(VkPhysicalDevice device) {
        uint32_t extension_count;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

        std::vector<VkExtensionProperties> available_extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(
            device, nullptr, &extension_count, available_extensions.data()
        );

        std::set<std::string> required_extensions(
            device_extensions.begin(), device_extensions.end()
        );

        for (const auto& extension : available_extensions) {
            required_extensions.erase(extension.extensionName);
        }

        return required_extensions.empty();
    }

    void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& create_info) {
        create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        create_info.pfnUserCallback = debug_callback;
    }

    void setup_debug_messenger() {
        if (!enable_validation_layers) {
            return;
        }

        VkDebugUtilsMessengerCreateInfoEXT create_info;
        populate_debug_messenger_create_info(create_info);

        if (create_debug_utils_messenger_EXT(instance, &create_info, nullptr, &debug_messenger) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger");
        }
    }

    void create_instance() {
        if (enable_validation_layers && !check_validation_layer_support()) {
            throw std::runtime_error("validation layers requested but not available!");
        }

        uint32_t available_extensions_count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &available_extensions_count, nullptr);
        std::vector<VkExtensionProperties> available_extensions(available_extensions_count);
        vkEnumerateInstanceExtensionProperties(
            nullptr, &available_extensions_count, available_extensions.data()
        );
        VkApplicationInfo appInfo{};
        std::cout << "available extensions:\n";

        for (const auto& extension : available_extensions) {
            std::cout << std::format("\t{}\n", extension.extensionName);
        }
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &appInfo;

        if (enable_validation_layers) {
            create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
            create_info.ppEnabledLayerNames = validation_layers.data();
        } else {
            create_info.enabledLayerCount = 0;
        }

        auto extensions = get_required_extensions();

        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
        if (enable_validation_layers) {
            create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
            create_info.ppEnabledLayerNames = validation_layers.data();
            populate_debug_messenger_create_info(debug_create_info);
            create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debug_create_info;
        } else {
            create_info.enabledLayerCount = 0;
            create_info.pNext = nullptr;
        }

        if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance");
        }
    }

    void main_loop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            draw_frame();
        }

        vkDeviceWaitIdle(device);
    }

    void draw_frame() {
        vkWaitForFences(device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);

        uint32_t image_index;
        VkResult result = vkAcquireNextImageKHR(
            device,
            swapchain,
            UINT64_MAX,
            image_available_semaphores[current_frame],
            VK_NULL_HANDLE,
            &image_index
        );

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreate_swapchain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swapchain image!");
        }

        vkResetFences(device, 1, &in_flight_fences[current_frame]);

        vkResetCommandBuffer(command_buffers[current_frame], 0);
        record_command_buffer(command_buffers[current_frame], image_index);

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore wait_semaphores[] = {image_available_semaphores[current_frame]};
        VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = wait_semaphores;
        submit_info.pWaitDstStageMask = wait_stages;

        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffers[current_frame];

        VkSemaphore signal_semaphores[] = {render_finished_semaphores[current_frame]};
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = signal_semaphores;

        if (vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[current_frame]) !=
            VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = signal_semaphores;

        VkSwapchainKHR swapchains[] = {swapchain};
        present_info.swapchainCount = 1;
        present_info.pSwapchains = swapchains;
        present_info.pImageIndices = &image_index;

        result = vkQueuePresentKHR(present_queue, &present_info);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ||
            framebuffer_resized) {
            framebuffer_resized = false;
            recreate_swapchain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swapchain images!");
        }

        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void create_sync_objects() {
        image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
        VkSemaphoreCreateInfo semaphore_info{};
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(
                    device, &semaphore_info, nullptr, &image_available_semaphores[i]
                ) != VK_SUCCESS ||
                vkCreateSemaphore(
                    device, &semaphore_info, nullptr, &render_finished_semaphores[i]
                ) != VK_SUCCESS ||
                vkCreateFence(device, &fence_info, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create semaphores!");
            }
        }
    }

    void cleanup() {
        cleanup_swapchain();

        vkDestroyPipeline(device, graphics_pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyRenderPass(device, render_pass, nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
            vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
            vkDestroyFence(device, in_flight_fences[i], nullptr);
        }

        vkDestroyCommandPool(device, command_pool, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enable_validation_layers) {
            destroy_debug_utils_messenger_EXT(instance, debug_messenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    bool check_validation_layer_support() {
        uint32_t layer_count;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

        std::vector<VkLayerProperties> available_layers(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

        for (const char* layer_name : validation_layers) {
            bool layer_found = false;

            for (const auto& layer_properties : available_layers) {
                if (strcmp(layer_name, layer_properties.layerName) == 0) {
                    layer_found = true;
                    break;
                }
            }

            if (!layer_found) {
                return false;
            }
        }

        return true;
    }

    VkSurfaceFormatKHR
    choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats) {
        for (const auto& available_format : available_formats) {
            if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return available_format;
            }
        }

        return available_formats[0];
    }

    VkPresentModeKHR
    choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes) {
        for (const auto& available_present_mode : available_present_modes) {
            if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return available_present_mode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    std::vector<const char*> get_required_extensions() {
        uint32_t glfw_extension_count = 0;
        const char** glfw_extensions;
        glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

        std::vector<const char*> extensions(
            glfw_extensions, glfw_extensions + glfw_extension_count
        );

        if (enable_validation_layers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        VkDebugUtilsMessageTypeFlagsEXT message_type,
        const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
        void* p_user_data
    ) {
        std::cerr << "validation layer : " << p_callback_data->pMessage << "\n";

        return VK_FALSE;
    }

    static std::vector<char> read_file(const std::string& file_name) {
        std::ifstream file(file_name, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t file_size = (size_t)file.tellg();
        std::vector<char> buffer(file_size);

        file.seekg(0);
        file.read(buffer.data(), file_size);
        file.close();

        return buffer;
    }
};

int main() {
    std::cout << std::filesystem::current_path();

    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
