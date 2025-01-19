#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

class Application
{
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    GLFWwindow* window;

    VkInstance instance;

    void initWindow()
    {
        // initializes the GLFW library
        glfwInit();

        // tell GLFW to not create an OpenGL context
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        // disable window resizing
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        // creating the actual window
        window = glfwCreateWindow(800, 600, "Hi :D", nullptr, nullptr);
    }

    void initVulkan()
    {
        createInstance();
    }

    void mainLoop()
    {
        // keep the application running until either an error occurs or the window is closed
        while (!glfwWindowShouldClose(window)) glfwPollEvents();
    }

    void cleanup()
    {
        // destroy the instance
        vkDestroyInstance(instance, nullptr);

        // destroying the window
        glfwDestroyWindow(window);

        // terminating GLFW
        glfwTerminate();
    }

    // Instance
    void createInstance()
    {
        // fill in a struct with some information about our application
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hi :)";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        // tells the driver which global extensions and validation layers to use
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        createInfo.enabledLayerCount = 0;

        // create an instance and check if the instance was created successfully
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create instance!");
        }
    }
};

int main()
{
    try
    {
        Application app;
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}