cmake_minimum_required (VERSION 3.27)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic)

project (vulkan-tutorial)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)

add_subdirectory(lib/glfw)
add_subdirectory(lib/glm)

set(SOURCE_DIRECTORY src)

file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS ${SOURCE_DIRECTORY}/*.cpp)

add_executable(app src/main.cpp)
target_include_directories(app PRIVATE ${SOURCE_DIRECTORY})

# target_include_directories(app PUBLIC "${VULKAN_SDK}/Include")
# target_include_directories(glfw PUBLIC "${VULKAN_SDK}/Include")

target_link_libraries(app Vulkan::Vulkan glfw glm)
