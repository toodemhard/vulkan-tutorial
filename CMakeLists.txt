cmake_minimum_required (VERSION 3.27)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


project (vulkan-tutorial)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)

add_subdirectory(lib/glfw)
add_subdirectory(lib/glm)

add_compile_options(-Wall)# -Wextra -Wpedantic)

set(SOURCE_DIRECTORY src)
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS ${SOURCE_DIRECTORY}/*.cpp)

# add_executable(app ${SOURCE_FILES})
add_executable(app src/main.cpp)
target_include_directories(app PRIVATE ${SOURCE_DIRECTORY})

set_target_properties(
    app PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

target_link_libraries(app Vulkan::Vulkan glfw glm)
