cmake_minimum_required(VERSION 2.8)
project(monocular_initialization)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 3 REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(${PROJECT_NAME} main.cpp monocular_initialization.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

