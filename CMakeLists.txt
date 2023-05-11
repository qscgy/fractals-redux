cmake_minimum_required(VERSION 3.10)
project(Fractal)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
set(CMAKE_BUILD_TYPE Debug)

include_directories(/usr/local/include /opt/homebrew/include)

add_executable(Fractal main.cpp)
target_link_libraries(Fractal Eigen3::Eigen)
target_link_libraries(Fractal ${OpenCV_LIBS})