cmake_minimum_required(VERSION 3.16)
project(Fractal)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(/usr/local/include /opt/homebrew/include /opt/homebrew/opt/llvm/include)
include_directories(colormap)

add_executable(Fractal main.cpp colormap/colormap.cpp colormap/colormap.h)
target_link_libraries(Fractal PUBLIC ${OpenCV_LIBS} )