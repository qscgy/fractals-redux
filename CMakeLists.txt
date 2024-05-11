cmake_minimum_required(VERSION 3.16)
project(Fractal)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_COMPILER "/usr/local/Cellar/llvm/16.0.3/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/local/Cellar/llvm/16.0.3/bin/clang++")
set (CMAKE_CXX_STANDARD 17)
set(OPENMP_LIBRARIES "/usr/local/Cellar/llvm/16.0.3/lib")
set(OPENMP_INCLUDES "/usr/local/Cellar/llvm/16.0.3/include")
if(APPLE)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

endif()

find_package(OpenMP REQUIRED)

include_directories(/usr/local/include /opt/homebrew/include /opt/homebrew/opt/llvm/include)
include_directories(colormap)

add_executable(Fractal main.cpp colormap/colormap.cpp colormap/colormap.h)
target_link_libraries(Fractal PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX ${OpenCV_LIBS} )
# OPENMP::OpenMP_CXX