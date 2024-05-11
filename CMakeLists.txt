cmake_minimum_required(VERSION 3.16)
project(Fractal)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_STANDARD 17)
# set(OPENMP_LIBRARIES "/opt/homebrew/opt/libomp/lib")
# set(OPENMP_INCLUDES "/opt/homebrew/opt/libomp/include")
# if(APPLE)
#     set(CMAKE_C_COMPILER clang)
#     set(CMAKE_CXX_COMPILER clang++)

#     if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
#         set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
#         set(OpenMP_C_LIB_NAMES "omp")
#         set(OpenMP_omp_LIBRARY omp)
#     endif()

#     if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
#         set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
#         set(OpenMP_CXX_LIB_NAMES "omp")
#         set(OpenMP_omp_LIBRARY omp)
#     endif()

# endif()

# find_package(OpenMP REQUIRED)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(/usr/local/include /opt/homebrew/include /opt/homebrew/opt/llvm/include)
include_directories(colormap)

add_executable(Fractal main.cpp colormap/colormap.cpp colormap/colormap.h)
# target_link_libraries(Fractal PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX ${OpenCV_LIBS} )
target_link_libraries(Fractal PUBLIC ${OpenCV_LIBS} )

# OPENMP::OpenMP_CXX