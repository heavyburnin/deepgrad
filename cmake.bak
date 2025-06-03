cmake_minimum_required(VERSION 3.10)
project(simd_tensor_backend C)

set(CMAKE_C_STANDARD 11)

# Create build/lib if it doesn't exist
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# Enable AVX2 or AVX512
add_compile_options(-mavx2 -O3)

# Source files
file(GLOB SOURCES "src/*.c")

add_library(simd_tensor_backend SHARED ${SOURCES})

target_compile_options(simd_tensor_backend PRIVATE -mavx2 -mfma -O3)
