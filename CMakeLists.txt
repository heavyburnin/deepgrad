cmake_minimum_required(VERSION 3.10)
project(simd_tensor_backend C)

set(CMAKE_C_STANDARD 11)

# Set output path
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Gather source files
file(GLOB SOURCES "src/*.c")

# Define the shared library
add_library(simd_tensor_backend SHARED ${SOURCES})

# Include headers
target_include_directories(simd_tensor_backend PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compiler optimizations and warnings
target_compile_options(simd_tensor_backend PRIVATE 
    -Wno-unused-parameter
    -mavx2 
    -mfma 
    -O3
    -Wall
    -Wextra
)

# Link system math and OpenMP
target_link_libraries(simd_tensor_backend PRIVATE m OpenMP::OpenMP_C)

# Install targets
install(TARGETS simd_tensor_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)