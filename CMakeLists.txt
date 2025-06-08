cmake_minimum_required(VERSION 3.10)
project(simd_tensor_backend C)

set(CMAKE_C_STANDARD 11)

# Create build/lib if it doesn't exist
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# Find OpenMP package
find_package(OpenMP)

# Source files
file(GLOB SOURCES "src/*.c")

add_library(simd_tensor_backend SHARED ${SOURCES})

# Include directories
target_include_directories(simd_tensor_backend PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Enable AVX2, FMA, and optimization flags
target_compile_options(simd_tensor_backend PRIVATE 
    -mavx2 
    -mfma 
    -O3
    -Wall
    -Wextra
)

# Link with math library and OpenMP if available
target_link_libraries(simd_tensor_backend PRIVATE m)

if(OpenMP_C_FOUND)
    target_compile_options(simd_tensor_backend PRIVATE ${OpenMP_C_FLAGS})
    target_link_libraries(simd_tensor_backend PRIVATE ${OpenMP_C_LIBRARIES})
endif()

# Install targets
install(TARGETS simd_tensor_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)