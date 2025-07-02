cmake_minimum_required(VERSION 3.14)
project(MatMulBenchmark)

if(NOT DEFINED QNN_SDK_PATH)
    # Fallback: two levels up from this example directory
    set(QNN_SDK_PATH ${CMAKE_CURRENT_LIST_DIR}/../..)
    message(STATUS "QNN_SDK_PATH not specified; using ${QNN_SDK_PATH}")
endif()

set(QNN_SDK_ROOT ${QNN_SDK_PATH})

# Include directories - 모든 필요한 QNN 헤더 디렉토리 포함
include_directories(
    ${QNN_SDK_ROOT}/include
    ${QNN_SDK_ROOT}/include/QNN
    ${QNN_SDK_ROOT}/include/QNN/HTP
    ${QNN_SDK_ROOT}/include/QNN/CPU
    ${QNN_SDK_ROOT}/include/QNN/GPU
    ${QNN_SDK_ROOT}/include/QNN/DSP
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Source file
add_executable(MatMulBenchmark MatMulBenchmark.cpp)

# Link QNN backend libraries at runtime (dlopen), so only dl required at build
find_package(Threads REQUIRED)
find_library(DL_LIB dl REQUIRED)

target_link_libraries(MatMulBenchmark PRIVATE ${DL_LIB} Threads::Threads)

# Optional: set C++17
set_target_properties(MatMulBenchmark PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Compiler flags for better error handling
target_compile_options(MatMulBenchmark PRIVATE 
    -Wall -Wextra
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3>
) 