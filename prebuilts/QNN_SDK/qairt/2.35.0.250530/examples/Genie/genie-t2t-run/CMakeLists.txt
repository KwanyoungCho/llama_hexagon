cmake_minimum_required(VERSION 3.14)
project(genie-t2t-run)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Create genie-t2t-run.exe
add_executable(genie-t2t-run ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_include_directories(genie-t2t-run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../..//include/Genie)
target_link_libraries(genie-t2t-run PRIVATE Genie)
