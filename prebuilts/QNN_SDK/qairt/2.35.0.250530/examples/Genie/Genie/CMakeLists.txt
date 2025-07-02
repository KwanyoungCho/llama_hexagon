cmake_minimum_required(VERSION 3.14)
project(genie)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
add_compile_definitions(qualla_EXPORTS)

# Include directories
set(GENIE_ENGINES_CPU_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src/qualla/engines/qnn-cpu)
set(GENIE_ENGINES_GPU_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src/qualla/engines/qnn-gpu)
set(GENIE_ENGINES_API_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src/qualla/engines/qnn-api)
set(GENIE_ENGINES_HTP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src/qualla/engines/qnn-htp)
set(GENIE_ENGINES_TOKENIZERS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src/qualla/tokenizers)
set(GENIE_QUALLA_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src/qualla/include)
set(GENIE_QUALLA_MMAPPEDFILE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src/qualla/MmappedFile/include)
set(GENIE_INCLUDE include src)
set(GENIE_PIPELINE_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline)
set(QNN_API_INCLUDE ../../../include/QNN)
set(QNN_API_HTP_INCLUDE ${QNN_API_INCLUDE}/HTP)
set(GENIE_C_API_HEADERS_INCLUDE ../../../include/Genie)

# Set the target directory
set(TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
file(MAKE_DIRECTORY ${TARGET_DIR})

# Define source files
file(GLOB_RECURSE SOURCES_GENIE_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER SOURCES_GENIE_CPP EXCLUDE REGEX "DmaBufAllocator.cpp")
set(RUST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/qualla/tokenizers/rust)

# Compiler and linker flags
set(COMMON_CFLAGS "${CMAKE_C_FLAGS}")
set(COMMON_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Definitions
set(QUALLA_ENABLE_ENGINE_QNN_HTP TRUE)
set(QUALLA_ENABLE_ENGINE_QNN_GPU FALSE)
set(QUALLA_ENABLE_ENGINE_QNN_CPU TRUE)
set(GENIE_E2T_FEATURE TRUE)

get_directory_property(defs COMPILE_DEFINITIONS)
list(FILTER defs EXCLUDE REGEX "QNN_API.*")
list(APPEND defs "GENIE_API=__declspec(dllexport)")
list(APPEND defs
    QUALLA_ENABLE_ENGINE_QNN_CPU=${QUALLA_ENABLE_ENGINE_QNN_CPU}
    QUALLA_ENABLE_ENGINE_QNN_HTP=${QUALLA_ENABLE_ENGINE_QNN_HTP}
    QUALLA_ENABLE_ENGINE_QNN_GPU=${QUALLA_ENABLE_ENGINE_QNN_GPU}
    NOMINMAX
    NOGDI
    SPILLFILL
    QUALLA_APPS=OFF
    FMT_HEADER_ONLY=1
    QUALLA_INTERNAL_QNN_SDK
    GENIE_SSD_FEATURE
    GENIE_SPD_FEATURE
    GENIE_LADE_FEATURE
    GENIE_MULTISTREAM_FEATURE
    GENIE_LORA_FEATURE
    GENIE_E2T_FEATURE
    JSON_HAS_THREE_WAY_COMPARISON=0)
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "${defs}")

set(TOKENIZERS_RUST_TARGET "x86_64-pc-windows-msvc")
set(TARGET_ARCH "x86_64-pc-windows-msvc")
if(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    set(TOKENIZERS_RUST_TARGET "aarch64-pc-windows-msvc")
	if (CMAKE_C_COMPILER_TARGET MATCHES ".*llvm.*")
		set(TOKENIZERS_CPP_CARGO_TARGET aarch64-pc-windows-gnullvm)
		list(APPEND TOKENIZERS_CPP_CARGO_FLAGS -Z build-std)
	endif()
	set(TARGET_ARCH "aarch64-pc-windows-msvc")
endif()

# Define the Rust library build as a custom target
add_custom_target(build_tokenizer ALL
    COMMAND cargo build --release --manifest-path=${RUST_SOURCE_DIR}/Cargo.toml --target ${TOKENIZERS_RUST_TARGET}
    WORKING_DIRECTORY ${RUST_SOURCE_DIR}
    COMMENT "Building Rust tokenizer library"
)

# Define the target library
add_library(Genie SHARED ${SOURCES_GENIE_CPP})
target_sources(Genie PUBLIC ${SOURCES_GENIE_CPP})

# Add include directories
target_include_directories(Genie PUBLIC
    ${GENIE_ENGINES_CPU_INCLUDE}
    ${GENIE_ENGINES_GPU_INCLUDE}
    ${GENIE_ENGINES_API_INCLUDE}
    ${GENIE_ENGINES_HTP_INCLUDE}
    ${GENIE_ENGINES_TOKENIZERS_INCLUDE}
    ${GENIE_QUALLA_INCLUDE}
    ${GENIE_QUALLA_MMAPPEDFILE_INCLUDE}
    ${GENIE_INCLUDE}
    ${QNN_API_INCLUDE}
    ${QNN_API_HTP_INCLUDE}
    ${GENIE_C_API_HEADERS_INCLUDE}
    ${GENIE_PIPELINE_INCLUDE})

set_target_properties(Genie PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(Genie PUBLIC ${defs})

# Link the Rust static library
set(LIBTOKENIZERS ${RUST_SOURCE_DIR}/target/${TARGET_ARCH}/release/tokenizers_capi.lib)
set(STATIC_LIBS_FOR_RUST "bcrypt.lib;advapi32.lib;kernel32.lib;ntdll.lib;userenv.lib;ws2_32.lib")
add_dependencies(Genie build_tokenizer)
target_link_libraries(Genie PUBLIC ${LIBTOKENIZERS} ${STATIC_LIBS_FOR_RUST})

# Output directories
set_target_properties(Genie PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${TARGET_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${TARGET_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${TARGET_DIR}
)
