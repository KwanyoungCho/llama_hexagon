//==============================================================================
//  MatMulBenchmark.cpp
//  Author: Example created for QNN matmul performance measurement on HTP (NPU)
//  Description:
//      This example demonstrates how to build a minimal QNN graph with a single
//      MatMul operation and benchmark its execution latency on the Qualcomm®
//      AI Engine Direct NPU backend (a.k.a. HTP).  The program allows the user
//      to specify matrix sizes (M, K, N), individual datatype configuration 
//      for each input tensor (fp16, fp32, int16, int8, uint8, etc.) and number 
//      of iterations.
//
//      IMPROVEMENTS:
//        - Individual datatype setting for input tensors (-t0, -t1)
//        - Extended datatype support (fp32, uint8, etc.)
//        - Mixed precision computation support
//        - Better error handling and validation
//        - Unified QNN resource management
//        - Pattern-based LLM benchmarking support
//
//      NOTE:
//        1. The code intentionally avoids the heavy SampleApp framework and
//           shows the bare-minimum sequence of QNN API calls.
//        2. Error handling is considerably simplified for readability; in
//           production code you should check *all* return values.
//        3. Datatype support varies by SoC / firmware version.  If the chosen
//           datatype is not supported by the device, graphFinalize() will fail.
//        4. The backend shared library names are SoC- and release-dependent.
//           Adjust BACKEND_LIB and SYSTEM_LIB paths below to match your
//           deployment.
//==============================================================================

//==============================================================================
// SECTION 1: INCLUDES AND MACRO DEFINITIONS
//==============================================================================
#include <QnnCommon.h>
#include <QnnBackend.h>
#include <QnnContext.h>
#include <QnnDevice.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnTensor.h>
#include <QnnTypes.h>
#include <QnnOpDef.h>
#include <QnnLog.h>
#include <QnnError.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <System/QnnSystemInterface.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include <dlfcn.h>
#include <time.h>
#include <algorithm>  // std::find_if를 위해 추가

// Helper macro for simple error checking
#define CHECK_QNN(x)                                            \
    do {                                                       \
        Qnn_ErrorHandle_t _err = (x);                          \
        if (_err != QNN_SUCCESS) {                             \
            std::cerr << "QNN ERROR at " << __FILE__ << ":"    \
                      << __LINE__ << " code=" << _err          \
                      << " (" << #x << ")"                     \
                      << std::endl;                            \
            std::exit(EXIT_FAILURE);                           \
        }                                                      \
    } while (0)


//==============================================================================
// SECTION 2: DATA TYPE AND CONFIGURATION DEFINITIONS
//==============================================================================

// Extended Datatype descriptor with more supported types
struct DTypeDesc {
    const char * name;
    Qnn_DataType_t dtype;
    size_t bytes;
};

// Quantization encoding type selection
enum class QuantizationType {
    UNDEFINED,         // No quantization (undefined)
    SCALE_OFFSET,      // Per-tensor quantization
    AXIS_SCALE_OFFSET  // Per-axis (channel) quantization
};

// Parse quantization type from string
static QuantizationType parseQuantizationType(const std::string& str) {
    if (str == "none" || str == "undefined") {
        return QuantizationType::UNDEFINED;
    } else if (str == "scale" || str == "scale_offset") {
        return QuantizationType::SCALE_OFFSET;
    } else if (str == "axis" || str == "axis_scale_offset") {
        return QuantizationType::AXIS_SCALE_OFFSET;
    } else {
        std::cerr << "Invalid quantization type: " << str << ". Use 'none', 'scale', or 'axis'." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Get quantization type name for display
static const char* getQuantizationTypeName(QuantizationType type) {
    switch (type) {
        case QuantizationType::UNDEFINED: return "undefined";
        case QuantizationType::SCALE_OFFSET: return "scale_offset";
        case QuantizationType::AXIS_SCALE_OFFSET: return "axis_scale_offset";
        default: return "unknown";
    }
}

// NPU MatMul supported data type combinations (from the official table)
struct MatMulConfig {
    const char * config_name;
    Qnn_DataType_t in0_type;
    Qnn_DataType_t in1_type;
    Qnn_DataType_t out_type;
};

// NPU MatMul Configuration Table
static const MatMulConfig SUPPORTED_MATMUL_CONFIGS[] = {
    // FP16 configurations
    {"FP16", QNN_DATATYPE_FLOAT_16, QNN_DATATYPE_FLOAT_16, QNN_DATATYPE_FLOAT_16},
    {"FP16_MIXED", QNN_DATATYPE_FLOAT_16, QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_FLOAT_16},
    
    // FP32 configurations  
    {"FP32", QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_FLOAT_32},
    
    // INT16 configurations (based on the table, ignoring bias in[2])
    {"INT16_U16_U8", QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_16},
    {"INT16_U16_S8", QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_16},
    {"INT16_U16_S16", QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_SFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_16},
    {"INT16_U16_U16", QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_16},
    {"INT16_S16_S8", QNN_DATATYPE_SFIXED_POINT_16, QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_16},
    {"INT16_S16_S16", QNN_DATATYPE_SFIXED_POINT_16, QNN_DATATYPE_SFIXED_POINT_16, QNN_DATATYPE_SFIXED_POINT_16},
    
    // INT8 configurations (based on the table, ignoring bias in[2])
    {"INT8_U8_U8", QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8},
    {"INT8_U8_S8", QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8},
    {"INT8_S8_U8", QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_UFIXED_POINT_8},
    {"INT8_S8_S8", QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_8}
};

static const size_t NUM_SUPPORTED_CONFIGS = sizeof(SUPPORTED_MATMUL_CONFIGS) / sizeof(MatMulConfig);

// Supported Data Types
static const DTypeDesc kTypes[] = {
    {"fp16",  QNN_DATATYPE_FLOAT_16,        2},
    {"fp32",  QNN_DATATYPE_FLOAT_32,        4},
    {"int16", QNN_DATATYPE_SFIXED_POINT_16, 2},
    {"int8",  QNN_DATATYPE_SFIXED_POINT_8,  1},
    {"uint8", QNN_DATATYPE_UFIXED_POINT_8,  1},
    {"uint16", QNN_DATATYPE_UFIXED_POINT_16, 2},
};

static const size_t NUM_TYPES = sizeof(kTypes) / sizeof(kTypes[0]);

//==============================================================================
// SECTION 3: UTILITY FUNCTIONS
//==============================================================================

// Find data type by name
static const DTypeDesc & findType(const std::string & name) {
    for (const auto & t : kTypes) {
        if (name == t.name) return t;
    }
    std::cerr << "Unsupported dtype " << name << ". Supported types: ";
    for (size_t i = 0; i < sizeof(kTypes)/sizeof(kTypes[0]); ++i) {
        std::cerr << kTypes[i].name;
        if (i < sizeof(kTypes)/sizeof(kTypes[0]) - 1) std::cerr << ", ";
    }
    std::cerr << std::endl;
    std::exit(EXIT_FAILURE);
}

// Fill buffer with random data for testing
static void randomFill(uint8_t * buf, size_t size) {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<uint32_t> dist(0, 255);
    for (size_t i = 0; i < size; ++i) buf[i] = static_cast<uint8_t>(dist(rng));
}

// Fill buffer with appropriate quantized data based on datatype
static void fillQuantizedData(uint8_t* buf, size_t size, Qnn_DataType_t dataType) {
    std::mt19937 rng(1234);
    
    switch (dataType) {
        case QNN_DATATYPE_SFIXED_POINT_8: {
            // INT8: -128 ~ 127 범위에서 적절한 값들
            std::uniform_int_distribution<int8_t> dist(-100, 100);
            int8_t* data = reinterpret_cast<int8_t*>(buf);
            for (size_t i = 0; i < size; ++i) {
                data[i] = dist(rng);
            }
            break;
        }
        case QNN_DATATYPE_UFIXED_POINT_8: {
            // UINT8: 0 ~ 255 범위에서 적절한 값들  
            std::uniform_int_distribution<uint8_t> dist(0, 200);
            for (size_t i = 0; i < size; ++i) {
                buf[i] = dist(rng);
            }
            break;
        }
        case QNN_DATATYPE_SFIXED_POINT_16: {
            // INT16: -32768 ~ 32767 범위에서 적절한 값들
            std::uniform_int_distribution<int16_t> dist(-10000, 10000);
            int16_t* data = reinterpret_cast<int16_t*>(buf);
            for (size_t i = 0; i < size / 2; ++i) {
                data[i] = dist(rng);
            }
            break;
        }
        case QNN_DATATYPE_UFIXED_POINT_16: {
            // UINT16: 0 ~ 65535 범위에서 적절한 값들
            std::uniform_int_distribution<uint16_t> dist(0, 20000);
            uint16_t* data = reinterpret_cast<uint16_t*>(buf);
            for (size_t i = 0; i < size / 2; ++i) {
                data[i] = dist(rng);
            }
            break;
        }
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_FLOAT_32:
        default:
            // Floating point or other types: use original random fill
            randomFill(buf, size);
            break;
    }
}

// Validate MatMul configuration compatibility
static const MatMulConfig* validateMatMulConfig(Qnn_DataType_t in0_type, Qnn_DataType_t in1_type) {
    for (size_t i = 0; i < NUM_SUPPORTED_CONFIGS; ++i) {
        const MatMulConfig& config = SUPPORTED_MATMUL_CONFIGS[i];
        if (config.in0_type == in0_type && config.in1_type == in1_type) {
            return &config;
        }
    }
    return nullptr;
}

// Print all supported configurations
static void printSupportedConfigurations() {
    printf("\n=== NPU MatMul Supported Configurations ===\n");
    printf("%-15s %-20s %-20s %-20s\n", "Config", "Input A (in[0])", "Input B (in[1])", "Output (out[0])");
    printf("%-15s %-20s %-20s %-20s\n", "------", "-------------", "-------------", "--------------");
    
    for (size_t i = 0; i < NUM_SUPPORTED_CONFIGS; ++i) {
        const MatMulConfig& config = SUPPORTED_MATMUL_CONFIGS[i];
        
        // Find type names
        const char* in0_name = "unknown";
        const char* in1_name = "unknown"; 
        const char* out_name = "unknown";
        
        for (size_t j = 0; j < NUM_TYPES; ++j) {
            if (kTypes[j].dtype == config.in0_type) in0_name = kTypes[j].name;
            if (kTypes[j].dtype == config.in1_type) in1_name = kTypes[j].name;
            if (kTypes[j].dtype == config.out_type) out_name = kTypes[j].name;
        }
        
        printf("%-15s %-20s %-20s %-20s\n", config.config_name, in0_name, in1_name, out_name);
    }
    printf("===========================================\n\n");
}

//==============================================================================
// SECTION 4: QNN INTERFACE AND SYSTEM MANAGEMENT
//==============================================================================

// Dynamic loader for QNN interface
using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

// Load QNN interface from shared library
static const QNN_INTERFACE_VER_TYPE * loadInterface(void ** libHandleOut, const char * libPath) {
    std::string fullPath = std::string("/data/local/tmp/") + libPath;
    void * h = dlopen(fullPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        std::cerr << "dlopen failed for " << fullPath << ": " << dlerror() << std::endl;
        return nullptr;
    }
    auto getProviders = reinterpret_cast<GetProvidersFn>(dlsym(h, "QnnInterface_getProviders"));
    if (!getProviders) {
        std::cerr << "Cannot resolve QnnInterface_getProviders" << std::endl;
        dlclose(h);
        return nullptr;
    }
    const QnnInterface_t ** providers = nullptr; uint32_t n = 0;
    if (getProviders(&providers, &n) != QNN_SUCCESS || !providers || n == 0) {
        std::cerr << "No interface providers" << std::endl;
        dlclose(h);
        return nullptr;
    }
    // Select the first provider that matches current core API major version.
    const QNN_INTERFACE_VER_TYPE * chosen = nullptr;
    for (uint32_t i = 0; i < n; ++i) {
        if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR) {
            chosen = &providers[i]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }
    if (!chosen) {
        std::cerr << "No compatible QNN interface version found" << std::endl;
        dlclose(h);
        return nullptr;
    }
    *libHandleOut = h;
    return chosen;
}

//==============================================================================
// SECTION 5: QNN CONTEXT MANAGEMENT
//==============================================================================

// Quantization memory manager for safe memory handling
struct QuantizationMemory {
    std::vector<std::vector<Qnn_ScaleOffset_t>> axisScaleOffsetStorages;
    
    // Add new axis scale offset storage and return pointer
    Qnn_ScaleOffset_t* addAxisScaleOffsetStorage(uint32_t numChannels, float baseScale, int32_t baseOffset) {
        axisScaleOffsetStorages.emplace_back(numChannels);
        auto& storage = axisScaleOffsetStorages.back();
        
        // Per-channel quantization: use different scale/offset for each channel
        // This provides better quantization accuracy than using identical values
        std::mt19937 rng(42); // Fixed seed for reproducible results
        std::uniform_real_distribution<float> scaleDist(baseScale * 0.8f, baseScale * 1.2f);
        std::uniform_int_distribution<int32_t> offsetDist(baseOffset - 10, baseOffset + 10);
        
        for (uint32_t i = 0; i < numChannels; ++i) {
            // Generate per-channel parameters with slight variation
            storage[i].scale = scaleDist(rng);
            storage[i].offset = offsetDist(rng);
        }
        return storage.data();
    }
    
    // Alternative: Add axis scale offset storage with user-provided arrays
    Qnn_ScaleOffset_t* addAxisScaleOffsetStorageFromArrays(const std::vector<float>& scales, 
                                                           const std::vector<int32_t>& offsets) {
        if (scales.size() != offsets.size()) {
            std::cerr << "ERROR: scales and offsets arrays must have the same size" << std::endl;
            return nullptr;
        }
        
        uint32_t numChannels = scales.size();
        axisScaleOffsetStorages.emplace_back(numChannels);
        auto& storage = axisScaleOffsetStorages.back();
        
        for (uint32_t i = 0; i < numChannels; ++i) {
            storage[i].scale = scales[i];
            storage[i].offset = offsets[i];
        }
        return storage.data();
    }
    
    void clear() {
        axisScaleOffsetStorages.clear();
    }
};

// QNN Context structure for unified resource management
struct QnnContext {
    void* libHandle;
    const QNN_INTERFACE_VER_TYPE* iface;
    Qnn_LogHandle_t logger;
    Qnn_BackendHandle_t backend;
    Qnn_DeviceHandle_t device;
    Qnn_ContextHandle_t ctx;
    std::string backend_name;
    QuantizationMemory quantMem;  // Add quantization memory manager
    
    QnnContext() : libHandle(nullptr), iface(nullptr), logger(nullptr), 
                   backend(nullptr), device(nullptr), ctx(nullptr) {}
};

// Initialize QNN (unified function for all benchmark modes)
static bool initializeQNN(QnnContext& qnn_ctx) {
    // Try HTP first, then CPU
    const char* backends[] = {"libQnnHtp.so", "libQnnCpu.so"};
    const char* backend_names[] = {"HTP", "CPU"};
    
    // Try loading backends
    for (int i = 0; i < 2; i++) {
        qnn_ctx.iface = loadInterface(&qnn_ctx.libHandle, backends[i]);
        if (qnn_ctx.iface) {
            qnn_ctx.backend_name = backend_names[i];
            break;
        }
    }
    
    if (!qnn_ctx.iface) {
        std::cerr << "ERROR: Failed to load any QNN backend" << std::endl;
        return false;
    }
    
    // Create logger
    if (qnn_ctx.iface->logCreate(nullptr, QNN_LOG_LEVEL_ERROR, &qnn_ctx.logger) != QNN_SUCCESS) {
        std::cerr << "ERROR: Failed to create logger" << std::endl;
        return false;
    }
    
    // Create backend
    if (qnn_ctx.iface->backendCreate(qnn_ctx.logger, nullptr, &qnn_ctx.backend) != QNN_SUCCESS) {
        std::cerr << "ERROR: Failed to create backend" << std::endl;
        return false;
    }
    
    // Create device (HTP vs CPU specific)
    Qnn_ErrorHandle_t deviceResult = QNN_SUCCESS;
    if (qnn_ctx.backend_name == "HTP") {
        // HTP device creation with platform info
        const QnnDevice_PlatformInfo_t* p_info = nullptr;
        deviceResult = qnn_ctx.iface->deviceGetPlatformInfo(nullptr, &p_info);
        
        if (deviceResult == QNN_SUCCESS && p_info) {
            QnnDevice_HardwareDeviceInfo_t* infos = p_info->v1.hwDevices;
            QnnHtpDevice_OnChipDeviceInfoExtension_t chipinfo = {};
            if (p_info->v1.numHwDevices > 0) {
                QnnDevice_DeviceInfoExtension_t devinfo = infos[0].v1.deviceInfoExtension;
                chipinfo = devinfo->onChipDevice;
            }
            
            QnnHtpDevice_CustomConfig_t soc_customconfig;
            soc_customconfig.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
            soc_customconfig.socModel = chipinfo.socModel;
            QnnDevice_Config_t soc_devconfig;
            soc_devconfig.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
            soc_devconfig.customConfig = &soc_customconfig;
            
            const QnnDevice_Config_t* p_deviceconfig[] = {&soc_devconfig, nullptr};
            deviceResult = qnn_ctx.iface->deviceCreate(qnn_ctx.logger, p_deviceconfig, &qnn_ctx.device);
            qnn_ctx.iface->deviceFreePlatformInfo(nullptr, p_info);
        } else {
            deviceResult = qnn_ctx.iface->deviceCreate(qnn_ctx.logger, nullptr, &qnn_ctx.device);
        }
    } else {
        deviceResult = qnn_ctx.iface->deviceCreate(qnn_ctx.logger, nullptr, &qnn_ctx.device);
    }
    
    if (deviceResult != QNN_SUCCESS && deviceResult != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
        std::cout << "WARNING: Device creation failed, continuing without device" << std::endl;
    }
    
    // Create context
    if (qnn_ctx.iface->contextCreate(qnn_ctx.backend, qnn_ctx.device, nullptr, &qnn_ctx.ctx) != QNN_SUCCESS) {
        std::cerr << "ERROR: Failed to create context" << std::endl;
        return false;
    }
    
    return true;
}

// Cleanup QNN resources (unified function)
static void cleanupQNN(QnnContext& qnn_ctx) {
    if (qnn_ctx.iface) {
        if (qnn_ctx.ctx) qnn_ctx.iface->contextFree(qnn_ctx.ctx, nullptr);
        if (qnn_ctx.device) qnn_ctx.iface->deviceFree(qnn_ctx.device);
        if (qnn_ctx.backend) qnn_ctx.iface->backendFree(qnn_ctx.backend);
        if (qnn_ctx.logger) qnn_ctx.iface->logFree(qnn_ctx.logger);
    }
    if (qnn_ctx.libHandle) {
        dlclose(qnn_ctx.libHandle);
    }
    // Clear quantization memory
    qnn_ctx.quantMem.clear();
}

//==============================================================================
// SECTION 6: BENCHMARK PATTERN FUNCTIONS
//==============================================================================

// Benchmark result structure
struct BenchmarkResult {
    int pattern;
    int seq_len;
    double avg_time_ms;
    double avg_gflops;
    bool success;
};

// Get pattern name for display
static const char* get_pattern_name(int pattern) {
    switch(pattern) {
        case 0: return "Attention [seq x 4096] x [4096 x seq]";
        case 1: return "Linear [seq x 4096] x [4096 x 4096]";
        case 2: return "FFN_Up [seq x 4096] x [4096 x 11008]";
        case 3: return "FFN_Down [seq x 11008] x [11008 x 4096]";
        default: return "Unknown";
    }
}

// Get matrix dimensions for LLM patterns
static void get_matrix_dims(int pattern, int seq_len, uint32_t* m, uint32_t* k, uint32_t* n) {
    switch(pattern) {
        case 0: // [seq x 4096] x [4096 x seq]
            *m = seq_len; *k = 4096; *n = seq_len;
            break;
        case 1: // [seq x 4096] x [4096 x 4096]
            *m = seq_len; *k = 4096; *n = 4096;
            break;
        case 2: // [seq x 4096] x [4096 x 11008]
            *m = seq_len; *k = 4096; *n = 11008;
            break;
        case 3: // [seq x 11008] x [11008 x 4096]
            *m = seq_len; *k = 11008; *n = 4096;
            break;
        default:
            *m = *k = *n = 0;
            break;
    }
}

//==============================================================================
// SECTION 7: TENSOR CREATION UTILITIES
//==============================================================================

// Tensor type enumeration for clear identification
enum class TensorRole {
    INPUT_A,
    INPUT_B, 
    OUTPUT
};

// Check if datatype requires quantization (excludes fp16, fp32)
static bool requiresQuantization(Qnn_DataType_t dataType) {
    return dataType != QNN_DATATYPE_FLOAT_16 && dataType != QNN_DATATYPE_FLOAT_32;
}

// Create quantization parameters for scale_offset encoding (symmetric)
static Qnn_QuantizeParams_t createScaleOffsetQuantization(float scale = 1.0f, int32_t offset = 0) {
    Qnn_QuantizeParams_t quantParams = {};
    quantParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    quantParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    quantParams.scaleOffsetEncoding.scale = scale;
    quantParams.scaleOffsetEncoding.offset = offset;
    return quantParams;
}

// Create quantization parameters for axis_scale_offset encoding (symmetric, per-channel)
static Qnn_QuantizeParams_t createAxisScaleOffsetQuantization(QuantizationMemory& quantMem, 
                                                             int32_t axis, uint32_t numChannels, 
                                                             float scale = 1.0f, int32_t offset = 0) {
    Qnn_QuantizeParams_t quantParams = {};
    quantParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    quantParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
    quantParams.axisScaleOffsetEncoding.axis = axis;
    quantParams.axisScaleOffsetEncoding.numScaleOffsets = numChannels;
    
    // Use memory manager for safe allocation
    quantParams.axisScaleOffsetEncoding.scaleOffset = 
        quantMem.addAxisScaleOffsetStorage(numChannels, scale, offset);
    
    return quantParams;
}

// Create appropriate quantization parameters based on tensor role and datatype
static Qnn_QuantizeParams_t createQuantizationParams(QuantizationMemory& quantMem, QuantizationType quantType,
                                                     Qnn_DataType_t dataType, uint32_t* dimensions, int32_t axis = 3) {
    // No quantization for floating point types (unless explicitly overridden)
    if (!requiresQuantization(dataType) && quantType != QuantizationType::UNDEFINED) {
        Qnn_QuantizeParams_t quantParams = {};
        quantParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        quantParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        quantParams.scaleOffsetEncoding = {0.0f, 0};
        return quantParams;
    }
    
    // Apply quantization based on user-specified type
    switch (quantType) {
        case QuantizationType::UNDEFINED:
            // Explicitly undefined quantization (user choice)
            {
                Qnn_QuantizeParams_t quantParams = {};
                quantParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
                quantParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
                quantParams.scaleOffsetEncoding = {0.0f, 0};
                return quantParams;
            }
            
        case QuantizationType::SCALE_OFFSET:
            return createScaleOffsetQuantization(1.0f, 0);
            
        case QuantizationType::AXIS_SCALE_OFFSET:
            // Use specified axis and corresponding dimension
            return createAxisScaleOffsetQuantization(quantMem, axis, dimensions[axis], 1.0f, 0);
            
        default:
            // Fallback: undefined quantization
            Qnn_QuantizeParams_t quantParams = {};
            quantParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
            quantParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
            quantParams.scaleOffsetEncoding = {0.0f, 0};
            return quantParams;
    }
}

// Unified tensor creation function with quantization support
static Qnn_Tensor_t createTensor(QuantizationMemory& quantMem, uint32_t id, const char* name, 
                                TensorRole role, Qnn_DataType_t dataType, uint32_t* dimensions,
                                QuantizationType quantType, int32_t axis = 3) {
    Qnn_TensorType_t tensorType = (role == TensorRole::OUTPUT) ? 
                                   QNN_TENSOR_TYPE_APP_READ : 
                                   QNN_TENSOR_TYPE_APP_WRITE;
    
    // Create appropriate quantization parameters
    Qnn_QuantizeParams_t quantParams = createQuantizationParams(quantMem, quantType, dataType, dimensions, axis);
    
    return Qnn_Tensor_t {
        .version = QNN_TENSOR_VERSION_1,
        .v1 = {
            .id = id,
            .name = name,
            .type = tensorType,
            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType = dataType,
            .quantizeParams = quantParams,
            .rank = 4,
            .dimensions = dimensions,
            .memType = QNN_TENSORMEMTYPE_RAW,
            .clientBuf = {.data = nullptr, .dataSize = 0}
        }
    };
}

// Create all tensors for MatMul operation with quantization support
static void createMatMulTensors(QnnContext& qnn_ctx, Qnn_GraphHandle_t graph,
                               const DTypeDesc& dt0, const DTypeDesc& dt1, const DTypeDesc& dtOut,
                               uint32_t* dimA_4d, uint32_t* dimB_4d, uint32_t* dimC_4d,
                               Qnn_Tensor_t& tenA, Qnn_Tensor_t& tenB, Qnn_Tensor_t& tenC,
                               QuantizationType quantTypeA, QuantizationType quantTypeB, QuantizationType quantTypeC) {
    // Create input tensor A with specified quantization
    tenA = createTensor(qnn_ctx.quantMem, 0, "tensor_A", TensorRole::INPUT_A, dt0.dtype, dimA_4d, quantTypeA, 2); // axis=2 for M dimension
    CHECK_QNN(qnn_ctx.iface->tensorCreateGraphTensor(graph, &tenA));
    
    // Create input tensor B with specified quantization
    tenB = createTensor(qnn_ctx.quantMem, 1, "tensor_B", TensorRole::INPUT_B, dt1.dtype, dimB_4d, quantTypeB, 3); // axis=3 for N dimension
    CHECK_QNN(qnn_ctx.iface->tensorCreateGraphTensor(graph, &tenB));
    
    // Create output tensor C with specified quantization
    tenC = createTensor(qnn_ctx.quantMem, 2, "tensor_C", TensorRole::OUTPUT, dtOut.dtype, dimC_4d, quantTypeC, 3); // axis=3 for N dimension
    CHECK_QNN(qnn_ctx.iface->tensorCreateGraphTensor(graph, &tenC));
    
}

//==============================================================================
// SECTION 8: CORE BENCHMARK EXECUTION FUNCTION
//==============================================================================

// Unified benchmark execution function (each experiment uses independent context)
static BenchmarkResult runMatMulBenchmark(uint32_t M, uint32_t K, uint32_t N,
                                         const DTypeDesc& dt0, const DTypeDesc& dt1, const DTypeDesc& dtOut,
                                         int iterations, 
                                         QuantizationType quantTypeA = QuantizationType::AXIS_SCALE_OFFSET,
                                         QuantizationType quantTypeB = QuantizationType::AXIS_SCALE_OFFSET,
                                         QuantizationType quantTypeC = QuantizationType::AXIS_SCALE_OFFSET,
                                         int pattern = -1, int seq_len = 0) {
    BenchmarkResult result = {pattern, seq_len, 0.0, 0.0, false};
    
    // 각 실험마다 새로운 QNN context 생성 (리소스 충돌 방지)
    QnnContext qnn_ctx;
    
    try {
        // Initialize QNN for this specific benchmark
        if (!initializeQNN(qnn_ctx)) {
            throw std::runtime_error("Failed to initialize QNN for benchmark");
        }
        
        // Create graph with backend-specific configuration
        Qnn_GraphHandle_t graph = nullptr;
        std::string graph_name = pattern >= 0 ? ("pattern_" + std::to_string(pattern) + "_" + std::to_string(seq_len)) : "matmul";
        
        if (qnn_ctx.backend_name == "HTP") {
            // HTP graph with VTCM and HVX config
            QnnHtpGraph_CustomConfig_t vtcmConfig;
            vtcmConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
            vtcmConfig.vtcmSizeInMB = 8;
            
            QnnHtpGraph_CustomConfig_t hvxConfig;
            hvxConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
            hvxConfig.numHvxThreads = 8;
            
            QnnGraph_Config_t vtcmGraphConfig;
            vtcmGraphConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            vtcmGraphConfig.customConfig = &vtcmConfig;
            
            QnnGraph_Config_t hvxGraphConfig;
            hvxGraphConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            hvxGraphConfig.customConfig = &hvxConfig;
            
            const QnnGraph_Config_t *pGraphConfig[] = {&vtcmGraphConfig, &hvxGraphConfig, nullptr};
            CHECK_QNN(qnn_ctx.iface->graphCreate(qnn_ctx.ctx, graph_name.c_str(), pGraphConfig, &graph));
        } else {
            CHECK_QNN(qnn_ctx.iface->graphCreate(qnn_ctx.ctx, graph_name.c_str(), nullptr, &graph));
        }
        
        // Prepare buffers
        size_t bytesA = static_cast<size_t>(M) * K * dt0.bytes;
        size_t bytesB = static_cast<size_t>(K) * N * dt1.bytes;
        size_t bytesC = static_cast<size_t>(M) * N * dtOut.bytes;
        
        std::vector<uint8_t> bufA(bytesA);
        std::vector<uint8_t> bufB(bytesB);
        std::vector<uint8_t> bufC(bytesC);
        
        fillQuantizedData(bufA.data(), bytesA, dt0.dtype);
        fillQuantizedData(bufB.data(), bytesB, dt1.dtype);
        
        // Create 4D tensors using unified creation function
        uint32_t dimA_4d[4] = {1, 1, M, K};
        uint32_t dimB_4d[4] = {1, 1, K, N};
        uint32_t dimC_4d[4] = {1, 1, M, N};
        
        Qnn_Tensor_t tenA, tenB, tenC;
        createMatMulTensors(qnn_ctx, graph, dt0, dt1, dtOut, dimA_4d, dimB_4d, dimC_4d, tenA, tenB, tenC,
                           quantTypeA, quantTypeB, quantTypeC);
        
        // Create MatMul op
        Qnn_Param_t params[2] = {};
        params[0].paramType = QNN_PARAMTYPE_SCALAR;
        params[0].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
        params[0].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
        params[0].scalarParam.bool8Value = 0;
        params[1].paramType = QNN_PARAMTYPE_SCALAR;
        params[1].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
        params[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
        params[1].scalarParam.bool8Value = 0;
        
        Qnn_Tensor_t inputsNode[2] = {tenA, tenB};
        Qnn_Tensor_t outputs[1] = {tenC};
        
        Qnn_OpConfig_t op{};
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "matmul_op";
        op.v1.packageName = "qti.aisw";
        op.v1.typeName = QNN_OP_MAT_MUL;
        op.v1.numOfParams = 2; op.v1.params = params;
        op.v1.numOfInputs = 2; op.v1.inputTensors = inputsNode;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = outputs;
        
        CHECK_QNN(qnn_ctx.iface->graphAddNode(graph, op));
        CHECK_QNN(qnn_ctx.iface->graphFinalize(graph, nullptr, nullptr));
        
        // Setup execution tensors
        Qnn_Tensor_t inputsExec[2] = {tenA, tenB};
        Qnn_Tensor_t outputsExec[1] = {tenC};
        
        inputsExec[0].v1.clientBuf.data = bufA.data();
        inputsExec[0].v1.clientBuf.dataSize = bytesA;
        inputsExec[1].v1.clientBuf.data = bufB.data();
        inputsExec[1].v1.clientBuf.dataSize = bytesB;
        outputsExec[0].v1.clientBuf.data = bufC.data();
        outputsExec[0].v1.clientBuf.dataSize = bytesC;
        
        // Benchmark execution
        CHECK_QNN(qnn_ctx.iface->graphExecute(graph, inputsExec, 2, outputsExec, 1, nullptr, nullptr)); // warmup
        
        double tot = 0.0;
        for (int i = 0; i < iterations; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            CHECK_QNN(qnn_ctx.iface->graphExecute(graph, inputsExec, 2, outputsExec, 1, nullptr, nullptr));
            auto t1 = std::chrono::high_resolution_clock::now();
            tot += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        
        result.avg_time_ms = tot / iterations;
        result.avg_gflops = (2.0 * M * K * N) / (result.avg_time_ms * 1e6);
        result.success = true;
        
    } catch (const std::exception& e) {
        result.success = false;
    } catch (...) {
        result.success = false;
    }
    
    // 각 실험 후 QNN context 정리 (리소스 해제)
    cleanupQNN(qnn_ctx);
    
    return result;
}

//==============================================================================
// SECTION 9: MAIN FUNCTION WITH CLI PARSING AND EXECUTION MODES
//==============================================================================

int main(int argc, char ** argv) {
    //--------------------------------------------------------------------------
    // 9.1: Default Parameters and Variables
    //--------------------------------------------------------------------------
    // default parameters
    uint32_t M = 128, K = 4096, N = 4096;
    std::string dtypeStr0 = "fp32";  // Input tensor A datatype (기본값을 fp32로 유지)
    std::string dtypeStr1 = "fp32";  // Input tensor B datatype (기본값을 fp32로 유지)
    std::string dtypeStrOut = "";    // Output tensor datatype (optional, derived from inputs)
    int iterations       = 10;

    // Pattern mode parameters (new)
    bool pattern_mode = false;
    bool all_patterns_mode = false;  // 모든 패턴 실행
    int pattern = -1;  // -1 means no pattern specified
    int max_seq_len = 4096;  // Maximum sequence length for pattern mode

    // Quantization parameters (new)
    QuantizationType quantTypeA = QuantizationType::AXIS_SCALE_OFFSET;  // Default: axis_scale_offset
    QuantizationType quantTypeB = QuantizationType::AXIS_SCALE_OFFSET;  // Default: axis_scale_offset  
    QuantizationType quantTypeC = QuantizationType::AXIS_SCALE_OFFSET;  // Default: axis_scale_offset

    //--------------------------------------------------------------------------
    // 9.2: Command Line Interface (CLI) Parsing
    //--------------------------------------------------------------------------
    // simple CLI parsing with enhanced options
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            M = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            K = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-t0") == 0 && i + 1 < argc) {
            dtypeStr0 = argv[++i];
        } else if (strcmp(argv[i], "-t1") == 0 && i + 1 < argc) {
            dtypeStr1 = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            // Legacy option: set both inputs to same type
            dtypeStr0 = dtypeStr1 = argv[++i];
        } else if (strcmp(argv[i], "-to") == 0 && i + 1 < argc) {
            dtypeStrOut = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            // Pattern mode: -p <pattern_id>
            pattern = std::stoi(argv[++i]);
            if (pattern < 0 || pattern > 3) {
                std::cerr << "Error: Pattern must be 0-3" << std::endl;
                return 1;
            }
            pattern_mode = true;
        } else if (strcmp(argv[i], "-a") == 0) {
            // All patterns mode: -a
            all_patterns_mode = true;
            pattern_mode = false;  // Disable single pattern mode
        } else if (strcmp(argv[i], "-qa") == 0 && i + 1 < argc) {
            // Input A quantization type: -qa <type>
            quantTypeA = parseQuantizationType(argv[++i]);
        } else if (strcmp(argv[i], "-qb") == 0 && i + 1 < argc) {
            // Input B quantization type: -qb <type>
            quantTypeB = parseQuantizationType(argv[++i]);
        } else if (strcmp(argv[i], "-qc") == 0 && i + 1 < argc) {
            // Output C quantization type: -qc <type>
            quantTypeC = parseQuantizationType(argv[++i]);
        } else if (strcmp(argv[i], "-maxseq") == 0 && i + 1 < argc) {
            // Maximum sequence length for pattern mode: -maxseq <seq>
            max_seq_len = std::stoi(argv[++i]);
            // Validate that max_seq_len is one of the supported values
            std::vector<int> valid_seq_lens = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
            bool valid = false;
            for (int seq : valid_seq_lens) {
                if (max_seq_len == seq) {
                    valid = true;
                    break;
                }
            }
            if (!valid) {
                std::cerr << "Error: max_seq_len must be one of: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Enhanced MatMulBenchmark with individual input datatype support\n"
                      << "Usage: MatMulBenchmark [options]\n\n"
                      << "Matrix Configuration (Manual Mode):\n"
                      << "  -m <M>        : Matrix A rows (default: 128)\n"
                      << "  -k <K>        : Matrix A cols / Matrix B rows (default: 4096)\n"
                      << "  -n <N>        : Matrix B cols (default: 4096)\n"
                      << "                  Computes C[M,N] = A[M,K] × B[K,N]\n\n"
                      << "Pattern Mode (LLM Patterns):\n"
                      << "  -p <pattern>  : Use predefined LLM patterns (overrides -m/-k/-n)\n"
                      << "                  0: Attention [seq x 4096] x [4096 x seq]\n"
                      << "                  1: Linear [seq x 4096] x [4096 x 4096]\n"
                      << "                  2: FFN_Up [seq x 4096] x [4096 x 11008]\n"
                      << "                  3: FFN_Down [seq x 11008] x [11008 x 4096]\n"
                      << "                  Pattern mode tests seq_len from 1 to maxseq\n"
                      << "  -a            : Test all patterns and generate summary report\n"
                      << "  -maxseq <seq> : Maximum sequence length for pattern mode (default: 4096)\n"
                      << "                  Must be one of: 1,2,4,8,16,32,64,128,256,512,1024,2048,4096\n\n"
                      << "Datatype Configuration:\n"
                      << "  -t0 <type>    : Input tensor A (in[0]) datatype\n"
                      << "  -t1 <type>    : Input tensor B (in[1]) datatype\n"
                      << "  -t <type>     : Set both input tensors to same type (legacy)\n"
                      << "  -to <type>    : Output tensor datatype (optional)\n\n"
                      << "Supported datatypes: fp16, fp32, int16, int8, uint8, uint16\n\n"
                      << "Quantization Configuration:\n"
                      << "  -qa <type>    : Input tensor A quantization (default: axis)\n"
                      << "  -qb <type>    : Input tensor B quantization (default: axis)\n"
                      << "  -qc <type>    : Output tensor C quantization (default: axis)\n"
                      << "                  Types: 'none' (no quantization), 'scale' (per-tensor), 'axis' (per-channel)\n\n"
                      << "Execution Configuration:\n"
                      << "  -i <iter>     : Number of iterations (default: 10)\n\n"
                      << "Examples:\n"
                      << "  # Manual matrix size with scale quantization\n"
                      << "  ./MatMulBenchmark -m 1024 -k 2048 -n 2048 -t int16 -qa scale -qb scale -qc scale\n"
                      << "  # Pattern mode: FFN_Up with mixed quantization\n"
                      << "  ./MatMulBenchmark -p 2 -t0 int16 -t1 int8 -qa axis -qb scale\n"
                      << "  # All patterns up to seq=1024 with axis quantization\n"
                      << "  ./MatMulBenchmark -a -t int8 -maxseq 1024 -qa axis -qb axis -qc axis\n"
                      << "  # FP16 with no quantization (explicit)\n"
                      << "  ./MatMulBenchmark -t fp16 -qa none -qb none -qc none\n"
                      << "  # Mixed: FP16 input, INT8 weights with different quantization\n"
                      << "  ./MatMulBenchmark -t0 fp16 -t1 int8 -qa none -qb axis -qc scale\n"
                      << std::endl;
            return 0;
        }
    }

    //--------------------------------------------------------------------------
    // 9.3: All Patterns Benchmark Mode (-a option)
    //--------------------------------------------------------------------------
    if (all_patterns_mode) {
        // All patterns benchmark mode (like matmul-benchmark.cpp)
        std::cout << "=================================================================\n";
        std::cout << "Comprehensive NPU MatMul Benchmark - All LLM Patterns\n";
        std::cout << "Input A dtype: " << dtypeStr0 << ", Input B dtype: " << dtypeStr1 << "\n";
        std::cout << "Input A quant: " << getQuantizationTypeName(quantTypeA) << ", ";
        std::cout << "Input B quant: " << getQuantizationTypeName(quantTypeB) << ", ";
        std::cout << "Output C quant: " << getQuantizationTypeName(quantTypeC) << "\n";
        std::cout << "Iterations per test: " << iterations << "\n";
        std::cout << "=================================================================\n\n";
        
        const auto & dt0 = findType(dtypeStr0);
        const auto & dt1 = findType(dtypeStr1);
        
        // Output datatype determination
        const DTypeDesc * dtOut = &dt0;
        if (!dtypeStrOut.empty()) {
            dtOut = &findType(dtypeStrOut);
        } else {
            if (dt0.bytes >= dt1.bytes) {
                dtOut = &dt0;
            } else {
                dtOut = &dt1;
            }
            if ((dt0.dtype == QNN_DATATYPE_FLOAT_16 || dt0.dtype == QNN_DATATYPE_FLOAT_32) ||
                (dt1.dtype == QNN_DATATYPE_FLOAT_16 || dt1.dtype == QNN_DATATYPE_FLOAT_32)) {
                if (dt0.dtype == QNN_DATATYPE_FLOAT_32 || dt1.dtype == QNN_DATATYPE_FLOAT_32) {
                    dtOut = &findType("fp32");
                } else {
                    dtOut = &findType("fp16");
                }
            }
        }
        
        std::vector<int> seq_lengths = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        
        // Filter seq_lengths to only include values up to max_seq_len
        std::vector<int> filtered_seq_lengths;
        for (int seq : seq_lengths) {
            if (seq <= max_seq_len) {
                filtered_seq_lengths.push_back(seq);
            }
        }
        seq_lengths = filtered_seq_lengths;
        
        std::vector<int> patterns = {0, 1, 2, 3};
        std::vector<BenchmarkResult> all_results;
        
        int total_tests = patterns.size() * seq_lengths.size();
        int current_test = 0;
        
        printf("Running %d total test cases (up to seq=%d)...\n\n", total_tests, max_seq_len);
        
        for (int test_pattern : patterns) {
            printf("=== %s ===\n", get_pattern_name(test_pattern));
            printf("%-8s %-12s %-12s %-8s\n", "Seq", "Time(ms)", "GFLOPS", "Status");
            printf("--------------------------------------------\n");
            
            for (int seq_len : seq_lengths) {
                current_test++;
                
                // Get matrix dimensions for this pattern and seq_len
                uint32_t test_M, test_K, test_N;
                get_matrix_dims(test_pattern, seq_len, &test_M, &test_K, &test_N);
                
                if (test_M == 0 || test_K == 0 || test_N == 0) {
                    printf("%-8d %-12s %-12s %-8s\n", seq_len, "INVALID", "INVALID", "SKIP");
                    BenchmarkResult result = {test_pattern, seq_len, 0.0, 0.0, false};
                    all_results.push_back(result);
                    continue;
                }
                
                printf("\r[%3d/%3d] Testing seq=%d... ", current_test, total_tests, seq_len);
                fflush(stdout);
                
                // Run the benchmark with pattern-specific dimensions using unified function
                BenchmarkResult result = runMatMulBenchmark(test_M, test_K, test_N, 
                                                           dt0, dt1, *dtOut, iterations, 
                                                           quantTypeA, quantTypeB, quantTypeC,
                                                           test_pattern, seq_len);
                all_results.push_back(result);
                
                printf("\r%-8d ", seq_len);
                if (result.success) {
                    printf("%-12.2f %-12.2f %-8s\n", result.avg_time_ms, result.avg_gflops, "OK");
                } else {
                    printf("%-12s %-12s %-8s\n", "FAILED", "FAILED", "SKIP");
                }
            }
            printf("\n");
        }
        
        // Generate comprehensive summary reports
        printf("\n=================================================================\n");
        printf("COMPREHENSIVE SUMMARY REPORT - GFLOPS\n");
        printf("=================================================================\n");
        
        for (int summary_pattern : patterns) {
            printf("\n%s:\n", get_pattern_name(summary_pattern));
            printf("%-8s %-12s\n", "Seq", "GFLOPS");
            printf("---------------------\n");
            
            for (int seq_len : seq_lengths) {
                auto it = std::find_if(all_results.begin(), all_results.end(),
                    [=](const BenchmarkResult& r) {
                        return r.pattern == summary_pattern && r.seq_len == seq_len;
                    });
                
                if (it != all_results.end() && it->success) {
                    printf("%-8d %-12.2f\n", seq_len, it->avg_gflops);
                } else {
                    printf("%-8d %-12s\n", seq_len, "FAILED");
                }
            }
        }
        
        printf("\n=================================================================\n");
        printf("COMPREHENSIVE SUMMARY REPORT - TIME (ms)\n");
        printf("=================================================================\n");
        
        for (int summary_pattern : patterns) {
            printf("\n%s:\n", get_pattern_name(summary_pattern));
            printf("%-8s %-12s\n", "Seq", "Time(ms)");
            printf("---------------------\n");
            
            for (int seq_len : seq_lengths) {
                auto it = std::find_if(all_results.begin(), all_results.end(),
                    [=](const BenchmarkResult& r) {
                        return r.pattern == summary_pattern && r.seq_len == seq_len;
                    });
                
                if (it != all_results.end() && it->success) {
                    printf("%-8d %-12.2f\n", seq_len, it->avg_time_ms);
                } else {
                    printf("%-8d %-12s\n", seq_len, "FAILED");
                }
            }
        }
        
        printf("\n=================================================================\n");
        printf("All Patterns Benchmark Complete\n");
        printf("=================================================================\n");
        
        return 0;
    }

    //--------------------------------------------------------------------------
    // 9.4: Single Pattern Benchmark Mode (-p option)  
    //--------------------------------------------------------------------------
    if (pattern_mode) {
        // Pattern-based benchmark mode (like matmul-benchmark.cpp)
        std::cout << "=================================================================\n";
        std::cout << "Pattern-based MatMul Benchmark for NPU\n";
        std::cout << "Pattern: " << get_pattern_name(pattern) << "\n";
        std::cout << "Input A dtype: " << dtypeStr0 << ", Input B dtype: " << dtypeStr1 << "\n";
        std::cout << "Input A quant: " << getQuantizationTypeName(quantTypeA) << ", ";
        std::cout << "Input B quant: " << getQuantizationTypeName(quantTypeB) << ", ";
        std::cout << "Output C quant: " << getQuantizationTypeName(quantTypeC) << "\n";
        std::cout << "Iterations per test: " << iterations << "\n";
        std::cout << "=================================================================\n\n";
        
        const auto & dt0 = findType(dtypeStr0);
        const auto & dt1 = findType(dtypeStr1);
        
        // Output datatype: if not specified, use higher precision of the two inputs  
        const DTypeDesc * dtOut = &dt0;
        if (!dtypeStrOut.empty()) {
            dtOut = &findType(dtypeStrOut);
        } else {
            if (dt0.bytes >= dt1.bytes) {
                dtOut = &dt0;
            } else {
                dtOut = &dt1;
            }
            // For mixed precision, prefer floating point over fixed point
            if ((dt0.dtype == QNN_DATATYPE_FLOAT_16 || dt0.dtype == QNN_DATATYPE_FLOAT_32) ||
                (dt1.dtype == QNN_DATATYPE_FLOAT_16 || dt1.dtype == QNN_DATATYPE_FLOAT_32)) {
                if (dt0.dtype == QNN_DATATYPE_FLOAT_32 || dt1.dtype == QNN_DATATYPE_FLOAT_32) {
                    dtOut = &findType("fp32");
                } else {
                    dtOut = &findType("fp16");
                }
            }
        }
        
        std::vector<int> seq_lengths = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        
        // Filter seq_lengths to only include values up to max_seq_len
        std::vector<int> filtered_seq_lengths;
        for (int seq : seq_lengths) {
            if (seq <= max_seq_len) {
                filtered_seq_lengths.push_back(seq);
            }
        }
        seq_lengths = filtered_seq_lengths;
        
        printf("Testing up to seq=%d\n", max_seq_len);
        printf("%-8s %-12s %-12s %-8s\n", "Seq", "Time(ms)", "GFLOPS", "Status");
        printf("--------------------------------------------\n");
        
        for (int seq_len : seq_lengths) {
            // Get matrix dimensions for this pattern and seq_len
            uint32_t test_M, test_K, test_N;
            get_matrix_dims(pattern, seq_len, &test_M, &test_K, &test_N);
            
            if (test_M == 0 || test_K == 0 || test_N == 0) {
                printf("%-8d %-12s %-12s %-8s\n", seq_len, "INVALID", "INVALID", "SKIP");
                continue;
            }
            
            // Run the benchmark with pattern-specific dimensions using unified function
            BenchmarkResult result = runMatMulBenchmark(test_M, test_K, test_N, 
                                                       dt0, dt1, *dtOut, iterations, 
                                                       quantTypeA, quantTypeB, quantTypeC,
                                                       pattern, seq_len);
            
            printf("\r%-8d ", seq_len);
            if (result.success) {
                printf("%-12.2f %-12.2f %-8s\n", result.avg_time_ms, result.avg_gflops, "OK");
            } else {
                printf("%-12s %-12s %-8s\n", "FAILED", "FAILED", "SKIP");
            }
        }
        
        printf("\n=================================================================\n");
        printf("Pattern Benchmark Complete\n");
        printf("=================================================================\n");
        
        return 0;
    }

    //--------------------------------------------------------------------------
    // 9.5: Single Matrix Benchmark Mode (default mode)
    //--------------------------------------------------------------------------
    // Original single matrix benchmark mode
    const auto & dt0 = findType(dtypeStr0);
    const auto & dt1 = findType(dtypeStr1);
    
    //--------------------------------------------------------------------------
    // 9.6: NPU MatMul Configuration Validation
    //--------------------------------------------------------------------------
    // ⭐ NPU MatMul Configuration Validation ⭐
    printf("Validating NPU MatMul configuration...\n");
    const MatMulConfig* valid_config = validateMatMulConfig(dt0.dtype, dt1.dtype);
    
    if (!valid_config) {
        printf("❌ ERROR: Unsupported data type combination!\n");
        printf("   Input A (in[0]): %s\n", dt0.name);
        printf("   Input B (in[1]): %s\n", dt1.name);
        printf("\nThis combination is not supported by NPU MatMul operation.\n");
        printSupportedConfigurations();
        printf("Please choose a supported combination from the table above.\n");
        printf("\nExample usage:\n");
        printf("  # FP16 precision\n");
        printf("  ./MatMulBenchmark -t0 fp16 -t1 fp16\n");
        printf("  # Mixed precision FP16 × INT8\n");
        printf("  ./MatMulBenchmark -t0 fp16 -t1 int8\n");
        printf("  # INT8 quantized\n");
        printf("  ./MatMulBenchmark -t0 int8 -t1 int8\n");
        return EXIT_FAILURE;
    }
    
    // ✅ Automatically determine output type based on NPU support table
    const DTypeDesc * dtOut = nullptr;
    if (!dtypeStrOut.empty()) {
        dtOut = &findType(dtypeStrOut);
        // Verify user-specified output type matches NPU requirements
        if (dtOut->dtype != valid_config->out_type) {
            printf("⚠️  WARNING: User-specified output type '%s' doesn't match NPU requirement.\n", dtOut->name);
            printf("   NPU requires output type: ");
            for (size_t i = 0; i < NUM_TYPES; ++i) {
                if (kTypes[i].dtype == valid_config->out_type) {
                    printf("%s\n", kTypes[i].name);
                    dtOut = &kTypes[i];
                    break;
                }
            }
            printf("   Using NPU-required output type instead.\n");
        }
    } else {
        // Find the required output type from the validation table
        for (size_t i = 0; i < NUM_TYPES; ++i) {
            if (kTypes[i].dtype == valid_config->out_type) {
                dtOut = &kTypes[i];
                break;
            }
        }
    }
    
    if (!dtOut) {
        printf("❌ ERROR: Failed to determine output data type\n");
        return EXIT_FAILURE;
    }
    
    printf("✅ Valid NPU MatMul configuration: %s\n", valid_config->config_name);
    printf("   Input A → %s, Input B → %s, Output → %s\n", dt0.name, dt1.name, dtOut->name);

    std::cout << "Running Enhanced MatMul benchmark:\n"
              << "  Matrix sizes: A[" << M << "," << K << "] × B[" << K << "," << N 
              << "] = C[" << M << "," << N << "]\n"
              << "  Input A (in[0]) dtype: " << dt0.name << " (" << dt0.bytes << " bytes)\n"
              << "  Input B (in[1]) dtype: " << dt1.name << " (" << dt1.bytes << " bytes)\n"
              << "  Output C dtype: " << dtOut->name << " (" << dtOut->bytes << " bytes)\n"
              << "  Input A quant: " << getQuantizationTypeName(quantTypeA) << "\n"
              << "  Input B quant: " << getQuantizationTypeName(quantTypeB) << "\n"
              << "  Output C quant: " << getQuantizationTypeName(quantTypeC) << "\n"
              << "  Iterations: " << iterations << "\n"
              << "  Total elements: " << (static_cast<size_t>(M) * K + static_cast<size_t>(K) * N + static_cast<size_t>(M) * N) << std::endl;

    // Validate matrix dimensions
    if (M == 0 || K == 0 || N == 0) {
        std::cerr << "ERROR: Matrix dimensions must be positive" << std::endl;
        return EXIT_FAILURE;
    }

    //--------------------------------------------------------------------------
    // 9.7: Benchmark Execution and Results Display
    //--------------------------------------------------------------------------
    std::cout << "Enhanced MatMul benchmark using unified functions" << std::endl;
    
    // Run benchmark using unified function (QNN context는 함수 내에서 관리)
    BenchmarkResult result = runMatMulBenchmark(M, K, N, dt0, dt1, *dtOut, iterations, 
                                               quantTypeA, quantTypeB, quantTypeC);
    
    // Display results
    if (result.success) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Average time: " << result.avg_time_ms << " ms\n"
                  << "Throughput: " << result.avg_gflops << " GFLOPS\n";
        
        size_t total_bytes = static_cast<size_t>(M) * K * dt0.bytes + 
                           static_cast<size_t>(K) * N * dt1.bytes + 
                           static_cast<size_t>(M) * N * dtOut->bytes;
        std::cout << "Memory bandwidth: " << total_bytes / (result.avg_time_ms * 1e6) << " GB/s" << std::endl;
    } else {
        std::cerr << "ERROR: Benchmark failed" << std::endl;
        return 1;
    }
    
    return 0;
} 