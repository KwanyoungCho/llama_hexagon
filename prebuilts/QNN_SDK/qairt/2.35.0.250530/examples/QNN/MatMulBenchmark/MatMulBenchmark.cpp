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

//------------------------------------------------------------------------------
// Helper macro for simple error checking
//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
// Paths to backend and system libraries (modify if necessary)
//------------------------------------------------------------------------------
#if defined(__aarch64__)
static const char* DEFAULT_BACKEND_LIB = "libQnnHtp.so";      // device side
#else
static const char* DEFAULT_BACKEND_LIB = "libQnnHtp.so";      // host x86 simulation
#endif

//------------------------------------------------------------------------------
// Extended Datatype descriptor with more supported types
//------------------------------------------------------------------------------
struct DTypeDesc {
    const char * name;
    Qnn_DataType_t dtype;
    size_t bytes;
};

// NPU MatMul supported data type combinations (from the official table)
struct MatMulConfig {
    const char * config_name;
    Qnn_DataType_t in0_type;
    Qnn_DataType_t in1_type;
    Qnn_DataType_t out_type;
};

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

static const DTypeDesc kTypes[] = {
    {"fp16",  QNN_DATATYPE_FLOAT_16,        2},
    {"fp32",  QNN_DATATYPE_FLOAT_32,        4},
    {"int16", QNN_DATATYPE_SFIXED_POINT_16, 2},
    {"int8",  QNN_DATATYPE_SFIXED_POINT_8,  1},
    {"uint8", QNN_DATATYPE_UFIXED_POINT_8,  1},
    {"uint16", QNN_DATATYPE_UFIXED_POINT_16, 2},
    {"int32", QNN_DATATYPE_SFIXED_POINT_32, 4},
    {"uint32", QNN_DATATYPE_UFIXED_POINT_32, 4},
};

static const size_t NUM_TYPES = sizeof(kTypes) / sizeof(kTypes[0]);

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

//------------------------------------------------------------------------------
// Util: allocate and fill host buffer with random data
//------------------------------------------------------------------------------
static void randomFill(uint8_t * buf, size_t size) {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<uint32_t> dist(0, 255);
    for (size_t i = 0; i < size; ++i) buf[i] = static_cast<uint8_t>(dist(rng));
}

//------------------------------------------------------------------------------
// NPU MatMul Configuration Validation
//------------------------------------------------------------------------------

// Function to validate and get output type for given input types
static const MatMulConfig* validateMatMulConfig(Qnn_DataType_t in0_type, Qnn_DataType_t in1_type) {
    for (size_t i = 0; i < NUM_SUPPORTED_CONFIGS; ++i) {
        const MatMulConfig& config = SUPPORTED_MATMUL_CONFIGS[i];
        if (config.in0_type == in0_type && config.in1_type == in1_type) {
            return &config;
        }
    }
    return nullptr;
}

// Function to print all supported configurations
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

//------------------------------------------------------------------------------
// Dynamic loader for QNN interface
//------------------------------------------------------------------------------
using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);

// Global System interface (following ggml-hexagon pattern)
static QnnSystemInterface_t* g_systemInterface = nullptr;
static void* g_systemLibHandle = nullptr;

// Global raw interface for ggml-hexagon style tensor creation
static QNN_INTERFACE_VER_TYPE g_qnnRawInterface;

static bool loadSystemInterface() {
    if (g_systemInterface) return true; // Already loaded
    
    // Silently try to load system interface (optional component)
    g_systemLibHandle = dlopen("/data/local/tmp/libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (!g_systemLibHandle) {
        // System interface is optional, silently continue
        return false;
    }
    
    auto getSystemInterface = (QnnSystemInterface_t*(*)())dlsym(g_systemLibHandle, "GetSystemInterface");
    if (!getSystemInterface) {
        // System interface symbol not found, silently continue
        dlclose(g_systemLibHandle);
        g_systemLibHandle = nullptr;
        return false;
    }
    
    g_systemInterface = getSystemInterface();
    if (!g_systemInterface) {
        // System interface creation failed, silently continue
        dlclose(g_systemLibHandle);
        g_systemLibHandle = nullptr;
        return false;
    }
    
    // Only show success message if actually loaded
    std::cout << "INFO: QNN System interface loaded successfully" << std::endl;
    return true;
}

static const QNN_INTERFACE_VER_TYPE * loadInterface(void ** libHandleOut, const char * libPath) {
    // Load system interface first (like ggml-hexagon)
    loadSystemInterface();
    
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

//------------------------------------------------------------------------------
// ggml-hexagon style tensor creation functions  
//------------------------------------------------------------------------------
static int deepCopyQnnTensor(Qnn_Tensor_t & src, Qnn_Tensor_t & dst) {
    dst.version = src.version;
    
    // Allocate and copy name
    size_t nameLen = strlen(src.v1.name);
    char * dstName = (char *)malloc(nameLen + 1);
    if (!dstName) return 1;
    strcpy(dstName, src.v1.name);
    dst.v1.name = dstName;
    
    dst.v1.id = src.v1.id;
    dst.v1.type = src.v1.type;
    dst.v1.dataFormat = src.v1.dataFormat;
    dst.v1.dataType = src.v1.dataType;
    dst.v1.quantizeParams = src.v1.quantizeParams;
    dst.v1.rank = src.v1.rank;
    
    // Allocate and copy dimensions
    size_t dimSize = src.v1.rank * sizeof(uint32_t);
    uint32_t * dstDims = (uint32_t *)malloc(dimSize);
    if (!dstDims) {
        free((void*)dstName);
        return 1;
    }
    memcpy(dstDims, src.v1.dimensions, dimSize);
    dst.v1.dimensions = dstDims;
    
    dst.v1.memType = src.v1.memType;
    dst.v1.clientBuf = src.v1.clientBuf;
    
    return 0;
}

static void freeTensor(Qnn_Tensor_t * tensor) {
    if (tensor) {
        free((void*)tensor->v1.name);
        free((void*)tensor->v1.dimensions);
        free(tensor);
    }
}

static Qnn_Tensor_t * createTensorHexagonStyle(const char * baseName, Qnn_TensorType_t tensorType,
                                              const DTypeDesc & dt, uint32_t * dims, uint32_t rank,
                                              void * dataPtr, uint32_t dataSize) {
    // Create unique tensor name like ggml-hexagon
    static int tensorIdx = 0;
    char uniqueName[128];
    snprintf(uniqueName, sizeof(uniqueName), "tensor_%s_%d", baseName, tensorIdx++);
    
    // Create template tensor
    Qnn_Tensor_t templateTensor = {};
    templateTensor.version = QNN_TENSOR_VERSION_1;
    templateTensor.v1.id = 0;
    templateTensor.v1.name = uniqueName;
    templateTensor.v1.type = tensorType;
    templateTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    templateTensor.v1.dataType = dt.dtype;
    templateTensor.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
    templateTensor.v1.rank = rank;
    templateTensor.v1.dimensions = dims;
    templateTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    templateTensor.v1.clientBuf = {dataPtr, dataSize};
    
    // Allocate dynamic memory like ggml-hexagon
    Qnn_Tensor_t * pTensor = (Qnn_Tensor_t *)calloc(1, sizeof(Qnn_Tensor_t));
    if (!pTensor) {
        printf("ERROR: Failed to allocate tensor memory\n");
        return nullptr;
    }
    
    // Deep copy like ggml-hexagon
    if (deepCopyQnnTensor(templateTensor, *pTensor) != 0) {
        printf("ERROR: Failed to deep copy tensor\n");
        free(pTensor);
        return nullptr;
    }
    
    printf("Created tensor: %s, type=%d, dataType=%d, rank=%d\n", 
           uniqueName, tensorType, dt.dtype, rank);
    
    return pTensor;
}

// Utility to construct a Qnn_Tensor_t (version 1) - Legacy function
//------------------------------------------------------------------------------
static Qnn_Tensor_t makeTensor(const char * name, Qnn_TensorType_t tensorType,
                               const DTypeDesc & dt, uint32_t * dims, uint32_t rank,
                               void * dataPtr, uint32_t dataSize) {
    Qnn_Tensor_t t{}; t.version = QNN_TENSOR_VERSION_1;
    t.v1.id = 0;
    t.v1.name = name;
    t.v1.type = tensorType;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType   = dt.dtype;
    t.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT; // no quantization (FP / int raw)
    t.v1.rank = rank;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf.data = dataPtr;
    t.v1.clientBuf.dataSize = dataSize;
    return t;
}

//------------------------------------------------------------------------------
// Pattern-based benchmark functions (like matmul-benchmark.cpp)
//------------------------------------------------------------------------------
struct BenchmarkResult {
    int pattern;
    int seq_len;
    double avg_time_ms;
    double avg_gflops;
    bool success;
};

static const char* get_pattern_name(int pattern) {
    switch(pattern) {
        case 0: return "Attention [seq x 4096] x [4096 x seq]";
        case 1: return "Linear [seq x 4096] x [4096 x 4096]";
        case 2: return "FFN_Up [seq x 4096] x [4096 x 11008]";
        case 3: return "FFN_Down [seq x 11008] x [11008 x 4096]";
        default: return "Unknown";
    }
}

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

//------------------------------------------------------------------------------
// main
//------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    // default parameters
    uint32_t M = 128, K = 4096, N = 4096;
    std::string dtypeStr0 = "fp32";  // Input tensor A datatype (기본값을 fp32로 유지)
    std::string dtypeStr1 = "fp32";  // Input tensor B datatype (기본값을 fp32로 유지)
    std::string dtypeStrOut = "";    // Output tensor datatype (optional, derived from inputs)
    int iterations       = 10;
    const char * backendLib = DEFAULT_BACKEND_LIB;

    // Pattern mode parameters (new)
    bool pattern_mode = false;
    bool all_patterns_mode = false;  // 모든 패턴 실행
    int pattern = -1;  // -1 means no pattern specified

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
        } else if (strcmp(argv[i], "-lib") == 0 && i + 1 < argc) {
            backendLib = argv[++i];
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
                      << "                  Pattern mode tests seq_len from 1 to 4096\n"
                      << "  -a            : Test all patterns and generate summary report\n\n"
                      << "Datatype Configuration:\n"
                      << "  -t0 <type>    : Input tensor A (in[0]) datatype\n"
                      << "  -t1 <type>    : Input tensor B (in[1]) datatype\n"
                      << "  -t <type>     : Set both input tensors to same type (legacy)\n"
                      << "  -to <type>    : Output tensor datatype (optional)\n\n"
                      << "Supported datatypes: fp16, fp32, int16, int8, uint8, uint16, int32, uint32\n\n"
                      << "Execution Configuration:\n"
                      << "  -i <iter>     : Number of iterations (default: 10)\n"
                      << "  -lib <path>   : Backend library path\n\n"
                      << "Examples:\n"
                      << "  # Manual matrix size\n"
                      << "  ./MatMulBenchmark -m 1024 -k 2048 -n 2048 -t fp16\n"
                      << "  # Pattern mode: FFN_Up with mixed precision\n"
                      << "  ./MatMulBenchmark -p 2 -t0 fp16 -t1 int8\n"
                      << "  # Pattern mode: Attention with fp32\n"
                      << "  ./MatMulBenchmark -p 0 -t fp32\n"
                      << "  # All patterns with comprehensive report\n"
                      << "  ./MatMulBenchmark -a -t fp32\n"
                      << std::endl;
            return 0;
        }
    }

    if (all_patterns_mode) {
        // All patterns benchmark mode (like matmul-benchmark.cpp)
        std::cout << "=================================================================\n";
        std::cout << "Comprehensive NPU MatMul Benchmark - All LLM Patterns\n";
        std::cout << "Input A dtype: " << dtypeStr0 << ", Input B dtype: " << dtypeStr1 << "\n";
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
        std::vector<int> patterns = {0, 1, 2, 3};
        std::vector<BenchmarkResult> all_results;
        
        int total_tests = patterns.size() * seq_lengths.size();
        int current_test = 0;
        
        printf("Running %d total test cases...\n\n", total_tests);
        
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
                
                // Run the benchmark with pattern-specific dimensions
                M = test_M; K = test_K; N = test_N;
                
                bool success = false;
                double avgMs = 0.0;
                
                try {
                    // Initialize QNN interface (simplified, always use HTP)
    void * libHandle = nullptr;
                    const QNN_INTERFACE_VER_TYPE * iface = loadInterface(&libHandle, "libQnnHtp.so");
                    if (!iface) {
                        throw std::runtime_error("Failed to load HTP backend");
                    }

                    // Create logger, backend, device, context (simplified)
    Qnn_LogHandle_t logger = nullptr;
                    CHECK_QNN(iface->logCreate(nullptr, QNN_LOG_LEVEL_ERROR, &logger));

    Qnn_BackendHandle_t backend = nullptr;
    CHECK_QNN(iface->backendCreate(logger, nullptr, &backend));

                    Qnn_DeviceHandle_t device = nullptr;
                    iface->deviceCreate(logger, nullptr, &device); // Ignore errors
                    
                    Qnn_ContextHandle_t ctx = nullptr;
                    CHECK_QNN(iface->contextCreate(backend, device, nullptr, &ctx));
                    
                    // Create HTP graph with VTCM and HVX threads (like ggml-hexagon)
                    Qnn_GraphHandle_t graph = nullptr;
                    
                    // VTCM configuration
                    QnnHtpGraph_CustomConfig_t vtcmConfig;
                    vtcmConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
                    vtcmConfig.vtcmSizeInMB = 8;
                    
                    // HVX threads configuration (like ggml-hexagon: 8 threads)
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
                    CHECK_QNN(iface->graphCreate(ctx, "all_pattern_matmul", pGraphConfig, &graph));
                    
                    // Prepare buffers
                    size_t bytesA = static_cast<size_t>(M) * K * dt0.bytes;
                    size_t bytesB = static_cast<size_t>(K) * N * dt1.bytes;
                    size_t bytesC = static_cast<size_t>(M) * N * dtOut->bytes;
                    
                    std::vector<uint8_t> bufA(bytesA);
                    std::vector<uint8_t> bufB(bytesB);
                    std::vector<uint8_t> bufC(bytesC);
                    
                    randomFill(bufA.data(), bytesA);
                    randomFill(bufB.data(), bytesB);
                    
                    // Create 4D tensors
                    uint32_t dimA_4d[4] = {1, 1, M, K};
                    uint32_t dimB_4d[4] = {1, 1, K, N};
                    uint32_t dimC_4d[4] = {1, 1, M, N};
                    
                    // Simplified tensor creation (reusing existing logic)
                    Qnn_Tensor_t tenA = {
                        .version = QNN_TENSOR_VERSION_1,
                        .v1 = {
                            .id = 0, .name = "all_A", .type = QNN_TENSOR_TYPE_APP_WRITE,
                            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, .dataType = dt0.dtype,
                            .quantizeParams = {QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
                            .rank = 4, .dimensions = dimA_4d, .memType = QNN_TENSORMEMTYPE_RAW,
                            .clientBuf = {.data = nullptr, .dataSize = 0}
                        }
                    };
                    CHECK_QNN(iface->tensorCreateGraphTensor(graph, &tenA));
                    
                    Qnn_Tensor_t tenB = {
                        .version = QNN_TENSOR_VERSION_1,
                        .v1 = {
                            .id = 1, .name = "all_B", .type = QNN_TENSOR_TYPE_APP_WRITE,
                            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, .dataType = dt1.dtype,
                            .quantizeParams = {QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
                            .rank = 4, .dimensions = dimB_4d, .memType = QNN_TENSORMEMTYPE_RAW,
                            .clientBuf = {.data = nullptr, .dataSize = 0}
                        }
                    };
                    CHECK_QNN(iface->tensorCreateGraphTensor(graph, &tenB));
                    
                    Qnn_Tensor_t tenC = {
                        .version = QNN_TENSOR_VERSION_1,
                        .v1 = {
                            .id = 2, .name = "all_C", .type = QNN_TENSOR_TYPE_APP_READ,
                            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, .dataType = dtOut->dtype,
                            .quantizeParams = {QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
                            .rank = 4, .dimensions = dimC_4d, .memType = QNN_TENSORMEMTYPE_RAW,
                            .clientBuf = {.data = nullptr, .dataSize = 0}
                        }
                    };
                    CHECK_QNN(iface->tensorCreateGraphTensor(graph, &tenC));
                    
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
                    op.v1.name = "all_matmul";
                    op.v1.packageName = "qti.aisw";
                    op.v1.typeName = QNN_OP_MAT_MUL;
                    op.v1.numOfParams = 2; op.v1.params = params;
                    op.v1.numOfInputs = 2; op.v1.inputTensors = inputsNode;
                    op.v1.numOfOutputs = 1; op.v1.outputTensors = outputs;
                    
                    CHECK_QNN(iface->graphAddNode(graph, op));
                    CHECK_QNN(iface->graphFinalize(graph, nullptr, nullptr));
                    
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
                    CHECK_QNN(iface->graphExecute(graph, inputsExec, 2, outputsExec, 1, nullptr, nullptr)); // warmup
                    
                    double tot = 0.0;
                    for (int i = 0; i < iterations; ++i) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        CHECK_QNN(iface->graphExecute(graph, inputsExec, 2, outputsExec, 1, nullptr, nullptr));
                        auto t1 = std::chrono::high_resolution_clock::now();
                        tot += std::chrono::duration<double, std::milli>(t1 - t0).count();
                    }
                    avgMs = tot / iterations;
                    success = true;
                    
                    // Cleanup
                    iface->contextFree(ctx, nullptr);
                    if (device != nullptr) iface->deviceFree(device);
                    iface->backendFree(backend);
                    iface->logFree(logger);
                    dlclose(libHandle);
                    
                } catch (const std::exception& e) {
                    success = false;
                } catch (...) {
                    success = false;
                }
                
                double gflops = success ? (2.0 * M * K * N) / (avgMs * 1e6) : 0.0;
                
                BenchmarkResult result = {test_pattern, seq_len, avgMs, gflops, success};
                all_results.push_back(result);
                
                printf("\r%-8d ", seq_len);
                if (success) {
                    printf("%-12.2f %-12.2f %-8s\n", avgMs, gflops, "OK");
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

    if (pattern_mode) {
        // Pattern-based benchmark mode (like matmul-benchmark.cpp)
        std::cout << "=================================================================\n";
        std::cout << "Pattern-based MatMul Benchmark for NPU\n";
        std::cout << "Pattern: " << get_pattern_name(pattern) << "\n";
        std::cout << "Input A dtype: " << dtypeStr0 << ", Input B dtype: " << dtypeStr1 << "\n";
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
            
            printf("\r[Processing] seq=%d... ", seq_len);
            fflush(stdout);
            
            // Run the benchmark with pattern-specific dimensions
            M = test_M; K = test_K; N = test_N;
            
            bool success = false;
            double avgMs = 0.0;
            
            try {
                // Initialize QNN interface (simplified, always use HTP)
                void * libHandle = nullptr;
                const QNN_INTERFACE_VER_TYPE * iface = loadInterface(&libHandle, "libQnnHtp.so");
                if (!iface) {
                    throw std::runtime_error("Failed to load HTP backend");
                }
                
                // Create logger, backend, device, context (simplified)
                Qnn_LogHandle_t logger = nullptr;
                CHECK_QNN(iface->logCreate(nullptr, QNN_LOG_LEVEL_ERROR, &logger));
                
                Qnn_BackendHandle_t backend = nullptr;
                CHECK_QNN(iface->backendCreate(logger, nullptr, &backend));

    Qnn_DeviceHandle_t device = nullptr;
                iface->deviceCreate(logger, nullptr, &device); // Ignore errors
                
                Qnn_ContextHandle_t ctx = nullptr;
                CHECK_QNN(iface->contextCreate(backend, device, nullptr, &ctx));
                
                // Create HTP graph with VTCM and HVX threads (like ggml-hexagon)
                Qnn_GraphHandle_t graph = nullptr;
                
                // VTCM configuration
                QnnHtpGraph_CustomConfig_t vtcmConfig;
                vtcmConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
                vtcmConfig.vtcmSizeInMB = 8;
                
                // HVX threads configuration (like ggml-hexagon: 8 threads)
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
                CHECK_QNN(iface->graphCreate(ctx, "pattern_matmul", pGraphConfig, &graph));
                
                // Prepare buffers
                size_t bytesA = static_cast<size_t>(M) * K * dt0.bytes;
                size_t bytesB = static_cast<size_t>(K) * N * dt1.bytes;
                size_t bytesC = static_cast<size_t>(M) * N * dtOut->bytes;
                
                std::vector<uint8_t> bufA(bytesA);
                std::vector<uint8_t> bufB(bytesB);
                std::vector<uint8_t> bufC(bytesC);
                
                randomFill(bufA.data(), bytesA);
                randomFill(bufB.data(), bytesB);
                
                // Create 4D tensors
                uint32_t dimA_4d[4] = {1, 1, M, K};
                uint32_t dimB_4d[4] = {1, 1, K, N};
                uint32_t dimC_4d[4] = {1, 1, M, N};
                
                Qnn_Tensor_t tenA = {
                    .version = QNN_TENSOR_VERSION_1,
                    .v1 = {
                        .id = 0, .name = "pattern_A", .type = QNN_TENSOR_TYPE_APP_WRITE,
                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, .dataType = dt0.dtype,
                        .quantizeParams = {QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
                        .rank = 4, .dimensions = dimA_4d, .memType = QNN_TENSORMEMTYPE_RAW,
                        .clientBuf = {.data = nullptr, .dataSize = 0}
                    }
                };
                CHECK_QNN(iface->tensorCreateGraphTensor(graph, &tenA));
                
                Qnn_Tensor_t tenB = {
                    .version = QNN_TENSOR_VERSION_1,
                    .v1 = {
                        .id = 1, .name = "pattern_B", .type = QNN_TENSOR_TYPE_APP_WRITE,
                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, .dataType = dt1.dtype,
                        .quantizeParams = {QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
                        .rank = 4, .dimensions = dimB_4d, .memType = QNN_TENSORMEMTYPE_RAW,
                        .clientBuf = {.data = nullptr, .dataSize = 0}
                    }
                };
                CHECK_QNN(iface->tensorCreateGraphTensor(graph, &tenB));
                
                Qnn_Tensor_t tenC = {
                    .version = QNN_TENSOR_VERSION_1,
                    .v1 = {
                        .id = 2, .name = "pattern_C", .type = QNN_TENSOR_TYPE_APP_READ,
                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, .dataType = dtOut->dtype,
                        .quantizeParams = {QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
                        .rank = 4, .dimensions = dimC_4d, .memType = QNN_TENSORMEMTYPE_RAW,
                        .clientBuf = {.data = nullptr, .dataSize = 0}
                    }
                };
                CHECK_QNN(iface->tensorCreateGraphTensor(graph, &tenC));
                
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
                op.v1.name = "pattern_matmul";
                op.v1.packageName = "qti.aisw";
                op.v1.typeName = QNN_OP_MAT_MUL;
                op.v1.numOfParams = 2; op.v1.params = params;
                op.v1.numOfInputs = 2; op.v1.inputTensors = inputsNode;
                op.v1.numOfOutputs = 1; op.v1.outputTensors = outputs;
                
                CHECK_QNN(iface->graphAddNode(graph, op));
                CHECK_QNN(iface->graphFinalize(graph, nullptr, nullptr));
                
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
                CHECK_QNN(iface->graphExecute(graph, inputsExec, 2, outputsExec, 1, nullptr, nullptr)); // warmup
                
                double tot = 0.0;
                for (int i = 0; i < iterations; ++i) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    CHECK_QNN(iface->graphExecute(graph, inputsExec, 2, outputsExec, 1, nullptr, nullptr));
                    auto t1 = std::chrono::high_resolution_clock::now();
                    tot += std::chrono::duration<double, std::milli>(t1 - t0).count();
                }
                avgMs = tot / iterations;
                success = true;
                
                // Cleanup
                iface->contextFree(ctx, nullptr);
                if (device != nullptr) iface->deviceFree(device);
                iface->backendFree(backend);
                iface->logFree(logger);
                dlclose(libHandle);
                
            } catch (const std::exception& e) {
                success = false;
            } catch (...) {
                success = false;
            }
            
            double gflops = success ? (2.0 * M * K * N) / (avgMs * 1e6) : 0.0;
            
            printf("\r%-8d ", seq_len);
            if (success) {
                printf("%-12.2f %-12.2f %-8s\n", avgMs, gflops, "OK");
            } else {
                printf("%-12s %-12s %-8s\n", "FAILED", "FAILED", "SKIP");
            }
        }
        
        printf("\n=================================================================\n");
        printf("Pattern Benchmark Complete\n");
        printf("=================================================================\n");
        
        return 0;
    }

    // Original single matrix benchmark mode
    const auto & dt0 = findType(dtypeStr0);
    const auto & dt1 = findType(dtypeStr1);
    
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
              << "  Iterations: " << iterations << "\n"
              << "  Total elements: " << (static_cast<size_t>(M) * K + static_cast<size_t>(K) * N + static_cast<size_t>(M) * N) << std::endl;

    // Validate matrix dimensions
    if (M == 0 || K == 0 || N == 0) {
        std::cerr << "ERROR: Matrix dimensions must be positive" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Enhanced MatMul benchmark - trying HTP (NPU) backend first" << std::endl;
    
    // 1. Try HTP (NPU) backend first for hardware acceleration
    void * libHandle = nullptr;
    const QNN_INTERFACE_VER_TYPE * iface = nullptr;
    
    // Try to load HTP backend first for NPU acceleration
    const char* backends[] = {"libQnnHtp.so", "libQnnCpu.so"};
    const char* backend_names[] = {"HTP", "CPU"};
    const char* loaded_backend_name = nullptr;
    
    for (int i = 0; i < 2; i++) {
        std::cout << "Trying " << backend_names[i] << " backend: " << backends[i] << std::endl;
        iface = loadInterface(&libHandle, backends[i]);
        if (iface) {
            std::cout << "SUCCESS: Loaded " << backend_names[i] << " backend" << std::endl;
            loaded_backend_name = backend_names[i];
            break;
        } else {
            std::cout << "FAILED: Could not load " << backend_names[i] << " backend" << std::endl;
        }
    }
    
    if (!iface) {
        std::cerr << "ERROR: Failed to load any QNN backend" << std::endl;
        return 1;
    }
    
    // Set global raw interface for ggml-hexagon style tensor creation
    g_qnnRawInterface = *iface;

    // Check API version compatibility
    Qnn_ApiVersion_t apiVersion;
    if (iface->backendGetApiVersion(&apiVersion) == QNN_SUCCESS) {
        std::cout << "QNN API Version: Core " << apiVersion.coreApiVersion.major 
                  << "." << apiVersion.coreApiVersion.minor 
                  << "." << apiVersion.coreApiVersion.patch
                  << ", Backend " << apiVersion.backendApiVersion.major
                  << "." << apiVersion.backendApiVersion.minor
                  << "." << apiVersion.backendApiVersion.patch << std::endl;
    }

    // 2. Create logger – use DEBUG level like ggml-hexagon
    Qnn_LogHandle_t logger = nullptr;
    // CHECK_QNN(iface->logCreate(/*callback=*/nullptr, QNN_LOG_LEVEL_DEBUG, &logger));
    CHECK_QNN(iface->logCreate(/*callback=*/nullptr, QNN_LOG_LEVEL_ERROR, &logger));

    // 3. Create backend (no extra configs) – pass logger
    Qnn_BackendHandle_t backend = nullptr;
    CHECK_QNN(iface->backendCreate(logger, nullptr, &backend));

    // 4. Create device with backend-specific configuration (following ggml-hexagon pattern exactly)
    Qnn_DeviceHandle_t device = nullptr;
    Qnn_ErrorHandle_t deviceResult = QNN_SUCCESS;
    
    // Check if we're using HTP backend (following ggml-hexagon EXACTLY)
    if (loaded_backend_name && strstr(loaded_backend_name, "HTP") != nullptr) {
        std::cout << "INFO: HTP backend detected, following ggml-hexagon device creation pattern..." << std::endl;
        
        // Get platform info exactly like ggml-hexagon (line 3459)
        const QnnDevice_PlatformInfo_t * p_info = nullptr;
        struct {
            uint32_t soc_model;
            size_t htp_arch;
            size_t vtcm_size_in_mb;
        } soc_info = {};
        
        deviceResult = iface->deviceGetPlatformInfo(nullptr, &p_info);
        if (deviceResult == QNN_SUCCESS) {
            std::cout << "SUCCESS: Got platform info, " << p_info->v1.numHwDevices << " HW devices" << std::endl;
            
            // Extract device info exactly like ggml-hexagon (line 3462-3474)
            QnnDevice_HardwareDeviceInfo_t * infos = p_info->v1.hwDevices;
            QnnHtpDevice_OnChipDeviceInfoExtension_t chipinfo = {};
            for (uint32_t i = 0; i < p_info->v1.numHwDevices; i++) {
                std::cout << "INFO: deviceID:" << infos[i].v1.deviceId 
                         << ", deviceType:" << infos[i].v1.deviceType 
                         << ", numCores " << infos[i].v1.numCores << std::endl;
                QnnDevice_DeviceInfoExtension_t devinfo = infos[i].v1.deviceInfoExtension;
                chipinfo = devinfo->onChipDevice;
                size_t htp_arch = (size_t) chipinfo.arch;
                std::cout << "INFO: htp_type:" << devinfo->devType 
                         << (devinfo->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP ? "(ON_CHIP)" : "") << std::endl;
                soc_info = { chipinfo.socModel, htp_arch, chipinfo.vtcmSize };
                std::cout << "INFO: Detected SoC model " << chipinfo.socModel 
                         << ", HTP arch " << htp_arch << ", VTCM " << chipinfo.vtcmSize << " MB" << std::endl;
            }
            iface->deviceFreePlatformInfo(nullptr, p_info);
        } else {
            std::cout << "WARNING: Failed to get platform info, are we in emulator?" << std::endl;
            soc_info = { 0, 0, 0 }; // Default like ggml-hexagon
        }

        // Create device config exactly like ggml-hexagon (line 3480-3497)
        QnnHtpDevice_CustomConfig_t soc_customconfig;
        soc_customconfig.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
        soc_customconfig.socModel = soc_info.soc_model;
        QnnDevice_Config_t soc_devconfig;
        soc_devconfig.option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
        soc_devconfig.customConfig = &soc_customconfig;

        const QnnDevice_Config_t * p_deviceconfig[] = { &soc_devconfig, nullptr };
        deviceResult = iface->deviceCreate(logger, p_deviceconfig, &device);
    } else {
        std::cout << "INFO: CPU backend detected, creating simple device..." << std::endl;
        // For non-HTP backends, simple device creation like ggml-hexagon (line 3499)
        deviceResult = iface->deviceCreate(logger, nullptr, &device);
    }
    
    // Handle device creation result exactly like ggml-hexagon (line 3501-3505)
    // 중요: ggml-hexagon에서는 device 생성 실패해도 device를 nullptr로 설정하지 않음
    if (deviceResult != QNN_SUCCESS && deviceResult != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
        std::cout << "WARNING: Failed to create QNN device (code=" << deviceResult << ")" << std::endl;
        // ggml-hexagon처럼 device를 nullptr로 설정하지 않고 그대로 유지
    } else {
        std::cout << "SUCCESS: Create device successfully" << std::endl;
    }

    // 5. Create context (following ggml-hexagon pattern exactly)
    Qnn_ContextHandle_t ctx = nullptr;
    
    // Create empty context config vector like ggml-hexagon
    std::vector<const QnnContext_Config_t *> contextConfigs;
    const QnnContext_Config_t ** configPtr = contextConfigs.empty() ? nullptr : contextConfigs.data();
    
    Qnn_ErrorHandle_t contextResult = iface->contextCreate(backend, device, configPtr, &ctx);
    if (contextResult != QNN_SUCCESS) {
        std::cout << "ERROR: Context creation failed (code=" << contextResult << "). Device=" 
                  << (device ? "valid" : "nullptr") << std::endl;
        return 1;
    } else {
        std::cout << "SUCCESS: Context created successfully with device=" 
                  << (device ? "valid" : "nullptr") << std::endl;
    }

    // 6. Create graph with HTP-specific configuration (following ggml-hexagon exactly)
    Qnn_GraphHandle_t graph = nullptr;
    
        if (loaded_backend_name && strstr(loaded_backend_name, "HTP") != nullptr) {
        printf("Creating HTP graph with VTCM and HVX threads config...\n");
        
        // VTCM configuration (like ggml-hexagon)
        QnnHtpGraph_CustomConfig_t vtcmConfig;
        vtcmConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
        vtcmConfig.vtcmSizeInMB = 8;
        
        // HVX threads configuration (like ggml-hexagon: 8 threads)
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

        CHECK_QNN(iface->graphCreate(ctx, "matmul", pGraphConfig, &graph));
        printf("SUCCESS: Created HTP graph with VTCM and HVX threads config\n");
    } else {
        // CPU backend - simple graph creation
    CHECK_QNN(iface->graphCreate(ctx, "matmul", nullptr, &graph));
    }

    // 7. Prepare dimensions
    uint32_t dimA[2] = {M, K}; // [M,K]
    uint32_t dimB[2] = {K, N}; // [K,N]
    uint32_t dimC[2] = {M, N}; // [M,N]

    // 8. Host buffers with individual datatypes
    size_t bytesA = static_cast<size_t>(M) * K * dt0.bytes;
    size_t bytesB = static_cast<size_t>(K) * N * dt1.bytes;
    size_t bytesC = static_cast<size_t>(M) * N * dtOut->bytes;

    std::vector<uint8_t> bufA(bytesA);
    std::vector<uint8_t> bufB(bytesB);
    std::vector<uint8_t> bufC(bytesC);

    randomFill(bufA.data(), bytesA);
    randomFill(bufB.data(), bytesB);

    // 9. Create graph tensors using mllm method (direct struct initialization)
    printf("Creating tensors using mllm method...\n");
    
    // Convert 2D to 4D dimensions (following mllm pattern)
    uint32_t dimA_4d[4] = {1, 1, dimA[0], dimA[1]};  // [1,1,M,K] 
    uint32_t dimB_4d[4] = {1, 1, dimB[0], dimB[1]};  // [1,1,K,N]  
    uint32_t dimC_4d[4] = {1, 1, dimC[0], dimC[1]};  // [1,1,M,N]
    
    // Input tensor A (4D로 변경)
    Qnn_Tensor_t tenA = {
        .version = QNN_TENSOR_VERSION_1,
        .v1 = {
            .id = 0,
            .name = "tensor_A",
            .type = QNN_TENSOR_TYPE_APP_WRITE,
            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType = dt0.dtype,
            .quantizeParams = {QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
            .rank = 4,  // 4D로 변경
            .dimensions = dimA_4d,  // 4D 차원 사용
            .memType = QNN_TENSORMEMTYPE_RAW,
            .clientBuf = {.data = nullptr, .dataSize = 0}
        }
    };
    CHECK_QNN(iface->tensorCreateGraphTensor(graph, &tenA));
    printf("SUCCESS: Created tensor A with 4D method\n");
    
    // Input tensor B (4D로 변경)
    Qnn_Tensor_t tenB = {
        .version = QNN_TENSOR_VERSION_1,
        .v1 = {
            .id = 1,
            .name = "tensor_B", 
            .type = QNN_TENSOR_TYPE_APP_WRITE,
            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType = dt1.dtype,
            .quantizeParams = {QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
            .rank = 4,  // 4D로 변경
            .dimensions = dimB_4d,  // 4D 차원 사용
            .memType = QNN_TENSORMEMTYPE_RAW,
            .clientBuf = {.data = nullptr, .dataSize = 0}
        }
    };
    CHECK_QNN(iface->tensorCreateGraphTensor(graph, &tenB));
    printf("SUCCESS: Created tensor B with 4D method\n");
    
    // Output tensor C (4D로 변경)
    Qnn_Tensor_t tenC = {
        .version = QNN_TENSOR_VERSION_1,
        .v1 = {
            .id = 2,
            .name = "tensor_C",
            .type = QNN_TENSOR_TYPE_APP_READ,
            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType = dtOut->dtype,
            .quantizeParams = {QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},
            .rank = 4,  // 4D로 변경
            .dimensions = dimC_4d,  // 4D 차원 사용
            .memType = QNN_TENSORMEMTYPE_RAW,
            .clientBuf = {.data = nullptr, .dataSize = 0}
        }
    };
    CHECK_QNN(iface->tensorCreateGraphTensor(graph, &tenC));
    printf("SUCCESS: Created tensor C with 4D method\n");

    // 10. Build MatMul op config (no bias, no transpose)
    Qnn_Param_t params[2] = {};
    params[0].paramType = QNN_PARAMTYPE_SCALAR;
    params[0].name      = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
    params[0].scalarParam.dataType   = QNN_DATATYPE_BOOL_8;
    params[0].scalarParam.bool8Value = 0;

    params[1].paramType = QNN_PARAMTYPE_SCALAR;
    params[1].name      = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
    params[1].scalarParam.dataType   = QNN_DATATYPE_BOOL_8;
    params[1].scalarParam.bool8Value = 0;

    Qnn_Tensor_t inputsNode[2] = {tenA, tenB};
    Qnn_Tensor_t outputs[1]    = {tenC};

    Qnn_OpConfig_t op{}; op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name        = "matmul";
    op.v1.packageName = "qti.aisw";
    op.v1.typeName    = QNN_OP_MAT_MUL;
    op.v1.numOfParams = 2; op.v1.params = params;
    op.v1.numOfInputs = 2; op.v1.inputTensors = inputsNode;
    op.v1.numOfOutputs = 1; op.v1.outputTensors = outputs;

    CHECK_QNN(iface->graphAddNode(graph, op));

    // 11. Finalize graph
    CHECK_QNN(iface->graphFinalize(graph, nullptr, nullptr));

    // 12. Build tensor set for execution with actual data buffers
    Qnn_Tensor_t inputsExec[2] = {tenA, tenB};
    Qnn_Tensor_t outputsExec[1] = {tenC};
    
    // Set actual data buffers for execution (following mllm pattern)
    inputsExec[0].v1.clientBuf.data = bufA.data();
    inputsExec[0].v1.clientBuf.dataSize = bytesA;
    
    inputsExec[1].v1.clientBuf.data = bufB.data();
    inputsExec[1].v1.clientBuf.dataSize = bytesB;
    
    outputsExec[0].v1.clientBuf.data = bufC.data();
    outputsExec[0].v1.clientBuf.dataSize = bytesC;
    
    printf("Setting data buffers for execution:\n");
    printf("  Input A: %zu bytes\n", bytesA);
    printf("  Input B: %zu bytes\n", bytesB);
    printf("  Output C: %zu bytes\n", bytesC);

    // 13. Benchmark execution (simple version)
    CHECK_QNN(iface->graphExecute(graph, inputsExec, 2, outputsExec, 1, nullptr, nullptr)); // warmup
    
    double tot=0.0;
    for(int i=0;i<iterations;++i){
        auto t0=std::chrono::high_resolution_clock::now();
        CHECK_QNN(iface->graphExecute(graph, inputsExec, 2, outputsExec, 1, nullptr, nullptr));
        auto t1=std::chrono::high_resolution_clock::now();
        tot += std::chrono::duration<double, std::milli>(t1-t0).count();
    }
    double avgMs = tot / iterations;
    double gflops = (2.0 * M * K * N) / (avgMs * 1e6);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average time: " << avgMs << " ms\n"
              << "Throughput: " << gflops << " GFLOPS\n" 
              << "Memory bandwidth: " << (bytesA + bytesB + bytesC) / (avgMs * 1e6) << " GB/s" << std::endl;

    // 14. Cleanup (no need to free stack tensors)
    
    iface->contextFree(ctx, nullptr);
    if (device != nullptr) {
    iface->deviceFree(device);
    }
    iface->backendFree(backend);
    iface->logFree(logger);
    
    dlclose(libHandle);
    
    // Cleanup system interface (like ggml-hexagon)
    if (g_systemLibHandle) {
        dlclose(g_systemLibHandle);
        g_systemLibHandle = nullptr;
        g_systemInterface = nullptr;
    }
    
    return 0;
} 