/*
 * Copyright (c) 2024-2025 The KanTV authors
 *
 * implementation of matrix multiplication benchmark for LLM patterns
 * Tests: [seq x 4096] × [4096 x seq/4096/11008] and [seq x 11008] × [11008 x 4096]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>
#if defined(__ANDROID__) || defined(__linux__)
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <limits.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/types.h>
#endif

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <tuple>
#include <queue>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <memory>
#include <regex>
#include <random>
#include <functional>
#include <unordered_map>
#include <condition_variable>
#include <cassert>
#include <unordered_set>
#include <utility>
#include <algorithm>

#include "gguf.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_HEXAGON
#include "ggml-hexagon.h"
#endif

#define TMPBUF_LEN 256

struct BenchmarkResult {
    int pattern;
    int seq_len;
    int backend;
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

static const char* get_backend_name(int backend) {
#ifdef GGML_USE_HEXAGON
    return ggml_backend_hexagon_get_devname(backend);
#else
    switch(backend) {
        case 0: return "QNN_CPU";
        case 1: return "QNN_GPU";
        case 2: return "QNN_NPU";
        case 3: return "Hexagon-cDSP";
        case 4: return "ggml";
        default: return "Unknown";
                }
#endif
}

static void get_matrix_dims(int pattern, int seq_len, int* m, int* k, int* n) {
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
    }
}

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    size_t size = ggml_nelements(tensor);
    
    // For backend tensors (buffer != nullptr), use ggml_backend_tensor_set
    if (tensor->buffer != nullptr) {
        std::vector<float> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    } else {
        // For ggml tensors (buffer == nullptr), directly write to tensor->data
        if (tensor->data == nullptr) {
            printf("Warning: tensor %s has no data allocated\n", tensor->name ? tensor->name : "unnamed");
            return;
        }
        float* tensor_data = (float*)tensor->data;
        for (size_t i = 0; i < size; i++) {
            tensor_data[i] = dis(gen);
        }
    }
}

static void initialize_tensors(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        init_tensor_uniform(t);
    }
}

static BenchmarkResult run_single_benchmark(int pattern, int seq_len, int backend_type, int iterations = 10) {
    BenchmarkResult result = {pattern, seq_len, backend_type, 0.0, 0.0, false};
    
    int m, k, n;
    get_matrix_dims(pattern, seq_len, &m, &k, &n);
    
    // Check for problematic dimensions with QNN backends
#ifdef GGML_USE_HEXAGON
    if ((m == 1 || k == 1 || n == 1) && backend_type != HEXAGON_BACKEND_GGML) {
        return result; // Skip this test case
    }
#endif

    size_t ctx_size = (size_t)m * k * sizeof(float) + (size_t)k * n * sizeof(float) + (size_t)m * n * sizeof(float);
    ctx_size *= 2; // Add buffer
    const size_t min_ctx_size = 256 * 1024;
    if (ctx_size < min_ctx_size) {
        ctx_size = min_ctx_size;
    }

         struct ggml_init_params params = {
         /*.mem_size   =*/ ctx_size,
         /*.mem_buffer =*/ NULL,
         /* no_alloc   =*/ false  // Default: allocate memory for ggml backend
     };

#ifdef GGML_USE_HEXAGON
     if (backend_type != HEXAGON_BACKEND_GGML) {
         params.no_alloc = true;  // Hexagon backends: use no_alloc
     }
#endif

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return result;
    }

    // Create tensors
    ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_set_input(A);
    ggml_set_input(B);
    
    ggml_tensor * C = ggml_mul_mat(ctx, A, B);
    ggml_set_output(C);

    // Initialize backend
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

#ifdef GGML_USE_HEXAGON
    if (backend_type != HEXAGON_BACKEND_GGML) {
        if (backend_type >= HEXAGON_BACKEND_CDSP) {
            ggml_backend_hexagon_set_cfg(backend_type, HWACCEL_CDSP);
        } else {
            ggml_backend_hexagon_set_cfg(backend_type, HWACCEL_QNN);
        }
        
        backend = ggml_backend_hexagon_init(backend_type, "/data/local/tmp/");
        if (!backend) {
            ggml_free(ctx);
            return result;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buffer) {
            ggml_free(ctx);
            ggml_backend_free(backend);
            return result;
        }
         } else {
         backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
         // ggml backend: no buffer allocation needed (uses context memory)
     }
#else
     backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
     // ggml backend: no buffer allocation needed (uses context memory)
#endif

     if (!backend) {
         ggml_free(ctx);
         if (buffer) ggml_backend_buffer_free(buffer);
         return result;
     }

    // Create compute graph
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    // Initialize matrices
    initialize_tensors(ctx);

    // Run benchmark iterations
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        int64_t start_time = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        int64_t end_time = ggml_time_us();
        
        double time_ms = (end_time - start_time) / 1000.0;
        times.push_back(time_ms);
    }

    // Calculate average
    double total_time = 0.0;
    for (double time : times) {
        total_time += time;
    }
    result.avg_time_ms = total_time / iterations;
    
    // Calculate GFLOPS
    double gflops = (2.0 * m * k * n) / (result.avg_time_ms * 1e6);
    result.avg_gflops = gflops;
    result.success = true;

    // Cleanup
    ggml_free(ctx);
    if (buffer) ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    return result;
}

// Custom benchmark with user-specified matrix dimensions
static BenchmarkResult run_custom_benchmark(int m, int k, int n, int backend_type, int iterations = 10) {
    BenchmarkResult result = {-1, 0, backend_type, 0.0, 0.0, false}; // pattern = -1 for custom
    
    // Check for problematic dimensions with QNN backends
#ifdef GGML_USE_HEXAGON
    if ((m == 1 || k == 1 || n == 1) && backend_type != HEXAGON_BACKEND_GGML) {
        return result; // Skip this test case
    }
#endif

    size_t ctx_size = (size_t)m * k * sizeof(float) + (size_t)k * n * sizeof(float) + (size_t)m * n * sizeof(float);
    ctx_size *= 2; // Add buffer
    const size_t min_ctx_size = 256 * 1024;
    if (ctx_size < min_ctx_size) {
        ctx_size = min_ctx_size;
    }

         struct ggml_init_params params = {
         /*.mem_size   =*/ ctx_size,
         /*.mem_buffer =*/ NULL,
         /* no_alloc   =*/ false  // Default: allocate memory for ggml backend
     };

#ifdef GGML_USE_HEXAGON
     if (backend_type != HEXAGON_BACKEND_GGML) {
         params.no_alloc = true;  // Hexagon backends: use no_alloc
     }
#endif

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return result;
    }

    // Create tensors
    ggml_tensor * A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * B = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_set_input(A);
    ggml_set_input(B);
    
    ggml_tensor * C = ggml_mul_mat(ctx, A, B);
    ggml_set_output(C);

    // Initialize backend
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

#ifdef GGML_USE_HEXAGON
    if (backend_type != HEXAGON_BACKEND_GGML) {
        if (backend_type >= HEXAGON_BACKEND_CDSP) {
            ggml_backend_hexagon_set_cfg(backend_type, HWACCEL_CDSP);
        } else {
            ggml_backend_hexagon_set_cfg(backend_type, HWACCEL_QNN);
        }
        
        backend = ggml_backend_hexagon_init(backend_type, "/data/local/tmp/");
        if (!backend) {
            ggml_free(ctx);
            return result;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buffer) {
            ggml_free(ctx);
            ggml_backend_free(backend);
            return result;
        }
         } else {
         backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
         // ggml backend: no buffer allocation needed (uses context memory)
     }
#else
     backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
     // ggml backend: no buffer allocation needed (uses context memory)
#endif

     if (!backend) {
         ggml_free(ctx);
         if (buffer) ggml_backend_buffer_free(buffer);
         return result;
     }

    // Create compute graph
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, C);

    // Initialize matrices
    initialize_tensors(ctx);

    // Run benchmark iterations
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; i++) {
        int64_t start_time = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        int64_t end_time = ggml_time_us();
        
        double time_ms = (end_time - start_time) / 1000.0;
        times.push_back(time_ms);
    }

    // Calculate average
    double total_time = 0.0;
    for (double time : times) {
        total_time += time;
    }
    result.avg_time_ms = total_time / iterations;
    
    // Calculate GFLOPS
    double gflops = (2.0 * m * k * n) / (result.avg_time_ms * 1e6);
    result.avg_gflops = gflops;
    result.success = true;

    // Cleanup
    ggml_free(ctx);
    if (buffer) ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);

    return result;
}

static void show_usage() {
    printf("Matrix multiplication benchmark for LLM patterns\n\n");
    printf("Usage: matmul-benchmark [options]\n\n");
    printf("Options:\n");
         printf(" -b <backends> Backend list: 0(QNN_CPU) 1(QNN_GPU) 2(QNN_NPU) 3(Hexagon-cDSP) 4(ggml)\n");
     printf("               Examples: -b 2 (single) or -b 2,3,4 (multiple)\n");
     printf(" -a            Test all backends\n");
     printf(" -i <iters>    Number of iterations per test (default: 10)\n");
     printf(" -c <m,k,n>    Custom matrix dimensions [m x k] x [k x n]\n");
     printf("               Examples: -c 512,4096,4096 or -c 1024,11008,4096\n");
     printf(" -h/?          Show this help\n\n");
     printf("Default Test patterns:\n");
     printf(" 0: Attention    [seq x 4096] x [4096 x seq]\n");
     printf(" 1: Linear       [seq x 4096] x [4096 x 4096]\n");
     printf(" 2: FFN Up       [seq x 4096] x [4096 x 11008]\n");
     printf(" 3: FFN Down     [seq x 11008] x [11008 x 4096]\n\n");
     printf("Default sequence lengths: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096\n\n");
     printf("Note: Use -c for custom matrix size, or run default patterns without -c\n");
}

static void get_timestring(char * p_currenttime) {
    if (nullptr == p_currenttime)
        return;

    auto time_to_string = [](const std::chrono::system_clock::time_point & tp)->std::string {
        auto as_time_t = std::chrono::system_clock::to_time_t(tp);
        struct tm tm;
        localtime_r(&as_time_t, &tm);
        char buf[TMPBUF_LEN];
        memset(buf, 0, TMPBUF_LEN);
        snprintf(buf, sizeof(buf), "%04d-%02d-%02d,%02d:%02d:%02d",
                 tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
        return buf;
    };

    std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
    snprintf(p_currenttime, TMPBUF_LEN, "%s", time_to_string(tp).c_str());
}

// Parse backend list from string like "2,3,4"
static std::vector<int> parse_backend_list(const char* backend_str) {
    std::vector<int> backends;
    std::string str(backend_str);
    std::stringstream ss(str);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        int backend = atoi(item.c_str());
        if (backend >= 0 && backend <= 4) {
            backends.push_back(backend);
        } else {
            printf("Warning: Invalid backend %d, skipping\n", backend);
        }
    }
    return backends;
}

// Parse custom matrix dimensions from string like "512,4096,4096"
static bool parse_custom_dims(const char* dims_str, int* m, int* k, int* n) {
    std::vector<int> dims;
    std::string str(dims_str);
    std::stringstream ss(str);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        int dim = atoi(item.c_str());
        if (dim <= 0) {
            printf("Error: Invalid dimension %d\n", dim);
            return false;
        }
        dims.push_back(dim);
    }
    
    if (dims.size() != 3) {
        printf("Error: Expected 3 dimensions (m,k,n), got %zu\n", dims.size());
        return false;
    }
    
    *m = dims[0];
    *k = dims[1]; 
    *n = dims[2];
    return true;
}

int main(int argc, char * argv[]) {
    bool test_all_backends = false;
    std::vector<int> specified_backends;
    int iterations = 10;
    bool custom_mode = false;
    int custom_m = 0, custom_k = 0, custom_n = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (0 == strcmp(argv[i], "-a")) {
            test_all_backends = true;
        } else if (0 == strcmp(argv[i], "-b")) {
            if (i + 1 < argc) {
                specified_backends = parse_backend_list(argv[i+1]);
                if (specified_backends.empty()) {
                    printf("Error: No valid backends specified\n");
                    show_usage();
                    return 1;
                }
                i++;
            }
        } else if (0 == strcmp(argv[i], "-i")) {
            if (i + 1 < argc) {
                iterations = atoi(argv[i+1]);
                if (iterations < 1) iterations = 1;
                i++;
            }
        } else if (0 == strcmp(argv[i], "-c")) {
            if (i + 1 < argc) {
                if (!parse_custom_dims(argv[i+1], &custom_m, &custom_k, &custom_n)) {
                    printf("Error: Failed to parse custom dimensions\n");
                    show_usage();
                    return 1;
                }
                custom_mode = true;
                i++;
            } else {
                printf("Error: -c option requires dimensions (m,k,n)\n");
                show_usage();
                return 1;
            }
        } else if (0 == strcmp(argv[i], "-h") || 0 == strcmp(argv[i], "?")) {
            show_usage();
            return 0;
        }
    }

    if (!test_all_backends && specified_backends.empty()) {
        printf("Error: Must specify either -a (all backends) or -b <backends>\n\n");
        show_usage();
        return 1;
    }

    char timestring[TMPBUF_LEN];
    get_timestring(timestring);

    printf("=================================================================\n");
    printf("Matrix Multiplication Benchmark for LLM Patterns\n");
    printf("Started at: %s\n", timestring);
    printf("Iterations per test: %d\n", iterations);
    if (custom_mode) {
        printf("Custom matrix size: [%d x %d] x [%d x %d]\n", custom_m, custom_k, custom_k, custom_n);
    }
    printf("=================================================================\n\n");

    std::vector<int> backends;
    if (test_all_backends) {
        backends = {0, 1, 2, 3, 4};
    } else {
        backends = specified_backends;
    }

    std::vector<BenchmarkResult> all_results;

    if (custom_mode) {
        // Custom matrix benchmark mode
        int total_tests = backends.size();
        int current_test = 0;

        printf("Running custom matrix benchmark: [%d x %d] x [%d x %d]\n\n", custom_m, custom_k, custom_k, custom_n);
        
        printf("%-15s %-12s %-12s %-8s\n", "Backend", "Time(ms)", "GFLOPS", "Status");
        printf("----------------------------------------------------\n");
        
        for (int backend : backends) {
            current_test++;

            // Progress indicator
            printf("\r[%d/%d] Testing %s... ", current_test, total_tests, get_backend_name(backend));
            fflush(stdout);
            
            BenchmarkResult result = run_custom_benchmark(custom_m, custom_k, custom_n, backend, iterations);
            all_results.push_back(result);
            
            // Clear progress line and print result
            printf("\r%-15s ", get_backend_name(backend));
            if (result.success) {
                printf("%-12.2f %-12.2f %-8s\n", result.avg_time_ms, result.avg_gflops, "OK");
            } else {
                printf("%-12s %-12s %-8s\n", "FAILED", "FAILED", "SKIP");
            }
        }
        
        printf("\n=================================================================\n");
        printf("Custom Matrix Benchmark Results Summary\n");
        printf("Matrix: [%d x %d] x [%d x %d]\n", custom_m, custom_k, custom_k, custom_n);
        printf("=================================================================\n");
        printf("%-15s %-12s %-12s\n", "Backend", "Time(ms)", "GFLOPS");
        printf("--------------------------------------------\n");
        for (const auto& result : all_results) {
            if (result.success) {
                printf("%-15s %-12.2f %-12.2f\n", get_backend_name(result.backend), result.avg_time_ms, result.avg_gflops);
            } else {
                printf("%-15s %-12s %-12s\n", get_backend_name(result.backend), "FAILED", "FAILED");
            }
        }
    } else {
        // Default pattern benchmark mode
        std::vector<int> seq_lengths = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        std::vector<int> patterns = {0, 1, 2, 3};

        // Calculate total tests for progress
        int total_tests = patterns.size() * seq_lengths.size() * backends.size();
        int current_test = 0;

        printf("Running %d total test cases...\n\n", total_tests);

        for (int pattern : patterns) {
            printf("=== %s ===\n", get_pattern_name(pattern));
            
            for (int backend : backends) {
                printf("\nBackend: %s\n", get_backend_name(backend));
                printf("%-8s %-12s %-12s %-8s\n", "Seq", "Time(ms)", "GFLOPS", "Status");
                printf("--------------------------------------------\n");
                
                for (int seq_len : seq_lengths) {
                    current_test++;
        
                    // Progress indicator
                    printf("\r[%3d/%3d] Testing seq=%d... ", current_test, total_tests, seq_len);
                    fflush(stdout);
                    
                    BenchmarkResult result = run_single_benchmark(pattern, seq_len, backend, iterations);
                    all_results.push_back(result);
                    
                    // Clear progress line and print result
                    printf("\r%-8d ", seq_len);
                    if (result.success) {
                        printf("%-12.2f %-12.2f %-8s\n", result.avg_time_ms, result.avg_gflops, "OK");
            } else {
                        printf("%-12s %-12s %-8s\n", "FAILED", "FAILED", "SKIP");
                    }
                }
            }
            printf("\n");
        }

        printf("\n=================================================================\n");
        printf("SUMMARY REPORT - GFLOPS\n");
        printf("=================================================================\n");

        for (int pattern : patterns) {
            printf("\n%s:\n", get_pattern_name(pattern));
            printf("%-8s", "Seq\\BE");
            for (int backend : backends) {
                printf(" %-12s", get_backend_name(backend));
            }
            printf("\n");
            
            for (int seq_len : seq_lengths) {
                printf("%-8d", seq_len);
                for (int backend : backends) {
                    auto it = std::find_if(all_results.begin(), all_results.end(),
                        [=](const BenchmarkResult& r) {
                            return r.pattern == pattern && r.seq_len == seq_len && r.backend == backend;
                        });
                    
                    if (it != all_results.end() && it->success) {
                        printf(" %-12.2f", it->avg_gflops);
        } else {
                        printf(" %-12s", "FAILED");
                    }
                }
                printf("\n");
            }
        }

        printf("\n=================================================================\n");
        printf("SUMMARY REPORT - TIME (ms)\n");
        printf("=================================================================\n");

        for (int pattern : patterns) {
            printf("\n%s:\n", get_pattern_name(pattern));
            printf("%-8s", "Seq\\BE");
            for (int backend : backends) {
                printf(" %-12s", get_backend_name(backend));
            }
            printf("\n");
            
            for (int seq_len : seq_lengths) {
                printf("%-8d", seq_len);
                for (int backend : backends) {
                    auto it = std::find_if(all_results.begin(), all_results.end(),
                        [=](const BenchmarkResult& r) {
                            return r.pattern == pattern && r.seq_len == seq_len && r.backend == backend;
                        });
                    
                    if (it != all_results.end() && it->success) {
                        printf(" %-12.2f", it->avg_time_ms);
        } else {
                        printf(" %-12s", "FAILED");
                    }
                }
                printf("\n");
            }
        }
    }

    get_timestring(timestring);
    printf("\n=================================================================\n");
    printf("Benchmark completed at: %s\n", timestring);
    printf("=================================================================\n");

    return 0;
} 