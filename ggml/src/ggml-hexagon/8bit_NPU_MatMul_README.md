# Q4_0 Per-Group Int8 MatMul with NPU-CPU Hybrid Architecture

## 🎯 목표
- Q4_0 quantized weight × FP32 activation의 row 단위 per-group quantization 지원
- NPU의 int8 행렬곱과 CPU의 scale 처리를 분리하여 최적 성능 달성
- Element-wise 연산을 통한 정확한 dequantization과 group별 accumulation

## 📊 핵심 아키텍처

### 기본 접근법
```
Weight (Q4_0 per-group) × Activation (FP32) = Output (FP32)
```

### Row 단위 Group 분할
- **Weight**: Row 단위로 group 분할 (각 group = 32 elements)
- **Activation**: Column 유지, Weight group에 맞춰 row 분할
- **Scale Matrix**: Weight의 각 group에 대응하는 scale 값

## 🔄 연산 Flow

### 1. 데이터 준비 단계
```
Input:
- Weight: Q4_0 format (quantized values + scales per group)
- Activation: FP32 format (transpose된 상태로 저장)

Preparation:
- Weight를 row 단위 group으로 분할
- 각 group의 quantized values 추출
- 각 group의 scale 값으로 scale matrix 생성
```

### 2. NPU Int8 행렬곱 수행
```
For each group:
  NPU: int8_weight_group × int8_activation_group = int32_result_group
```

### 3. CPU Scale Matrix 생성 및 적용
```
CPU: weight_scales × activation_scales = combined_scales_matrix
NPU: int32_result ⊗ combined_scales_matrix = fp32_dequantized_result
```

### 4. Group별 Accumulation
```
NPU: group_results[0] ⊕ group_results[1] ⊕ ... ⊕ group_results[n] = final_output
```

## 📋 구현 계획

### **Phase 1: 데이터 구조 및 Group 분할**
- [ ] Q4_0 group tiling 구조체 설계
- [ ] Row 단위 group 분할 알고리즘
- [ ] Scale matrix 생성 로직

### **Phase 2: NPU Int8 행렬곱 엔진**
- [ ] Group별 int8 matmul QNN graph 생성
- [ ] Batch processing 최적화
- [ ] Memory layout 최적화

### **Phase 3: CPU Scale Matrix 생성**
- [ ] Per-group scale 추출
- [ ] Broadcasting 최적화
- [ ] CPU-NPU 동기화

### **Phase 4: NPU Element-wise 연산**
- [ ] Dequantization (int32 → fp32 conversion)
- [ ] Element-wise multiplication
- [ ] Element-wise accumulation

### **Phase 5: 메인 통합 함수**
- [ ] End-to-end pipeline 구현
- [ ] Error handling 및 validation
- [ ] Performance profiling

## 🚀 구현 로드맵 (8주)

| 주차 | Phase | 주요 작업 | 산출물 |
|------|-------|-----------|---------|
| **Week 1-2** | Phase 1 | 기본 인프라 구축 | Group 분할 알고리즘, 데이터 구조 |
| **Week 3-4** | Phase 2 | NPU 연산 엔진 | Int8 matmul QNN graph |
| **Week 5-6** | Phase 3-4 | CPU scale + NPU element-wise | Scale matrix 생성, Element-wise 연산 |
| **Week 7-8** | Phase 5 | 통합 및 최적화 | 완전한 Q4_0 int8 matmul 함수 |

## 🔍 기존 Hexagon Backend FP32 MatMul 분석

### **QNN SDK API 사용 패턴 분석**

#### **1. QNN Instance 관리 구조**
```cpp
class qnn_instance {
    // 핵심 QNN 핸들들
    Qnn_GraphHandle_t _qnn_graph_handle;
    Qnn_ContextHandle_t _qnn_context_handle;
    Qnn_BackendHandle_t _qnn_backend_handle;
    Qnn_DeviceHandle_t _qnn_device_handle;
    
    // 동적 라이브러리 및 인터페이스
    QNN_INTERFACE_VER_TYPE _qnn_raw_interface;
    qnn_interface _qnn_interface;  // Function pointer wrapper
};
```

#### **2. QNN Graph 생성 패턴**
```cpp
// 위치: ggml-hexagon.cpp:3706-3800
int qnn_instance::init_qnn_graph(const std::string & graph_name, 
                                  HEXAGONBackend device, 
                                  size_t vtcm_size_in_mb, 
                                  size_t hvx_threads) {
    // 1. Graph 생성
    error = _qnn_raw_interface.graphCreate(_qnn_context_handle, 
                                          graph_name.c_str(), 
                                          NULL, &_qnn_graph_handle);
    
    // 2. HTP 성능 설정
    htp_set_memory_grow_size(size);
    htp_set_n_hvx_threads(n_threads);
    
    return QNN_SUCCESS;
}
```

#### **3. QNN Tensor 생성 핵심 함수**
```cpp
// 위치: ggml-hexagon.cpp:4089-4170
static Qnn_Tensor_t * ggmlqnn_create_general_tensor(
    qnn_instance * instance, 
    Qnn_GraphHandle_t graph_handle,
    const ggml_tensor * tensor, 
    const char * name,
    Qnn_TensorType_t qnn_tensor_type,    // APP_WRITE/APP_READ/NATIVE
    Qnn_DataType_t qnn_data_type,        // FLOAT_32/INT8_t/etc
    uint32_t rank, 
    uint32_t * dims,
    void * data, 
    uint32_t data_size) {
    
    Qnn_Tensor_t qnn_tensor = {
        .version = QNN_TENSOR_VERSION_1,
        .v1 = {
            .id = 0,
            .name = tensor_name,
            .type = qnn_tensor_type,
            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType = qnn_data_type,
            .rank = rank,
            .dimensions = tensor_dims,
            .memType = QNN_TENSORMEMTYPE_RAW,  // 또는 MEMHANDLE
            .clientBuf = {.data = data, .dataSize = data_size}
        }
    };
    
    // QNN Graph에 등록
    _qnn_interface.tensorCreateGraphTensor(graph_handle, p_qnn_tensor);
    return p_qnn_tensor;
}
```

### **ION 메모리 관리 상세 분석**

#### **1. ION 메모리가 필요한 이유 확인됨**
```cpp
// 위치: ggml-hexagon.cpp:3078-3130
// QNN SDK는 NPU 사용 시 ION 메모리 필수 사용
Qnn_MemDescriptor_t descriptor = {
    {rank, dimensions, nullptr},
    data_type, 
    QNN_MEM_TYPE_ION,      // ← ION 메모리 타입 필수!
    {{mem_fd}}             // File descriptor 필요
};
```

#### **2. RPC 메모리 할당 패턴**
```cpp
// 위치: ggml-hexagon.cpp:2960-2995
void * qnn_instance::alloc_rpcmem(size_t bytes, size_t alignment) {
    // 1. RPC 메모리 할당 (ION 기반)
    void * buf = _pfn_rpc_mem_alloc(RPCMEM_HEAP_ID_SYSTEM, 
                                   RPCMEM_DEFAULT_FLAGS, 
                                   allocate_bytes);
    
    // 2. 정렬된 포인터 계산
    auto aligned_buf = reinterpret_cast<void *>(
        ggmlqnn_align_to(alignment, reinterpret_cast<intptr_t>(buf)));
    
    // 3. 매핑 테이블에 저장
    _rpcmem_store_map.insert({aligned_buf, buf});
    _rpcmem_usage_map.insert({aligned_buf, bytes});
    
    return aligned_buf;
}
```

#### **3. ION 메모리 등록 과정**
```cpp
// 위치: ggml-hexagon.cpp:3045-3090
Qnn_MemHandle_t qnn_instance::register_rpcmem(void * p_data, 
                                              const uint32_t rank, 
                                              uint32_t * dimensions, 
                                              Qnn_DataType_t data_type) {
    // 1. RPC 메모리를 File Descriptor로 변환
    int32_t mem_fd = rpcmem_to_fd(p_data);
    
    // 2. QNN Memory Descriptor 생성
    Qnn_MemDescriptor_t descriptor = {
        {rank, dimensions, nullptr},
        data_type, 
        QNN_MEM_TYPE_ION,
        {{mem_fd}}
    };
    
    // 3. QNN Context에 메모리 등록
    error = _qnn_interface.qnn_mem_register(_qnn_context_handle, 
                                           &descriptor, 1, &handle);
    
    return handle;
}
```

### **QNN Graph 실행 흐름 분석**

#### **1. Element-wise 연산 예제 (Add/Mul/etc)**
```cpp
// 위치: ggml-hexagon.cpp:4206-4358
static void ggmlqnn_compute_elementwise(ggml_backend_hexagon_context * ctx, 
                                        ggml_tensor * op) {
    // Graph 캐싱 확인
    if (ctx->qnn_singlenode_graph_map.find(graph_name) != 
        ctx->qnn_singlenode_graph_map.end()) {
        // 기존 Graph 재사용
        graph_handle = std::get<0>(graph_item);
        // Tensor들 재사용
    } else {
        // 새 Graph 생성
        // 1. Tensor 생성
        p_tensor0 = ggmlqnn_create_compute_tensor(instance, graph_handle, src0, 
                                                 QNN_TENSOR_TYPE_APP_WRITE);
        p_tensor1 = ggmlqnn_create_compute_tensor(instance, graph_handle, src1, 
                                                 QNN_TENSOR_TYPE_APP_WRITE);
        p_tensor2 = ggmlqnn_create_compute_tensor(instance, graph_handle, dst, 
                                                 QNN_TENSOR_TYPE_APP_READ);
        
        // 2. Op Config 생성
        Qnn_OpConfig_t op_config = ggmlqnn_create_op_config(
            ggml_op_name, QNN_OP_PACKAGE_NAME_QTI_AISW, qnn_op_name, 
            nullptr, 0, input_tensors.data(), input_param_count, 
            output_tensors, 1);
        
        // 3. Graph에 Node 추가
        _qnn_interface.graphAddNode(graph_handle, op_config);
        
        // 4. Graph Finalize
        _qnn_interface.graphFinalize(graph_handle, nullptr, nullptr);
        
        // 5. Graph 캐싱
        ctx->qnn_singlenode_graph_map[graph_name] = 
            std::make_tuple(graph_handle, qnn_elementwise_tensors);
    }
    
    // 데이터 설정
    if (enable_npu_rpc) {
        // ION 메모리 사용 시
        uint8_t * qnn_buffer = instance->get_rpcmem_from_memhandle(
            QNN_VER_PTR(*p_tensor0)->memHandle);
        memcpy(qnn_buffer, src0->data, ggml_nbytes(src0));
    } else {
        // 일반 메모리 사용 시
        QNN_VER_PTR(*p_tensor0)->clientBuf = {src0->data, data_size};
    }
    
    // Graph 실행
    _qnn_interface.graphExecute(graph_handle, input_tensors.data(), 
                               input_param_count, output_tensors, 1, 
                               nullptr, nullptr);
}
```

#### **2. Matrix Multiplication 복잡한 예제**
```cpp
// 위치: ggml-hexagon.cpp:4359-4590 (4D MatMul)
// 복잡한 transpose, reshape, tile 연산들을 조합
// 1. Reshape: [B0, H0, M, K] → [B0, M, K]
// 2. Tile: [B0, M, K] → [B1, M, K] (broadcasting)
// 3. Permute: [B1, H1, N, K] → [B1, H1, K, N]
// 4. MatMul: [B1, M, K] × [B1, K, N] → [B1, M, N]
// 5. Reshape: [B1, M, N] → [B1, H1, M, N]
```

### **메모리 관리 Best Practices**

#### **1. Memory Alignment 처리**
```cpp
// 위치: ggml-hexagon.cpp:2234-2240
static intptr_t ggmlqnn_align_to(size_t alignment, intptr_t offset) {
    return ((offset + alignment - 1) / alignment) * alignment;
}
```

#### **2. Memory Pool 관리**
```cpp
// RPC 메모리 풀 용량 제한 및 사용량 추적
if (_rpcmem_usage > (_rpcmem_capacity - (8 * SIZE_IN_MB))) {
    // 8MB 여유 공간 확보
    return nullptr;
}
```

#### **3. Resource 정리**
```cpp
// Memory handle 해제
void qnn_instance::unregister_rpcmem(Qnn_MemHandle_t mem_handle) {
    _qnn_interface.qnn_mem_de_register(&mem_handle, 1);
    _qnn_mem_set.erase(mem_handle);
}
```

## 💡 NPU Int8 MatMul 구현 핵심 인사이트

### **1. QNN Tensor Type 전략**
- **Input Weight/Activation**: `QNN_TENSOR_TYPE_APP_WRITE`
- **Intermediate Results**: `QNN_TENSOR_TYPE_NATIVE` 
- **Final Output**: `QNN_TENSOR_TYPE_APP_READ`

### **2. Memory Type 선택**
- **ION 메모리 필수**: NPU 사용 시 `QNN_TENSORMEMTYPE_MEMHANDLE`
- **일반 메모리**: CPU 사용 시 `QNN_TENSORMEMTYPE_RAW`

### **3. Graph Caching 최적화**
```cpp
// Graph 재사용으로 오버헤드 최소화
std::map<std::string, qnn_singlenode_res_t> qnn_singlenode_graph_map;
```

### **4. Error Handling 패턴**
```cpp
#define CHECK_QNN_API(error, result) \
    do { \
        error = (result); \
        if (QNN_SUCCESS != error) { \
            GGMLHEXAGON_LOG_WARN("QNN API error = %d(%s)\n", \
                error, ggmlqnn_get_qnnerror_string(error)); \
        } \
    } while (0)
```

### **5. Performance Profiling 구조**
```cpp
hexagon_perf op_perf(graph_name, ggml_original_opname, input_size, output_size);
op_perf.start();
// ... QNN operations ...
op_perf.info(); // 성능 측정 결과 출력
```

## 🔍 기존 Hexagon Backend FP32 MatMul 분석

### **QNN SDK API 사용 패턴 분석**

#### **1. QNN Instance 관리 구조**
```cpp
class qnn_instance {
    // 핵심 QNN 핸들들
    Qnn_GraphHandle_t _qnn_graph_handle;
    Qnn_ContextHandle_t _qnn_context_handle;
    Qnn_BackendHandle_t _qnn_backend_handle;
    Qnn_DeviceHandle_t _qnn_device_handle;
    
    // 동적 라이브러리 및 인터페이스
    QNN_INTERFACE_VER_TYPE _qnn_raw_interface;
    qnn_interface _qnn_interface;  // Function pointer wrapper
};
```

#### **2. QNN Graph 생성 패턴**
```cpp
// 위치: ggml-hexagon.cpp:3706-3800
int qnn_instance::init_qnn_graph(const std::string & graph_name, 
                                  HEXAGONBackend device, 
                                  size_t vtcm_size_in_mb, 
                                  size_t hvx_threads) {
    // 1. Graph 생성
    error = _qnn_raw_interface.graphCreate(_qnn_context_handle, 
                                          graph_name.c_str(), 
                                          NULL, &_qnn_graph_handle);
    
    // 2. HTP 성능 설정
    htp_set_memory_grow_size(size);
    htp_set_n_hvx_threads(n_threads);
    
    return QNN_SUCCESS;
}
```

#### **3. QNN Tensor 생성 핵심 함수**
```cpp
// 위치: ggml-hexagon.cpp:4089-4170
static Qnn_Tensor_t * ggmlqnn_create_general_tensor(
    qnn_instance * instance, 
    Qnn_GraphHandle_t graph_handle,
    const ggml_tensor * tensor, 
    const char * name,
    Qnn_TensorType_t qnn_tensor_type,    // APP_WRITE/APP_READ/NATIVE
    Qnn_DataType_t qnn_data_type,        // FLOAT_32/INT8_t/etc
    uint32_t rank, 
    uint32_t * dims,
    void * data, 
    uint32_t data_size) {
    
    Qnn_Tensor_t qnn_tensor = {
        .version = QNN_TENSOR_VERSION_1,
        .v1 = {
            .name = tensor_name,
            .type = qnn_tensor_type,
            .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            .dataType = qnn_data_type,
            .rank = rank,
            .dimensions = tensor_dims,
            .memType = QNN_TENSORMEMTYPE_RAW,  // 또는 MEMHANDLE
            .clientBuf = {.data = data, .dataSize = data_size}
        }
    };
    
    // QNN Graph에 등록
    _qnn_interface.tensorCreateGraphTensor(graph_handle, p_qnn_tensor);
    return p_qnn_tensor;
}
```

### **ION 메모리 관리 상세 분석**

#### **1. ION 메모리가 필요한 이유 확인됨**
```cpp
// QNN SDK는 NPU 사용 시 ION 메모리 필수 사용
Qnn_MemDescriptor_t descriptor = {
    {rank, dimensions, nullptr},
    data_type, 
    QNN_MEM_TYPE_ION,      // ← ION 메모리 타입 필수!
    {{mem_fd}}             // File descriptor 필요
};
```

#### **2. RPC 메모리 할당 패턴**
```cpp
// RPC 메모리 = ION 메모리
void * qnn_instance::alloc_rpcmem(size_t bytes, size_t alignment) {
    // 1. RPC 메모리 할당 (ION 기반)
    void * buf = _pfn_rpc_mem_alloc(RPCMEM_HEAP_ID_SYSTEM, 
                                   RPCMEM_DEFAULT_FLAGS, 
                                   allocate_bytes);
    
    // 2. 정렬된 포인터 계산
    auto aligned_buf = reinterpret_cast<void *>(
        ggmlqnn_align_to(alignment, reinterpret_cast<intptr_t>(buf)));
    
    // 3. 매핑 테이블에 저장
    _rpcmem_store_map.insert({aligned_buf, buf});
    
    return aligned_buf;
}
```

#### **3. ION 메모리 등록 과정**
```cpp
Qnn_MemHandle_t qnn_instance::register_rpcmem(void * p_data, 
                                              const uint32_t rank, 
                                              uint32_t * dimensions, 
                                              Qnn_DataType_t data_type) {
    // 1. RPC 메모리를 File Descriptor로 변환
    int32_t mem_fd = rpcmem_to_fd(p_data);
    
    // 2. QNN Memory Descriptor 생성
    Qnn_MemDescriptor_t descriptor = {
        {rank, dimensions, nullptr},
        data_type, QNN_MEM_TYPE_ION, {{mem_fd}}
    };
    
    // 3. QNN Context에 메모리 등록
    error = _qnn_interface.qnn_mem_register(_qnn_context_handle, 
                                           &descriptor, 1, &handle);
    
    return handle;
}
```

### **QNN Graph 실행 흐름 분석**

#### **1. Graph 생성 → 실행 패턴**
```cpp
// 1. Graph 캐싱 확인
if (ctx->qnn_singlenode_graph_map.find(graph_name) != 
    ctx->qnn_singlenode_graph_map.end()) {
    // 기존 Graph 재사용
} else {
    // 2. 새 Graph 생성
    // - Tensor 생성 (ggmlqnn_create_compute_tensor)
    // - Op Config 생성 (ggmlqnn_create_op_config)
    // - Graph에 Node 추가 (graphAddNode)
    // - Graph Finalize (graphFinalize)
    // - Graph 캐싱
}

// 3. 데이터 설정
if (enable_npu_rpc) {
    // ION 메모리 사용 시 - memcpy 필요
    uint8_t * qnn_buffer = instance->get_rpcmem_from_memhandle(handle);
    memcpy(qnn_buffer, src_data, data_size);
} else {
    // 일반 메모리 사용 시 - pointer 설정
    QNN_VER_PTR(*tensor)->clientBuf = {src_data, data_size};
}

// 4. Graph 실행
_qnn_interface.graphExecute(graph_handle, inputs, input_count, 
                           outputs, output_count, nullptr, nullptr);
```

### **메모리 관리 핵심 인사이트**

#### **1. ION 메모리 필수 사용 확인**
- NPU backend 사용 시 `QNN_MEM_TYPE_ION` 필수
- RPC 메모리 = ION 메모리 (Android/Linux 환경)
- File descriptor 기반 메모리 공유

#### **2. Memory Handle 관리**
```cpp
// Memory handle과 pointer 양방향 매핑
std::unordered_map<void *, Qnn_MemHandle_t> _qnn_mem_set;
std::unordered_map<void *, Qnn_MemHandle_t> _qnn_rpc_buffer_to_handles;
```

#### **3. Memory Alignment 및 Pool 관리**
```cpp
// 정렬 처리
static intptr_t ggmlqnn_align_to(size_t alignment, intptr_t offset);

// Pool 사용량 제한 (8MB 여유 공간 확보)
if (_rpcmem_usage > (_rpcmem_capacity - (8 * SIZE_IN_MB))) {
    return nullptr;
}
```

---

## 🎯 **다음 단계: Q4_0 Int8 MatMul 구현 시작**

위 분석을 바탕으로 다음 순서로 구현을 진행합니다:

1. **기존 QNN infrastructure 활용**
2. **Group 단위 tiling 로직 구현**  
3. **NPU int8 matmul Graph 생성**
4. **CPU scale matrix 연산과 NPU element-wise 연산 통합**

모든 구현은 기존 `ggmlqnn_compute_elementwise` 및 `ggmlqnn_compute_mul_mat` 패턴을 따라 일관성 있는 코드 스타일을 유지할 예정입니다.
