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

### 2. NPU Int8 행렬곱 단계  
```
For each group (32 elements):
  - Weight group (int8) × Activation slice (quantized to int8)
  - Result: int32 intermediate results
```

### 3. CPU Scale Matrix 생성
```
- Weight scale × Activation scale matrix 생성
- Broadcasting을 통해 올바른 차원으로 확장
```

### 4. NPU Element-wise 연산
```
- int32 결과를 float로 dequantization (NPU)
- Scale matrix와 element-wise 곱셈 (NPU)
- Group 결과들을 element-wise 덧셈으로 accumulation (NPU)
```

## 🎨 참고 아키텍처: Android ARM CPU 구현 분석

### ARM Q4_0 x Q8_0 구현의 핵심 인사이트

#### 1. **Generic Dot Product 패턴** (ggml-cpu/quants.c)
```c
// Q4_0 block 구조: {fp16 scale, uint8 qs[16]} - 32개 4-bit 값들
for (block_idx = 0; block_idx < num_blocks; ++block_idx) {
    int sumi0 = 0, sumi1 = 0;
    
    for (j = 0; j < 16; ++j) {
        // 4-bit -> signed int8 변환 (subtract 8 for offset)
        int v0 = (x[block_idx].qs[j] & 0x0F) - 8;  // lower 4-bit
        int v1 = (x[block_idx].qs[j] >>   4) - 8;  // upper 4-bit
        
        // 16개씩 나누어 dot product
        sumi0 += v0 * y[block_idx].qs[j];
        sumi1 += v1 * y[block_idx].qs[j + 16];
    }
    
    // Per-block scale 적용
    result += (sumi0 + sumi1) * weight_scale * activation_scale;
}
```

#### 2. **ARM NEON 벡터화** (ggml-cpu/arch/arm/quants.c)
```c
// 16바이트 SIMD 로딩
uint8x16_t v0_0 = vld1q_u8(x0->qs);

// 4-bit -> 8-bit 변환
int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));    // lower 4-bit
int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));   // upper 4-bit

// Bias 제거 (subtract 8)
int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);

// NEON dot product + scale 적용
int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), scale_combined);
```

#### 3. **Matrix Repack 전략** (ggml-cpu/arch/arm/repack.cpp)
```c
// Activation을 4x4, 4x8, 8x8 타일로 repack
void ggml_quantize_mat_q8_0_4x4(const float * x, void * vy, int64_t k) {
    // 4개 row를 interleave하여 메모리 접근 최적화
    for (int row_iter = 0; row_iter < 4; row_iter++) {
        // Per-row scale 계산
        float amax = find_max_abs(row_data);
        float scale = amax / 127.0f;
        
        // Quantization with interleaved storage
        for (int j = 0; j < elements_per_row; j++) {
            int src_id = (j % 16) / 4;  // interleaving pattern
            quantized_output[interleaved_index] = round(input * inv_scale);
        }
    }
}

// Optimized GEMV/GEMM with tiled processing
void ggml_gemv_q4_0_4x4_q8_0(int n, float * s, const void * vx, const void * vy, int nr, int nc) {
    // ARM NEON dot product with 4x4 tiles
    // 메모리 접근 패턴이 cache-friendly하도록 최적화
}
```

#### 4. **Type Traits 시스템** (ggml-cpu/ggml-cpu.c)
```c
[GGML_TYPE_Q4_0] = {
    .from_float    = quantize_row_q4_0,
    .vec_dot       = ggml_vec_dot_q4_0_q8_0,  // architecture-specific
    .vec_dot_type  = GGML_TYPE_Q8_0,
    .nrows         = 1,  // or 2 with ARM_FEATURE_MATMUL_INT8
}
```

### NPU 구현에 적용할 핵심 인사이트

#### 1. **메모리 레이아웃 최적화**
- ARM의 interleaved storage 패턴을 NPU tensor layout에 적용
- Group 단위 처리를 위한 efficient memory tiling

#### 2. **Scale 처리 분리**
- ARM처럼 integer 연산과 scale 적용을 분리
- NPU int8 matmul → CPU scale matrix → NPU element-wise ops

#### 3. **Vector 연산 패턴**
- 4-bit unpacking을 NPU reshape/slice 연산으로 대체
- Bias subtraction을 NPU add 연산으로 처리

#### 4. **Chunked Processing**
- ARM의 block-wise 처리를 NPU group-wise 처리로 확장
- Memory bandwidth 최적화를 위한 적절한 chunk size 선택

## 📋 구현 시 핵심 참고사항

### 🔍 **참고할 코드 위치 및 함수들**

#### **Q4_0 데이터 구조 이해**
```c
// 위치: ggml/src/ggml-common.h:165-175
typedef struct {
    ggml_half d;           // scale factor (FP16)
    uint8_t qs[QK4_0 / 2]; // 16 bytes = 32 nibbles (4-bit values)
} block_q4_0;

// QK4_0 = 32 (group size)
// 각 block은 32개의 4-bit 값 + 1개의 scale 값
```

#### **ARM 4-bit Unpacking 패턴**
```c
// 참고: ggml-cpu/arch/arm/quants.c:175-190
// NPU 구현에서 reshape/slice 연산으로 대체할 패턴
const uint8x16_t m4b = vdupq_n_u8(0x0F);        // mask for lower 4-bit
const int8x16_t s8b = vdupq_n_s8(0x8);          // bias offset

// Lower 4-bit 추출
int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
// Upper 4-bit 추출  
int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));

// Bias 제거 (4-bit unsigned → signed int8)
int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);  // subtract 8
int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
```

#### **메모리 인터리빙 패턴**
```c
// 참고: ggml-cpu/arch/arm/repack.cpp:70-120
// NPU tensor layout 최적화에 활용할 패턴
const int blck_size_interleave = 4;  // 4x4 tiling

for (int j = 0; j < QK8_0 * 4; j++) {
    int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
    int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
    src_offset += (j % blck_size_interleave);
    
    // Interleaved memory access for cache efficiency
    float x0 = srcv[src_id][src_offset] * id[src_id];
    y[i].qs[j] = roundf(x0);
}
```

#### **Scale 처리 분리 패턴**
```c
// 참고: ggml-cpu/quants.c:126-137
// CPU에서 scale matrix 생성할 때 참고할 패턴
for (int ib = 0; ib < nb; ++ib) {
    // 1. 먼저 integer 연산 수행
    int sumi0 = 0, sumi1 = 0;
    for (int j = 0; j < qk/2; ++j) {
        const int v0 = (x[ib].qs[j] & 0x0F) - 8;
        const int v1 = (x[ib].qs[j] >>   4) - 8;
        sumi0 += (v0 * y[ib].qs[j]);
        sumi1 += (v1 * y[ib].qs[j + qk/2]);
    }
    
    // 2. 나중에 scale 적용
    int sumi = sumi0 + sumi1;
    sumf += sumi * GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);
}
```

### 🛠️ **메모리 관리 Best Practices**

#### **1. NPU Tensor 레이아웃 최적화**
```c
// ARM repack 패턴을 NPU에 적용
typedef struct {
    // Group-wise storage for efficient NPU access
    int8_t *weight_groups[MAX_GROUPS];     // 각 group별 int8 weights
    float *scale_values[MAX_GROUPS];       // 각 group별 scale values
    size_t group_size;                     // 32 for Q4_0
    size_t total_groups;
    
    // NPU-friendly layout
    hexagon_tensor_layout npu_layout;      // NPU optimal memory layout
} hexagon_q4_0_memory_layout;
```

#### **2. 메모리 할당 전략**
```c
// 참고: ggml-cpu/ggml-cpu.c:1513-1520
// 메모리 정렬 및 효율적 할당 패턴
static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {
    void * ptr = *p;
    size_t offset = (uintptr_t)ptr % align;
    if (offset != 0) {
        ptr = (char *)ptr + (align - offset);
    }
    *p = (char *)ptr + size;
    return ptr;
}

// NPU 구현에서 활용
void * npu_buffer = hexagon_aligned_alloc(total_size, NPU_ALIGNMENT);
int8_t * weight_buffer = incr_ptr_aligned(&npu_buffer, weight_size, 128);
float * scale_buffer = incr_ptr_aligned(&npu_buffer, scale_size, 64);
```

#### **3. 캐시 효율적 접근 패턴**
```c
// 참고: ggml-cpu/arch/arm/repack.cpp:238-280
// NPU에서 group 단위 처리할 때 적용할 패턴
void process_q4_0_groups_cache_friendly(
    const block_q4_0 *weights,
    const float *activations,
    float *output,
    const hexagon_cache_config *config
) {
    // 1. Group-wise prefetching
    for (int group_idx = 0; group_idx < num_groups; group_idx += PREFETCH_GROUPS) {
        // 2. Cache-line aligned processing
        for (int i = 0; i < PREFETCH_GROUPS && (group_idx + i) < num_groups; i++) {
            // 3. NPU 연산 수행
            process_single_group(weights + group_idx + i, activations, output);
        }
    }
}
```

### ⚡ **성능 최적화 핵심 팁**

#### **1. NEON → NPU 연산 변환 패턴**
```c
// ARM NEON 패턴:
int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);

// NPU 등가 연산:
hexagon_tensor_t int_result = hexagon_npu_matmul_int8(
    weight_int8_tensor,      // v0_0ls, v0_0hs 대응
    activation_int8_tensor,  // v1_0l, v1_0h 대응
    &npu_config
);
```

#### **2. Scale Broadcasting 최적화**
```c
// ARM에서 scale 적용 패턴:
sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), scale_combined);

// NPU 구현:
hexagon_tensor_t float_result = hexagon_npu_convert_int32_to_float(int_result);
hexagon_tensor_t scaled_result = hexagon_npu_elementwise_mul(
    float_result, 
    scale_matrix_broadcasted
);
```

#### **3. Group별 Accumulation 패턴**
```c
// 참고: ggml-cpu/ops.cpp:1124-1196 (mul_mat computation)
// NPU에서 group 결과들을 효율적으로 accumulate
for (int group = 0; group < num_groups; group++) {
    // Group별 NPU 연산
    hexagon_tensor_t group_result = process_group(group);
    
    // 효율적 accumulation (in-place operation)
    if (group == 0) {
        hexagon_npu_copy(group_result, &accumulated_result);
    } else {
        hexagon_npu_elementwise_add_inplace(&accumulated_result, group_result);
    }
}
```

### 🔧 **구현 시 주의사항**

#### **1. Q4_0 Offset 처리**
```c
// 중요: Q4_0는 4-bit unsigned 값에서 -8 bias를 적용
// ARM 구현에서: (data & 0x0F) - 8
// NPU에서는 별도 tensor operation으로 처리 필요

hexagon_tensor_t unpack_4bit_to_int8_with_bias(
    const hexagon_tensor_t *packed_4bit,
    int8_t bias_value  // -8 for Q4_0
) {
    // 1. 4-bit unpack
    // 2. Bias subtract
    // 3. Return signed int8 tensor
}
```

#### **2. Scale 정밀도 유지**
```c
// ARM에서 FP16 scale → FP32 변환 패턴
// 참고: ggml-cpu/quants.c:137
float scale = GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d);

// NPU 구현에서도 정밀도 유지 필요
float weight_scale = fp16_to_fp32(q4_block.d);
float combined_scale = weight_scale * activation_scale;
```

#### **3. 메모리 정렬 요구사항**
```c
// Hexagon NPU 메모리 정렬 요구사항 준수
#define HEXAGON_NPU_ALIGNMENT 128
#define HEXAGON_VECTOR_ALIGNMENT 64

void * allocate_npu_buffer(size_t size) {
    return hexagon_aligned_alloc(size, HEXAGON_NPU_ALIGNMENT);
}
```

### 🧪 **검증 및 디버깅 전략**

#### **1. 단계별 검증**
```c
// Phase 1: Q4_0 → int8 변환 검증
void verify_q4_0_conversion(const block_q4_0 *input, const int8_t *output) {
    // ARM generic 구현과 결과 비교
    float arm_result = ggml_vec_dot_q4_0_q8_0_generic(...);
    float npu_result = hexagon_q4_0_matmul(...);
    assert(fabs(arm_result - npu_result) < TOLERANCE);
}

// Phase 2: Scale matrix 검증
void verify_scale_matrix(const float *expected, const float *actual, size_t size) {
    for (size_t i = 0; i < size; i++) {
        assert(fabs(expected[i] - actual[i]) < SCALE_TOLERANCE);
    }
}
```

#### **2. 성능 프로파일링**
```c
// ARM baseline 측정
uint64_t arm_start = get_time_ns();
ggml_vec_dot_q4_0_q8_0(n, &result, 0, weights, 0, activations, 0, 1);
uint64_t arm_time = get_time_ns() - arm_start;

// NPU 구현 측정
uint64_t npu_start = get_time_ns();
hexagon_q4_0_matmul_hybrid(weights, activations, &result, &params);
uint64_t npu_time = get_time_ns() - npu_start;

printf("Speedup: %.2fx\n", (double)arm_time / npu_time);
```

## 🚀 구현 로드맵

### Phase 1: 데이터 구조 및 Group 분할 (Week 1-2)
```c
typedef struct {
    int32_t *quantized_values;  // NPU int8 tensor
    float *scale_matrix;        // CPU-generated scales  
    int group_size;             // 32 for Q4_0
    int num_groups;
} hexagon_q4_0_tiles;

// Q4_0 → int8 변환 (ARM 4-bit unpacking 패턴 참고)
void convert_q4_0_to_int8_tiles(const block_q4_0 *q4_weights, hexagon_q4_0_tiles *tiles);
```

### Phase 2: NPU Int8 행렬곱 엔진 (Week 3-4)
```c
// ARM NEON dot product 패턴을 NPU 연산으로 변환
bool hexagon_npu_int8_matmul_tiled(
    const int8_t *weight_tiles,     // group-wise int8 weights
    const int8_t *activation_slice, // quantized activations
    int32_t *output_buffer,         // intermediate int32 results
    const hexagon_tile_config *config
);
```

### Phase 3: CPU Scale Matrix 생성 (Week 5)
```c
// ARM scale combining 패턴 참고
void generate_scale_matrix(
    const float *weight_scales,     // per-group scales from Q4_0
    const float *activation_scales, // per-tensor scales
    float *combined_scales,         // output scale matrix
    const matrix_dims *dims
);
```

### Phase 4: NPU Element-wise 연산 (Week 6)
```c
// ARM의 vmlaq_n_f32 패턴을 NPU element-wise ops로 구현
bool hexagon_npu_elementwise_scale_and_accumulate(
    const int32_t *int_results,    // from matmul
    const float *scale_matrix,     // from CPU
    float *final_output,           // accumulated results
    const hexagon_elementwise_config *config
);
```

### Phase 5: 메인 통합 함수 (Week 7-8)
```c
// ARM ggml_compute_forward_mul_mat 패턴 참고
bool ggml_hexagon_q4_0_matmul_hybrid(
    const block_q4_0 *weights,
    const float *activations, 
    float *output,
    const hexagon_matmul_params *params
) {
    // 1. Group 분할 및 tile 생성
    // 2. NPU int8 matmul (group-wise)
    // 3. CPU scale matrix 생성
    // 4. NPU element-wise scale + accumulation
    // 5. 결과 반환
}
```

## 📈 성능 예상 및 벤치마크 계획

### 성능 목표
- **기존 CPU Q4_0 대비**: 3-5x speedup (NPU int8 가속)
- **메모리 대역폭**: 30-50% 절약 (tiling 최적화)
- **정확도**: Float32 기준 99.9% 이상 유지

### 벤치마크 케이스
- **Small**: 1024×1024, 2048×2048 matrices
- **Medium**: 4096×4096, 8192×4096 matrices  
- **Large**: 16384×8192+ matrices (LLM workloads)

### 비교 대상
- ARM NEON Q4_0 구현 (baseline)
- Hexagon Vector Q4_0 구현
- NPU Int8 hybrid 구현 (목표)

---

**다음 단계**: Phase 1 구현부터 시작하여 ARM CPU 구현의 인사이트를 실제 NPU 코드로 변환
