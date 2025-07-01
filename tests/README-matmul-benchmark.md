# Matrix Multiplication Benchmark

이 벤치마크는 GGML을 사용하여 [m × k] × [k × n] = [m × n] 형태의 실제 행렬곱 성능을 측정합니다.

## 컴파일

```bash
cd build
make matmul-benchmark
```

## 사용법

```bash
./matmul-benchmark [옵션]
```

### ⚠️ 중요: GGML 행렬곱 제약사항

**GGML에서 `ggml_mul_mat(A, B)`는 다음 조건을 만족해야 합니다:**
- A와 B의 첫 번째 차원(열)이 같아야 함: `A.ne[0] == B.ne[0]`
- 실제 연산: `A[k,m] × B[k,n] = C[m,n]` (k는 공통 차원)

### 기본 파라미터

- `-m <rows>`: 행렬 A의 행 수 (기본값: 1024)
- `-k <common>`: 공통 차원 - A의 열 수 = B의 행 수 (기본값: 1024)  
- `-n <cols>`: 행렬 B의 열 수 (기본값: 1024)

**텐서 생성:**
- A: `ggml_new_tensor_2d(ctx, type, k, m)` → A[k,m] (GGML 형식)
- B: `ggml_new_tensor_2d(ctx, type, k, n)` → B[k,n] (GGML 형식)
- C: 결과는 C[m,n] 형태가 됩니다

**⚠️ 중요한 제약사항:**

1. **QNN 백엔드 제한사항:**
   - 행렬 차원 중 하나라도 `1`이 포함되면 QNN 백엔드에서 rank mismatch 오류 발생
   - 예: `A[1×4096] × B[4096×1]` 같은 경우 자동으로 CPU 백엔드로 전환됨
   - 배치 크기가 1인 경우 (시퀀스 길이 1) CPU 백엔드 사용 권장

2. **메모리 요구사항:**
   - 최소 256KB의 컨텍스트 메모리가 필요합니다
   - 작은 행렬도 GGML 메타데이터와 계산 그래프를 위한 추가 공간이 필요함

### 백엔드 설정 (GGML_USE_HEXAGON이 활성화된 경우)

- `-b <backend>`: 백엔드 타입
  - 0: QNN_CPU
  - 1: QNN_GPU  
  - 2: QNN_NPU (기본값)
  - 3: Hexagon-cDSP
  - 4: ggml (CPU)
- `-a <algo>`: 행렬곱 알고리즘 타입 (기본값: 0)

### 기타 옵션

- `-v`: 자세한 출력 (작은 행렬의 경우 내용 출력)
- `-h` 또는 `?`: 도움말 출력

## 예제 실행

### 기본 실행 (1024×1024 × 1024×1024)
```bash
./matmul-benchmark
```

### 사용자 정의 크기 (512×256 × 256×128)
```bash
./matmul-benchmark -m 512 -k 256 -n 128
```

### CPU 백엔드 사용
```bash
./matmul-benchmark -b 4 -m 1024 -k 1024 -n 1024
```

### 작은 행렬로 결과 확인
```bash
./matmul-benchmark -m 4 -k 4 -n 4 -v
```

### 1이 포함된 차원 (자동으로 CPU 백엔드로 전환됨)
```bash
# 시퀀스 길이 1 (자동 CPU 백엔드)
./matmul-benchmark -m 1 -k 4096 -n 11008 -b 0

# 배치 크기 1 (자동 CPU 백엔드)  
./matmul-benchmark -m 512 -k 1 -n 2048 -b 1
```

## 출력 예제

```
Matrix multiplication: A[1024 x 1024] × B[1024 x 1024] = C[1024 x 1024]
Backend type: 2, Algorithm type: 0
Allocating memory: 12582912 bytes (12.0 MB)
Created tensors:
  A: [1024 x 1024]
  B: [1024 x 1024]
  C: [1024 x 1024]
Using hexagon backend: QNN_NPU
Initializing matrices...
Starting matrix multiplication...
Computation completed!

Results:
  Duration: 45 ms
  GFLOPS: 47.24

Summary: A[1024x1024] × B[1024x1024] = C[1024x1024] in 45 ms (47.24 GFLOPS)
```

## 주요 특징

1. **실제 행렬곱**: 기존 ggmlhexagon-benchmark와 달리 [m×k] × [k×n] 형태의 실제 행렬곱을 수행
2. **유연한 크기**: m, k, n을 독립적으로 설정 가능
3. **성능 측정**: GFLOPS 계산을 통한 성능 평가
4. **백엔드 지원**: CPU, QNN, Hexagon 등 다양한 백엔드 지원
5. **검증 모드**: 작은 행렬에서 결과 확인 가능

## 벤치마크 시나리오

### 정사각 행렬
```bash
./matmul-benchmark -m 1024 -k 1024 -n 1024
```

### 직사각 행렬 (배치 처리 시뮬레이션)
```bash
./matmul-benchmark -m 32 -k 4096 -n 4096    # 작은 배치, 큰 임베딩
./matmul-benchmark -m 1024 -k 512 -n 2048   # 중간 배치, 다양한 크기
```

### 다양한 백엔드 비교
```bash
# CPU 백엔드
./matmul-benchmark -b 4 -m 1024 -k 1024 -n 1024

# QNN NPU 백엔드  
./matmul-benchmark -b 2 -m 1024 -k 1024 -n 1024

# Hexagon cDSP 백엔드
./matmul-benchmark -b 3 -m 1024 -k 1024 -n 1024
``` 