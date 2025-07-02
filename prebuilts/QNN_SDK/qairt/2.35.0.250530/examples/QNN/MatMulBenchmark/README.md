# Enhanced MatMulBenchmark

QNN SDK를 사용한 NPU 행렬곱셈 성능 측정 도구입니다. 개별 입력 텐서의 데이터 타입 설정과 혼합 정밀도 연산을 지원합니다.

## 주요 개선사항

### ✨ 새로운 기능
- **개별 입력 텐서 데이터 타입 설정**: `-t0`, `-t1` 옵션으로 각 입력 텐서의 데이터 타입을 독립적으로 설정
- **확장된 데이터 타입 지원**: fp16, fp32, int8, int16, uint8, uint16, int32, uint32
- **혼합 정밀도 연산**: 서로 다른 데이터 타입 조합으로 성능 측정
- **자동 출력 타입 결정**: 입력 타입에 따라 최적의 출력 타입 자동 선택
- **향상된 에러 메시지**: 지원되는 데이터 타입 목록 표시
- **메모리 대역폭 측정**: 실제 처리량 분석

## 빌드 방법

```bash
# Android 디바이스용 빌드
./build_matmul_htp_v75.sh <ANDROID_NDK_ROOT> <QNN_SDK_ROOT>

# 예시
./build_matmul_htp_v75.sh $PWD/prebuilts/android-ndk-r28 \
                          $PWD/prebuilts/QNN_SDK/qairt/2.35.0.250530
```

## 사용법

### 기본 문법
```bash
./MatMulBenchmark [options]
```

### 주요 옵션

#### 행렬 크기 설정
- `-m <M>`: 행렬 A의 행 수 (기본값: 128)
- `-k <K>`: 행렬 A의 열 수 / 행렬 B의 행 수 (기본값: 4096)  
- `-n <N>`: 행렬 B의 열 수 (기본값: 4096)

**계산식**: C[M,N] = A[M,K] × B[K,N]

#### 데이터 타입 설정
- `-t0 <type>`: 입력 텐서 A (in[0])의 데이터 타입
- `-t1 <type>`: 입력 텐서 B (in[1])의 데이터 타입  
- `-t <type>`: 두 입력 텐서 모두 같은 타입으로 설정 (호환성 옵션)
- `-to <type>`: 출력 텐서 데이터 타입 (선택사항)

**지원되는 데이터 타입**: `fp16`, `fp32`, `int16`, `int8`, `uint8`, `uint16`, `int32`, `uint32`

#### 실행 설정
- `-i <iter>`: 반복 횟수 (기본값: 10)
- `-lib <path>`: 백엔드 라이브러리 경로

## 사용 예시

### 1. 기본 실행 (fp16 × fp16)
```bash
./MatMulBenchmark -m 1024 -k 4096 -n 4096 -t fp16 -i 10
```

### 2. 혼합 정밀도 연산 (fp16 × int8)
```bash
./MatMulBenchmark -m 512 -k 2048 -n 2048 -t0 fp16 -t1 int8 -i 20
```

### 3. 고정밀도 연산 (fp32 × fp32)
```bash
./MatMulBenchmark -m 256 -k 1024 -n 1024 -t0 fp32 -t1 fp32 -i 5
```

### 4. 정수 연산 (int8 × int8)
```bash
./MatMulBenchmark -m 1024 -k 1024 -n 1024 -t0 int8 -t1 int8 -i 50
```

### 5. 다양한 크기 테스트 (uint8 × int16)
```bash
./MatMulBenchmark -m 2048 -k 8192 -n 1024 -t0 uint8 -t1 int16 -i 10
```

## 출력 예시

```
Running Enhanced MatMul benchmark:
  Matrix sizes: A[1024,4096] × B[4096,4096] = C[1024,4096]
  Input A (in[0]) dtype: fp16 (2 bytes)
  Input B (in[1]) dtype: int8 (1 byte)
  Output C dtype: fp16 (2 bytes)
  Iterations: 10
  Total elements: 25165824

Results:
  Average time: 15.234 ms
  Throughput: 2186.432 GFLOPS
  Memory bandwidth: 45.678 GB/s
```

## 성능 분석 지침

### 1. 데이터 타입별 특성
- **fp32**: 최고 정밀도, 높은 메모리 사용량
- **fp16**: 균형잡힌 정밀도와 성능  
- **int8**: 최고 성능, 낮은 정밀도
- **uint8**: 활성화 함수에 최적화

### 2. 혼합 정밀도 권장사항
- **fp16 × int8**: 가중치 양자화 시나리오
- **fp32 × fp16**: 고정밀도 연산 필요시
- **int8 × int8**: 최대 처리량 달성

### 3. 행렬 크기 최적화
- **작은 행렬 (< 512)**: 지연시간 중심
- **중간 행렬 (512-2048)**: 균형 잡힌 성능
- **큰 행렬 (> 2048)**: 처리량 중심

## 문제 해결

### 일반적인 오류

1. **데이터 타입 오류**
   ```
   ERROR: Unsupported dtype xyz. Supported types: fp16, fp32, int16, int8, uint8, uint16, int32, uint32
   ```
   → 지원되는 데이터 타입을 사용하세요.

2. **그래프 종료 오류**
   ```
   QNN ERROR at MatMulBenchmark.cpp:XXX code=XXXX
   ```
   → 해당 SoC/펌웨어 버전에서 선택한 데이터 타입이 지원되지 않을 수 있습니다.

3. **행렬 차원 오류**
   ```
   ERROR: Matrix dimensions must be positive
   ```
   → 모든 행렬 차원은 양수여야 합니다.

### 디버깅 팁
- `-h` 옵션으로 전체 도움말 확인
- 작은 행렬 크기로 먼저 테스트
- 지원되는 데이터 타입 조합 확인

## 라이선스

Qualcomm Technologies의 QNN SDK 예제 코드 기반으로 개발되었습니다. 