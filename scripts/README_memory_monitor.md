# QNN Inference Memory Monitoring Tools

이 도구들은 Android 디바이스에서 QNN inference 실행 중 시스템 메모리 사용량을 모니터링하고 시각화하는 기능을 제공합니다.

## 📁 파일 구성

### 1. 스크립트 파일들
- **`monitor_memory.sh`**: Android 디바이스에서 실행되는 메모리 모니터링 스크립트
- **`run_inference_with_memory_monitor.sh`**: PC에서 실행하는 메인 스크립트 (adb 통해 디바이스 제어)
- **`plot_memory_usage.py`**: 메모리 로그를 분석하여 그래프를 생성하는 Python 스크립트

### 2. 생성되는 파일들
- **`./memory_logs/`**: 메모리 로그 CSV 파일들이 저장되는 디렉토리
- **`./memory_plots/`**: 생성된 그래프들이 저장되는 디렉토리

## 🚀 사용 방법

### Step 1: 메모리 모니터링과 함께 inference 실행

```bash
# 기본 사용법 (첫 번째 연결된 디바이스 사용)
cd /home/chokwans99/tmp/ggml-hexagon
./scripts/run_inference_with_memory_monitor.sh

# 특정 디바이스 시리얼 지정
./scripts/run_inference_with_memory_monitor.sh DEVICE_SERIAL
```

### Step 2: 그래프 생성

```bash
# 기본 사용법
python3 scripts/plot_memory_usage.py ./memory_logs/memory_YYYYMMDD_HHMMSS.csv

# 고급 옵션
python3 scripts/plot_memory_usage.py ./memory_logs/memory_YYYYMMDD_HHMMSS.csv \
    --output-dir ./custom_plots \
    --format pdf \
    --dpi 300 \
    --style seaborn
```

## 📊 생성되는 그래프들

### 1. `system_memory.png`
- **상단 그래프**: 전체 시스템 메모리 사용량 (Total, Used, Free, Available)
- **하단 그래프**: 메모리 세부사항 (Buffers, Cached, Swap)

### 2. `process_memory.png`
- **상단 그래프**: llama-cli 프로세스 메모리 사용량 (RSS, VSS)
- **하단 그래프**: 메모리 효율성 (RSS/VSS 비율)

### 3. `memory_pressure.png`
- 시스템 메모리 압박 상황 분석
- 색상 영역: 녹색(낮음), 노란색(중간), 빨간색(높음)

### 4. `memory_overview.png`
- 4개 패널로 구성된 종합 분석 뷰
- 시스템 메모리, 프로세스 메모리, 메모리 분포, 통계 테이블

## 📈 모니터링 데이터

### CSV 로그 파일 구조
```csv
timestamp,total_memory_kb,free_memory_kb,available_memory_kb,buffers_kb,cached_kb,swap_total_kb,swap_free_kb,proc_rss_kb,proc_vss_kb
1703001234.567,8388608,2097152,3145728,524288,1048576,0,0,1048576,2097152
```

### 수집되는 메트릭들
- **System Memory**: Total, Free, Available, Buffers, Cached, Swap
- **Process Memory**: RSS (물리 메모리), VSS (가상 메모리)
- **Timing**: 0.1초 간격으로 수집

## ⚙️ 설정 옵션

### `run_inference_with_memory_monitor.sh` 수정 가능한 부분

```bash
# inference 명령어 변경
INFERENCE_CMD="cd $DEVICE_WORK_DIR && export LD_LIBRARY_PATH=$DEVICE_WORK_DIR && ./llama-cli -m /sdcard/qwen1_5-1_8b-chat-q4_0.gguf -ngl 99 -t 8 -n 256 -mg 2 -no-cnv -p \"hello\""

# 디바이스 작업 디렉토리 변경  
DEVICE_WORK_DIR="/data/local/tmp/llama_QNN"
```

### `monitor_memory.sh` 수정 가능한 부분

```bash
# 모니터링 간격 변경 (기본: 0.1초)
sleep 0.1  # 더 빠른 샘플링은 0.05, 더 느린 샘플링은 0.5
```

### `plot_memory_usage.py` 옵션들

```bash
# 출력 형식 옵션
--format png|pdf|svg     # 기본: png
--dpi 300                # 기본: 300
--style default|seaborn|ggplot  # 기본: seaborn

# 출력 디렉토리
--output-dir ./custom_plots  # 기본: ./memory_plots
```

## 🔧 요구사항

### PC 환경
- **adb**: Android Debug Bridge 설치 및 PATH 설정
- **Python 3.6+**: pandas, matplotlib, numpy 패키지
- **bc**: 계산기 유틸리티 (bash 스크립트용)

```bash
# Python 패키지 설치
pip install pandas matplotlib numpy

# Ubuntu/Debian에서 bc 설치
sudo apt install bc

# ADB 설치 확인
adb version
```

### Android 디바이스 환경
- **USB 디버깅** 활성화
- **llama-cli 바이너리** 및 **모델 파일** 준비
- **QNN 라이브러리들** `/data/local/tmp/llama_QNN/`에 설치

## 🐛 문제 해결

### 일반적인 문제들

1. **adb 연결 실패**
   ```bash
   adb devices  # 디바이스 목록 확인
   adb kill-server && adb start-server  # adb 재시작
   ```

2. **권한 오류**
   ```bash
   chmod +x scripts/*.sh scripts/*.py
   ```

3. **Python 패키지 누락**
   ```bash
   pip install -r requirements.txt  # 필요시 requirements.txt 생성
   ```

4. **메모리 로그 파일이 비어있음**
   - Android 디바이스의 `/proc/meminfo` 접근 권한 확인
   - inference 명령어가 올바른지 확인

### 디버깅 팁

```bash
# 로그 파일 내용 확인
head -5 ./memory_logs/memory_YYYYMMDD_HHMMSS.csv

# 디바이스에서 수동으로 메모리 정보 확인
adb shell cat /proc/meminfo

# inference 프로세스 실행 확인
adb shell ps | grep llama
```

## 📝 사용 예시

### 완전한 실행 예시

```bash
# 1. 메모리 모니터링과 함께 inference 실행
cd /home/chokwans99/tmp/ggml-hexagon
./scripts/run_inference_with_memory_monitor.sh

# 출력 예시:
# === QNN Inference Memory Monitor ===
# Device: SM8650
# Log file: ./memory_logs/memory_20241215_143022.csv
# Starting inference with memory monitoring...
# Inference completed with exit code: 0
# Peak process RSS memory: 1234.56 MB

# 2. 그래프 생성
python3 scripts/plot_memory_usage.py ./memory_logs/memory_20241215_143022.csv

# 출력 예시:
# Loaded 1250 data points
# === Memory Usage Analysis ===
# Total System Memory: 8192 MB
# Peak Used Memory: 6543 MB (79.9%)
# Peak RSS: 1235 MB
# System memory plot saved: ./memory_plots/system_memory.png
# Process memory plot saved: ./memory_plots/process_memory.png
# Memory pressure plot saved: ./memory_plots/memory_pressure.png
# Overview plot saved: ./memory_plots/memory_overview.png
```

## 📚 추가 정보

이 도구들은 QNN inference의 메모리 사용 패턴을 분석하여 다음과 같은 최적화에 활용할 수 있습니다:

- **메모리 효율성 분석**: RSS/VSS 비율로 메모리 사용 효율성 평가
- **메모리 압박 상황 파악**: 시스템 메모리 부족 상황 감지
- **성능 최적화**: 메모리 사용량에 따른 inference 성능 상관관계 분석
- **리소스 계획**: 필요한 최소 메모리 요구사항 결정 