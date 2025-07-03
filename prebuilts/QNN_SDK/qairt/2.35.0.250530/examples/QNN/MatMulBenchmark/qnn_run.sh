#!/bin/bash

# Create results directory
mkdir -p ./matmul_results

echo "Starting comprehensive NPU MatMul benchmark..."
echo "Testing all supported data type combinations from QNN Configuration table"

# FP16 combinations
echo "=== FP16 Tests ==="
adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t fp16 -qa none -qb none -qc none -i 30 -maxseq 1024" > ./matmul_results/fp16_fp16.txt

# FP32 combinations  
echo "=== FP32 Tests ==="
adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t fp32 -qa none -qb none -qc none -i 30 -maxseq 1024" > ./matmul_results/fp32_fp32.txt

# INT16 combinations
echo "=== INT16 Tests ==="
adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t int16 -qa none -qb none -qc none -i 30 -maxseq 1024" > ./matmul_results/int16_int16.txt

# INT8 combinations
echo "=== INT8 Tests ==="
adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t int8 -qa none -qb none -qc none  -i 30 -maxseq 1024" > ./matmul_results/int8_int8.txt

echo "=== Benchmark Complete ==="
echo "Results saved in ./matmul_results/ directory:"
echo "- FP16 combinations: fp16_fp16.txt"
echo "- FP32 combinations: fp32_fp32.txt" 
echo "- INT16 combinations: int16_int16.txt"
echo "- INT8 combinations: int8_int8.txt"
echo ""
echo "Total: 7 unique data type combinations tested"
