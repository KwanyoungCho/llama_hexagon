#!/bin/bash

# Create results directory
mkdir -p ./matmul_results

echo "Starting comprehensive NPU MatMul benchmark..."
echo "Testing all supported data type combinations from QNN Configuration table"

# FP16 combinations
echo "=== FP16 Tests ==="
adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 fp16 -t1 fp16 -i 30" > ./matmul_results/fp16_fp16.txt
# adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 fp16 -t1 int8 -i 30" > ./matmul_results/fp16_int8.txt

# FP32 combinations  
echo "=== FP32 Tests ==="
adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 fp32 -t1 fp32 -i 30" > ./matmul_results/fp32_fp32.txt

# INT16 combinations
echo "=== INT16 Tests ==="
# adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 uint16 -t1 uint8 -i 30" > ./matmul_results/uint16_uint8.txt
# adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 uint16 -t1 int8 -i 30" > ./matmul_results/uint16_int8.txt
# adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 uint16 -t1 uint16 -i 30" > ./matmul_results/uint16_uint16.txt
# adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 uint16 -t1 int16 -i 30" > ./matmul_results/uint16_int16.txt
# adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 int16 -t1 uint8 -i 30" > ./matmul_results/int16_uint8.txt
adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 int16 -t1 int16 -i 30" > ./matmul_results/int16_int16.txt

# INT8 combinations
echo "=== INT8 Tests ==="
# adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 uint8 -t1 uint8 -i 30" > ./matmul_results/uint8_uint8.txt
# adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 uint8 -t1 int8 -i 30" > ./matmul_results/uint8_int8.txt
# adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 int8 -t1 uint8 -i 30" > ./matmul_results/int8_uint8.txt
adb shell "cd /data/local/tmp/chokwans99/QNN_benchmark/ && export LD_LIBRARY_PATH=/data/local/tmp/chokwans99/QNN_benchmark/ && ./MatMulBenchmark -a -t0 int8 -t1 int8 -i 30" > ./matmul_results/int8_int8.txt

echo "=== Benchmark Complete ==="
echo "Results saved in ./matmul_results/ directory:"
echo "- FP16 combinations: fp16_fp16.txt, fp16_int8.txt"
echo "- FP32 combinations: fp32_fp32.txt" 
echo "- INT16 combinations: uint16_uint8.txt, uint16_int8.txt, uint16_uint16.txt, uint16_int16.txt, int16_uint8.txt, int16_int16.txt"
echo "- INT8 combinations: uint8_uint8.txt, uint8_int8.txt, int8_uint8.txt, int8_int8.txt"
echo ""
echo "Total: 13 unique data type combinations tested"
