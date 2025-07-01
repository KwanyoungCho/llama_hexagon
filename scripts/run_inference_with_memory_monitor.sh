#!/bin/bash

# Script to run inference with memory monitoring on Android device via adb
# Usage: ./run_inference_with_memory_monitor.sh [device_serial]

set -e

DEVICE_SERIAL="$1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOCAL_LOG_DIR="./memory_logs"
DEVICE_WORK_DIR="/data/local/tmp/llama_QNN"
DEVICE_LOG_FILE="memory_${TIMESTAMP}.csv"
LOCAL_LOG_FILE="${LOCAL_LOG_DIR}/${DEVICE_LOG_FILE}"

# Create local log directory
mkdir -p "$LOCAL_LOG_DIR"

# ADB command prefix
if [ -n "$DEVICE_SERIAL" ]; then
    ADB_CMD="adb -s $DEVICE_SERIAL"
else
    ADB_CMD="adb"
fi

echo "=== QNN Inference Memory Monitor ==="
echo "Timestamp: $(date)"
echo "Device: $(${ADB_CMD} shell getprop ro.product.model 2>/dev/null || echo 'Unknown')"
echo "Log file: $LOCAL_LOG_FILE"
echo

# Check if device is connected
if ! ${ADB_CMD} shell echo "Device connected" >/dev/null 2>&1; then
    echo "Error: No Android device connected or adb not working"
    echo "Please check:"
    echo "1. Device is connected via USB"
    echo "2. USB debugging is enabled"
    echo "3. adb is in PATH"
    exit 1
fi

# Push monitoring script to device
echo "Pushing monitoring script to device..."
${ADB_CMD} push "$(dirname "$0")/monitor_memory.sh" "$DEVICE_WORK_DIR/"
${ADB_CMD} shell chmod +x "$DEVICE_WORK_DIR/monitor_memory.sh"

# Inference command
#INFERENCE_CMD="cd $DEVICE_WORK_DIR && export LD_LIBRARY_PATH=$DEVICE_WORK_DIR && ./llama-cli -m /sdcard/Qwen2.5-3B-Instruct-Q4_0.gguf -ngl 99 -t 8 -n 256 -mg 2 -no-cnv -p \"every day of your life, it is important to take the time to smell the roses — to appreciate the experiences that lead to happiness. This is part of being truly happy.Happiness is a state of mind. It starts with accepting where you are, knowing where you are going and planning to enjoy every moment along the way. You know how to be happy, and feel that you have enough time or money or love or whatever you need to achieve your goals. And just feeling that you have enough of everything means that you do indeed have enough.You have to choose to be happy, and focus upon being happy, in order to be happy. If you instead focus upon knowing that you will be happy if you achieve something, you will never be happy, as you have not learned to smell the roses. The irony is that when you are happy, you are inevitably more productive, and far more likely to achieve what everything-seekers are seeking. you will never be happy, as you have not learned to smell the roses. The irony is that when you are happy, you are inevitably more productive, and far more likely to achieve what everything-seekers are seeking.\""
INFERENCE_CMD="cd $DEVICE_WORK_DIR && export LD_LIBRARY_PATH=$DEVICE_WORK_DIR && ./llama-cli -m /sdcard/qwen1_5-1_8b-chat-q4_0.gguf -ngl 99 -t 8 -n 256 -mg 2 -no-cnv -p \"every day of your life, it is important to take the time to smell the roses — to appreciate the experiences that lead to happiness. This is part of being truly happy.Happiness is a state of mind. It starts with accepting where you are, knowing where you are going and planning to enjoy every moment along the way. You know how to be happy, and feel that you have enough time or money or love or whatever you need to achieve your goals. And just feeling that you have enough of everything means that you do indeed have enough.You have to choose to be happy, and focus upon being happy, in order to be happy. If you instead focus upon knowing that you will be happy if you achieve something, you will never be happy, as you have not learned to smell the roses. The irony is that when you are happy, you are inevitably more productive, and far more likely to achieve what everything-seekers are seeking. you will never be happy, as you have not learned to smell the roses. The irony is that when you are happy, you are inevitably more productive, and far more likely to achieve what everything-seekers are seeking.\""
echo "Starting inference with memory monitoring..."
echo "Command: $INFERENCE_CMD"
echo

# Run inference with memory monitoring
${ADB_CMD} shell "cd $DEVICE_WORK_DIR && ./monitor_memory.sh $DEVICE_LOG_FILE sh -c '$INFERENCE_CMD'"

INFERENCE_EXIT_CODE=$?

echo
echo "Inference completed with exit code: $INFERENCE_EXIT_CODE"

# Pull log file from device
echo "Pulling log file from device..."
${ADB_CMD} pull "$DEVICE_WORK_DIR/$DEVICE_LOG_FILE" "$LOCAL_LOG_FILE"

# Clean up device
${ADB_CMD} shell rm -f "$DEVICE_WORK_DIR/$DEVICE_LOG_FILE"

echo "Log file saved to: $LOCAL_LOG_FILE"

# Show basic statistics
if [ -f "$LOCAL_LOG_FILE" ]; then
    line_count=$(wc -l < "$LOCAL_LOG_FILE")
    echo "Total data points collected: $((line_count - 1))"
    
    echo
    echo "=== Memory Usage Summary ==="
    
    # Peak memory usage
    peak_rss=$(tail -n +2 "$LOCAL_LOG_FILE" | cut -d',' -f9 | sort -n | tail -1)
    peak_vss=$(tail -n +2 "$LOCAL_LOG_FILE" | cut -d',' -f10 | sort -n | tail -1)
    
    echo "Peak process RSS memory: $(echo "scale=2; $peak_rss/1024" | bc -l) MB"
    echo "Peak process VSS memory: $(echo "scale=2; $peak_vss/1024" | bc -l) MB"
    
    # System memory stats
    min_free=$(tail -n +2 "$LOCAL_LOG_FILE" | cut -d',' -f3 | sort -n | head -1)
    max_free=$(tail -n +2 "$LOCAL_LOG_FILE" | cut -d',' -f3 | sort -n | tail -1)
    
    echo "Minimum free system memory: $(echo "scale=2; $min_free/1024" | bc -l) MB"
    echo "Maximum free system memory: $(echo "scale=2; $max_free/1024" | bc -l) MB"
    
    echo
    echo "Use 'python3 scripts/plot_memory_usage.py $LOCAL_LOG_FILE' to generate graphs"
fi

exit $INFERENCE_EXIT_CODE 