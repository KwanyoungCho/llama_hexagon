#!/system/bin/sh

# Memory monitoring script for Android
# Usage: ./monitor_memory.sh <output_log_file> <inference_command>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <output_log_file> <inference_command...>"
    echo "Example: $0 memory.log ./llama-cli -m model.gguf -ngl 99 -t 8 -n 256"
    exit 1
fi

OUTPUT_LOG="$1"
shift  # Remove first argument, rest are inference command

echo "Starting memory monitoring..."
echo "Output log: $OUTPUT_LOG"
echo "Command: $@"

# Initialize log file with header
printf "timestamp,total_memory_kb,free_memory_kb,available_memory_kb,buffers_kb,cached_kb,swap_total_kb,swap_free_kb,proc_rss_kb,proc_vss_kb\n" > "$OUTPUT_LOG"

# Function to get system memory info
get_memory_info() {
    local timestamp=$(date +%s.%3N)
    
    # Get system memory from /proc/meminfo and clean up whitespace
    local total_mem=$(grep "MemTotal:" /proc/meminfo | awk '{print $2}' | tr -d '\r\n')
    local free_mem=$(grep "MemFree:" /proc/meminfo | awk '{print $2}' | tr -d '\r\n')
    local available_mem=$(grep "MemAvailable:" /proc/meminfo | awk '{print $2}' | tr -d '\r\n')
    local buffers=$(grep "Buffers:" /proc/meminfo | awk '{print $2}' | tr -d '\r\n')
    local cached=$(grep "Cached:" /proc/meminfo | awk '{print $2}' | tr -d '\r\n')
    local swap_total=$(grep "SwapTotal:" /proc/meminfo | awk '{print $2}' | tr -d '\r\n')
    local swap_free=$(grep "SwapFree:" /proc/meminfo | awk '{print $2}' | tr -d '\r\n')
    
    # Default values if not found
    total_mem=${total_mem:-0}
    free_mem=${free_mem:-0}
    available_mem=${available_mem:-0}
    buffers=${buffers:-0}
    cached=${cached:-0}
    swap_total=${swap_total:-0}
    swap_free=${swap_free:-0}
    
    # Get process memory info if process is running
    local proc_rss=0
    local proc_vss=0
    
    if [ ! -z "$INFERENCE_PID" ] && [ -d "/proc/$INFERENCE_PID" ]; then
        # Get RSS and VSS from /proc/[pid]/status and clean up whitespace
        proc_rss=$(grep "VmRSS:" /proc/$INFERENCE_PID/status 2>/dev/null | awk '{print $2}' | tr -d '\r\n')
        proc_vss=$(grep "VmSize:" /proc/$INFERENCE_PID/status 2>/dev/null | awk '{print $2}' | tr -d '\r\n')
        proc_rss=${proc_rss:-0}
        proc_vss=${proc_vss:-0}
    fi
    
    # Use echo -n to avoid newline issues and manually add newline
    echo -n "$timestamp,$total_mem,$free_mem,$available_mem,$buffers,$cached,$swap_total,$swap_free,$proc_rss,$proc_vss" >> "$OUTPUT_LOG"
    echo "" >> "$OUTPUT_LOG"
}

# Function to monitor memory in background
monitor_memory() {
    while true; do
        get_memory_info
        sleep 0.1  # Monitor every 100ms
    done
}

# Start memory monitoring in background
monitor_memory &
MONITOR_PID=$!

echo "Memory monitoring started (PID: $MONITOR_PID)"

# Run the inference command and capture its PID
echo "Starting inference command: $@"
"$@" &
INFERENCE_PID=$!

echo "Inference started (PID: $INFERENCE_PID)"

# Wait for inference to complete
wait $INFERENCE_PID
INFERENCE_EXIT_CODE=$?

echo "Inference completed with exit code: $INFERENCE_EXIT_CODE"

# Stop memory monitoring
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo "Memory monitoring stopped"
echo "Log saved to: $OUTPUT_LOG"

# Show basic statistics
if [ -f "$OUTPUT_LOG" ]; then
    local line_count=$(wc -l < "$OUTPUT_LOG")
    echo "Total data points collected: $((line_count - 1))"
    
    # Show peak memory usage
    echo "Peak process RSS memory usage:"
    tail -n +2 "$OUTPUT_LOG" | cut -d',' -f9 | sort -n | tail -1 | awk '{printf "%.2f MB\n", $1/1024}'
    
    echo "Peak process VSS memory usage:"
    tail -n +2 "$OUTPUT_LOG" | cut -d',' -f10 | sort -n | tail -1 | awk '{printf "%.2f MB\n", $1/1024}'
fi

exit $INFERENCE_EXIT_CODE 