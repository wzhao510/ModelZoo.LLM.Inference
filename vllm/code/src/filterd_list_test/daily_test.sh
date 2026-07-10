#!/bin/bash
ulimit -n 65536
# set -ex

###############################################################################
# Configuration — edit this section per benchmark run
###############################################################################

# Directory to store all logs
LOG_DIR="${LOG_DIR:-./bench_logs/$(date +%Y%m%d_%H%M%S)}"

# Default common parameters for all models
DEFAULT_TP=8
DEFAULT_GPU_MEM_UTIL=0.9
DEFAULT_MAX_MODEL_LEN=32768
DEFAULT_PORT=8000

# Client script path
CLIENT_SCRIPT="${CLIENT_SCRIPT:-./client_random.py}"

###############################################################################
# Model registry
# Two ways to specify models:
#   1. Pass a config file:   bash bench_vllm.sh -c failed_configs.txt
#   2. Edit the MODELS array below (used when no -c flag is given)
# Format: "model_path|tp|gpu_mem|max_len|port|extra_args"
# Leave a field empty (e.g. "model|||32768||--trust-remote-code") to use defaults
###############################################################################
MODELS=()
CONFIG_FILE=""

usage() {
    echo "Usage: $0 [-c CONFIG_FILE]"
    echo "  -c CONFIG_FILE   Read model entries from a config file"
    echo "                   (one pipe-delimited entry per line, skip comments and blanks)"
    echo ""
    echo "  Without -c, MODELS are read from the hardcoded array in this script."
    exit 1
}

# Parse command line
while getopts "c:h" opt; do
    case "$opt" in
        c) CONFIG_FILE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

if [ -n "$CONFIG_FILE" ]; then
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: config file not found: $CONFIG_FILE"
        exit 1
    fi
    while read -r line; do
        test -z "$line" && continue
        case "$line" in
            "#"*) continue ;;
        esac
        MODELS+=("$line")
    done < "$CONFIG_FILE"
    echo "Loaded ${#MODELS[@]} model(s) from $CONFIG_FILE"
else
    # Hardcoded fallback -- edit these for one-off runs
    MODELS+=("/mxstorage/pde_ai/models/llm/Qwen/Qwen3.5-397B-A17B/|8|0.9|32768|8000|--trust-remote-code")
    # MODELS+=("/path/to/another-model|4|0.85|16384|8001|--trust-remote-code --enable-prefix-caching")
    # MODELS+=("/path/to/third-model|8|0.9|65536|8002|--trust-remote-code --enable-chunked-prefill")
fi

###############################################################################
# Helper functions
###############################################################################

env_collect() {
    local env_dir="$1"
    mkdir -p "$env_dir"

    echo "[$(date '+%H:%M:%S')] Collecting environment info into $env_dir ..."

    mx-smi topo  > "$env_dir/mx-smi-topo.log"  2>&1 || true
    mx-smi       > "$env_dir/mx-smi.log"       2>&1 || true
    lscpu            > "$env_dir/lscpu.log"            2>&1 || true
    pip list         > "$env_dir/pip-list.log"         2>&1 || true
    env | sort       > "$env_dir/env.log"              2>&1 || true

    echo "[$(date '+%H:%M:%S')] Environment info collected."
}

kill_gpu_processes() {
    echo "[$(date '+%H:%M:%S')] Killing all GPU processes..."
    pgrep pt_main_thread | xargs -r kill -9
    pgrep python3 | xargs -r kill -9
    pgrep vllm | xargs -r kill -9
    mx-smi --kill-all-process
    sleep 3
    rm -rf ~/.config/vllm
    echo "[$(date '+%H:%M:%S')] GPU processes cleaned up."
}

start_server() {
    local model_path="$1"
    local tp="$2"
    local gpu_mem="$3"
    local max_len="$4"
    local port="$5"
    local extra_args="$6"
    local server_log="$7"

    echo "[$(date '+%H:%M:%S')] Starting vLLM server for: $model_path"
    echo "  tp=$tp  gpu_mem=$gpu_mem  max_len=$max_len  port=$port"
    echo "  extra_args=$extra_args"

    # Build command string, then eval — handles extra_args with JSON/quotes safely
    local run_cmd
    run_cmd="vllm serve \"$model_path\" -tp \"$tp\" --gpu-memory-utilization \"$gpu_mem\" --max-model-len \"$max_len\" --port \"$port\""
    if [ -n "$extra_args" ]; then
        run_cmd="$run_cmd $extra_args"
    fi

    eval "$run_cmd > \"$server_log\" 2>&1 &"

    SERVER_PID=$!
    echo "  Server PID: $SERVER_PID"
}

wait_for_server() {
    local port="$1"
    local server_log="$2"
    local timeout_sec="${3:-360}"
    local start_ts now elapsed

    echo "[$(date '+%H:%M:%S')] Waiting for server on port $port (timeout: ${timeout_sec}s)..."

    start_ts=$(date +%s)

    while true; do
        # 1. Check if server process is still alive
        if [ -n "${SERVER_PID:-}" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] ERROR: Server process $SERVER_PID died unexpectedly."
            echo "=== Last 50 lines of server log ==="
            tail -50 "$server_log"
            return 1
        fi

        # 2. Check server log for Traceback (fatal Python error)
        if grep -q "Traceback" "$server_log" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] ERROR: Traceback found in server log!"
            grep -A 20 "Traceback" "$server_log" | head -40
            return 1
        fi

        # 3. Try the endpoint
        if curl -s "localhost:${port}/v1/completions" > /dev/null 2>&1; then
            elapsed=$(($(date +%s) - start_ts))
            echo "[$(date '+%H:%M:%S')] Server is ready on port $port (took ${elapsed}s)."
            return 0
        fi

        # 4. Check timeout
        now=$(date +%s)
        if [ $(( now - start_ts )) -gt "$timeout_sec" ]; then
            echo "[$(date '+%H:%M:%S')] ERROR: Timeout waiting for server on port $port."
            echo "=== Last 50 lines of server log ==="
            tail -50 "$server_log"
            return 1
        fi

        sleep 2
    done
}

run_client() {
    local model_path="$1"
    local port="$2"
    local client_log="$3"
    local rc

    echo "[$(date '+%H:%M:%S')] Running client benchmark..."
    echo "  model_path=$model_path  port=$port"

    python "$CLIENT_SCRIPT" \
        --model-path "$model_path" \
        --port "$port" \
        > "$client_log" 2>&1

    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] WARNING: Client exited with code $rc"
    else
        echo "[$(date '+%H:%M:%S')] Client benchmark finished."
    fi
    return $rc
}

benchmark_one_model() {
    local entry="$1"
    local idx="$2"

    # Parse pipe-delimited fields (compatible with bash 3.x+)
    local model_path tp gpu_mem max_len port extra_args
    model_path=$( echo "$entry" | cut -d'|' -f1 )
    tp=$(         echo "$entry" | cut -d'|' -f2 )
    gpu_mem=$(    echo "$entry" | cut -d'|' -f3 )
    max_len=$(    echo "$entry" | cut -d'|' -f4 )
    port=$(       echo "$entry" | cut -d'|' -f5 )
    extra_args=$( echo "$entry" | cut -d'|' -f6 )

    # Apply defaults for empty fields
    tp="${tp:-$DEFAULT_TP}"
    gpu_mem="${gpu_mem:-$DEFAULT_GPU_MEM_UTIL}"
    max_len="${max_len:-$DEFAULT_MAX_MODEL_LEN}"
    port="${port:-$DEFAULT_PORT}"

    local model_name run_dir server_log client_log client_rc
    model_name=$(basename "$model_path")

    run_dir="${LOG_DIR}/${idx}_${model_name}"
    mkdir -p "$run_dir"

    server_log="${run_dir}/server.log"
    client_log="${run_dir}/client.log"

    echo ""
    echo "============================================================"
    echo " Benchmark [$idx]: $model_name"
    echo "   Model path : $model_path"
    echo "   TP         : $tp"
    echo "   GPU mem    : $gpu_mem"
    echo "   Max len    : $max_len"
    echo "   Port       : $port"
    echo "   Extra args : ${extra_args:-none}"
    echo "   Log dir    : $run_dir"
    echo "============================================================"

    kill_gpu_processes

    SERVER_PID=""
    start_server "$model_path" "$tp" "$gpu_mem" "$max_len" "$port" "$extra_args" "$server_log"

    if ! wait_for_server "$port" "$server_log" 3600; then
        echo "[$(date '+%H:%M:%S')] FATAL: Server failed to start for $model_name"
        kill_gpu_processes
        return 1
    fi

    run_client "$model_path" "$port" "$client_log"
    client_rc=$?

    # Final health check: curl the port to confirm server is still alive
    echo "[$(date '+%H:%M:%S')] Final health check on port $port ..."
    local server_alive=0
    if curl -s --max-time 10 "localhost:${port}/v1/completions" > /dev/null 2>&1; then
        server_alive=1
        echo "[$(date '+%H:%M:%S')] Server still alive on port $port."
    else
        echo "[$(date '+%H:%M:%S')] Server unreachable on port $port after client run!"
        # Also check server log for fresh Traceback
        if grep -q "Traceback" "$server_log" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] Traceback found in server log after client exit:"
            grep -A 10 "Traceback" "$server_log" | tail -15
        fi
    fi

    kill_gpu_processes

    # Model passes only if: client ran without error AND server stayed alive
    local final_rc=0
    if [ "$client_rc" -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] FAIL: client exited with code $client_rc"
        final_rc=1
    fi
    if [ "$server_alive" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] FAIL: server died during benchmark"
        final_rc=1
    fi

    if [ "$final_rc" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] PASS [$idx]: $model_name (client_rc=$client_rc, server_alive=$server_alive)"
    else
        echo "[$(date '+%H:%M:%S')] FAIL [$idx]: $model_name (client_rc=$client_rc, server_alive=$server_alive)"
    fi
    return $final_rc
}

###############################################################################
# Main
###############################################################################

# Write status files to LOG_DIR after each model completes
write_status_files() {
    local status_file="$LOG_DIR/status.txt"
    local passed_file="$LOG_DIR/passed.txt"
    local failed_file="$LOG_DIR/failed.txt"
    local remaining_file="$LOG_DIR/remaining.txt"

    # passed.txt — full config lines, can be fed back with -c
    printf '%s\n' "${passed_configs[@]}" > "$passed_file"

    # failed.txt — full config lines, for extract_failed.py or rerun
    printf '%s\n' "${failed_configs[@]}" > "$failed_file"

    # remaining.txt — full config lines, not yet run
    printf '%s\n' "${remaining_configs[@]}" > "$remaining_file"

    # status.txt — human-readable summary
    {
        echo "============================================================"
        echo " Benchmark status — $(date)"
        echo " Log directory: $LOG_DIR"
        echo "============================================================"
        echo "  Total:     ${#MODELS[@]}"
        echo "  Passed:    ${#passed_models[@]}"
        echo "  Failed:    ${#failed_models[@]}"
        echo "  Remaining: ${#remaining_configs[@]}"
        echo ""
        echo " Passed:"
        for m in "${passed_models[@]}"; do
            echo "   [P] $m"
        done
        echo ""
        echo " Failed:"
        for m in "${failed_models[@]}"; do
            echo "   [F] $m"
        done
        echo ""
        echo " Remaining:"
        for m in "${remaining_models[@]}"; do
            echo "   [ ] $m"
        done
    } > "$status_file"
}

main() {
    mkdir -p "$LOG_DIR"

    local total i idx model_name
    local passed_models=()
    local failed_models=()
    local passed_configs=()
    local failed_configs=()
    local remaining_models=()
    local remaining_configs=()
    local pending_models=()

    total=${#MODELS[@]}

    # Collect pending names
    for i in "${!MODELS[@]}"; do
        model_name=$(echo "${MODELS[$i]}" | cut -d'|' -f1)
        model_name=$(basename "$model_name")
        pending_models+=("$model_name")
    done
    remaining_models=("${pending_models[@]}")
    remaining_configs=("${MODELS[@]}")

    echo "============================================================"
    echo " Benchmark started at $(date)"
    echo " Log directory: $LOG_DIR"
    echo " Models to benchmark: $total"
    echo "============================================================"
    echo ""
    echo " Pending:"
    for m in "${pending_models[@]}"; do
        echo "   [ ] $m"
    done
    echo ""

    # Collect environment info once before all benchmarks
    env_collect "${LOG_DIR}/env"

    # Write initial status
    write_status_files

    for i in "${!MODELS[@]}"; do
        idx=$((i + 1))
        model_name="${pending_models[$i]}"

        echo ""
        echo ">>> Progress: [$idx/$total] <<<"

        if benchmark_one_model "${MODELS[$i]}" "$idx"; then
            passed_models+=("$model_name")
            passed_configs+=("${MODELS[$i]}")
            echo ">>> [$idx/$total] PASS: $model_name"
        else
            failed_models+=("$model_name")
            failed_configs+=("${MODELS[$i]}")
            echo ">>> [$idx/$total] FAIL: $model_name"
        fi

        # Update remaining lists
        remaining_models=()
        remaining_configs=()
        local j
        for j in "${!MODELS[@]}"; do
            if [ "$j" -gt "$i" ]; then
                remaining_models+=("${pending_models[$j]}")
                remaining_configs+=("${MODELS[$j]}")
            fi
        done

        # Write status files after each model
        write_status_files

        # Print current status
        echo ""
        echo "--- Current status after $idx/$total ---"
        echo "  Passed (${#passed_models[@]}):"
        for m in "${passed_models[@]}"; do
            echo "    [P] $m"
        done
        echo "  Failed (${#failed_models[@]}):"
        for m in "${failed_models[@]}"; do
            echo "    [F] $m"
        done
        echo "  Remaining (${#remaining_models[@]}):"
        for m in "${remaining_models[@]}"; do
            echo "    [ ] $m"
        done
        echo "----------------------------------------"
        echo "  Status files updated: $LOG_DIR/{passed,failed,remaining,status}.txt"
    done

    # Final summary
    echo ""
    echo "============================================================"
    echo " Benchmark finished at $(date)"
    echo " Logs saved to: $LOG_DIR"
    echo "============================================================"
    echo "  Total:   $total"
    echo "  Passed:  ${#passed_models[@]}"
    echo "  Failed:  ${#failed_models[@]}"
    echo ""
    if [ "${#passed_models[@]}" -gt 0 ]; then
        echo "  Passed models:"
        for m in "${passed_models[@]}"; do
            echo "    [P] $m"
        done
    fi
    if [ "${#failed_models[@]}" -gt 0 ]; then
        echo ""
        echo "  Failed models (config saved to $LOG_DIR/failed.txt):"
        for m in "${failed_models[@]}"; do
            echo "    [F] $m"
        done
    fi
    echo "============================================================"

    if [ "${#failed_models[@]}" -gt 0 ]; then
        return 1
    fi
    return 0
}

main "$@"
