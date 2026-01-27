echo "=========================================================="
echo "启动时间: $(date)"
echo "embedding 测试脚本"
echo "=========================================================="

# --- 1. 参数校验 ---
if [ "$#" -lt 1 ]; then
    echo "错误: 未提供模型路径"
    echo "用法: $0 <path_to_your_model> <the least gpu required> <gpu memory utilization>"
    echo "示例: bash test_embedding.sh /mxstorage/pde_ai/models/llm/BAAI/bge-large-zh/"
    exit 1
fi

MODEL_PATH="$1"
GPU_MEMORY_UTILIZATION="$2"
HOST="localhost"
PORT="9003"
WAIT_TIME=12000
ERROR_THRESHOLD=0.0001

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 提供的模型路径 '$MODEL_PATH' 不存在或不是目录。"
    exit 1
fi

echo "=========================================================="
echo "测试模型: $MODEL_PATH"
echo "API 服务器: http://${HOST}:${PORT}"
echo "=========================================================="

# --- 2. a100 词向量 前五维 ---
# TODO:后续增加新embedding模型 需要在a100测好 加到这里
# 当前数据由a100 vllm 0.11.0/0.11.2得出
EXPECTED_ANSWERS=()
EXPECTED_ANSWERS_BGE_LARGE_ZH=("0.01192" "-0.036339" "-0.03152" "0.00290" "-0.01120")
EXPECTED_ANSWERS_BGE_SMALL_ZH_V1_5=("0.034775" "0.024889" "0.057533" "0.019016" "0.013210")
case $(basename ${MODEL_PATH}) in
    "bge-large-zh")
        EXPECTED_ANSWERS=("${EXPECTED_ANSWERS_BGE_LARGE_ZH[@]}")
        ;;
    "bge-small-zh-v1.5")
        EXPECTED_ANSWERS=("${EXPECTED_ANSWERS_BGE_SMALL_ZH_V1_5[@]}")
        ;;
    *)
        echo "error: not support $(basename ${MODEL_PATH})"
        exit 1
        ;;
esac

compare_with_tolerance() {
    local actual=$1
    local expected=$2
    local threshold=$3
    
    local diff=$(echo "scale=10; $actual - $expected" | bc)

    diff=$(echo "scale=10; if ($diff < 0) -($diff) else $diff" | bc)
    if (( $(echo "$diff < $threshold" | bc -l) )); then
        echo "1"
    else
        echo "0"
    fi
}

# --- 3. 运行测试用例 ---
run_test_case() {

    echo "----------------------------------------------------------"

    vllm serve "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.85} \
        --trust-remote-code \
        --tensor-parallel-size 1  \
        &

    SERVER_PID=$!
    echo "vLLM 启动进程 PID: $SERVER_PID"
    echo "等待模型加载中（最长 ${WAIT_TIME}s）..."

    SECONDS_WAITED=0
    INTERVAL=20
    MODEL_READY=false
    while [ $SECONDS_WAITED -lt $WAIT_TIME ]; do
        STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://${HOST}:${PORT}/v1/models)
        if [ "$STATUS_CODE" -eq 200 ]; then
            MODEL_READY=true
            echo " 模型已就绪（等待 ${SECONDS_WAITED}s）"
            break
        fi
        sleep $INTERVAL
        SECONDS_WAITED=$((SECONDS_WAITED + INTERVAL))
        echo "服务尚未就绪 (${SECONDS_WAITED}s)..."
    done

    if [ "$MODEL_READY" = false ]; then
        echo " 错误: vLLM 启动超时"
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null
        return
    fi
    echo "----------------------------------------------------------"
    echo "发送请求中..."
    response=$(curl -s POST http://${HOST}:${PORT}/v1/embeddings \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_PATH"'",
            "input": "请介绍下你自己"
        }')
    echo "actural embedding:"
    echo "$response" | jq -r '.data[0].embedding[:5] | join(",")'
    extracted=$(echo "$response" | jq -r '.data[0].embedding[:5] | join(",")')

    # 转成数组
    IFS=',' read -ra ACTUAL_EMBEDDINGS <<< "$extracted"
    echo "误差阈值: $ERROR_THRESHOLD"
    echo ""

    extracted_count=${#ACTUAL_EMBEDDINGS[@]}
    # 逐个比较
    all_pass=true
    for i in "${!EXPECTED_ANSWERS[@]}"; do
        if [ $i -ge $extracted_count ]; then
            echo "   [index $i]: 错误 - 提取的值不足"
            all_pass=false
            continue
        fi

        actual_value="${ACTUAL_EMBEDDINGS[$i]}"
        expected_value="${EXPECTED_ANSWERS[$i]}"
        
        # 调用比较函数
        result=$(compare_with_tolerance "$actual_value" "$expected_value" "$ERROR_THRESHOLD")
        
        if [ "$result" -eq "1" ]; then
            echo "   [index $i]: ✓ 通过"
        else
            echo "   [index $i]: ✗ 失败"
            echo "       actual: $actual_value"
            diff=$(echo "scale=6; $actual_value - $expected_value" | bc)
            abs_diff=$(echo "scale=6; if ($diff < 0) -($diff) else $diff" | bc)
            echo "       误差: $abs_diff"
            all_pass=false
        fi
    done

    # --- 关闭服务 ---
    echo "----------------------------------------------------------"
    echo "关闭 vLLM 服务 (PID: $SERVER_PID)..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    sleep 10

    if [ "$all_pass" = true ]; then
        echo "所有embedding值都在误差范围内"
        return 1
    else
        echo "部分embedding值超出误差范围"
        return 0
    fi
}

# --- 4. 运行所有测试 ---
run_test_case 
result=$?
# --- 5. 最终汇总 ---
echo "=========================================================="
case $result in
    0) echo "$(basename ${MODEL_PATH}) 精度异常";;
    1) echo "$(basename ${MODEL_PATH}) 精度正常";;
esac
echo "$(basename ${MODEL_PATH}) 正确率：$((result*100))%"
echo "=========================================================="