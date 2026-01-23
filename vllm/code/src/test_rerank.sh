echo "=========================================================="
echo "启动时间: $(date)"
echo "rerank 测试脚本"
echo "=========================================================="

# --- 1. 参数校验 ---
if [ "$#" -lt 1 ]; then
    echo "错误: 未提供模型路径"
    echo "用法: $0 <path_to_your_model> <the least gpu required> <gpu memory utilization>"
    echo "示例: bash test_rerank.sh /mxstorage/pde_ai/models/llm/BAAI/bge-reranker-v2-m3/ 0.85"
    exit 1
fi

MODEL_PATH="$1"
GPU_MEMORY_UTILIZATION="$2"
HOST="localhost"
PORT="8000"
WAIT_TIME=12000

if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 提供的模型路径 '$MODEL_PATH' 不存在或不是目录。"
    exit 1
fi

echo "=========================================================="
echo "测试模型: $MODEL_PATH"
echo "API 服务器: http://${HOST}:${PORT}"
echo "=========================================================="

# --- 2. 期望回答 ---
EXPECTED_ANSWERS=(
    "人工智能正在改变各行各业"
    "机器学习是AI的核心技术"
    "深度学习框架如TensorFlow很流行"
    "AI伦理是重要议题"
    "今天天气很好"
)


# --- 3. 运行单个测试用例 ---
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

    RESPONSE=$(curl -s http://${HOST}:${PORT}/v1/rerank  \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL_PATH}"'",
            "query": "人工智能的未来发展",
            "temperature": 0.7, 
            "top_p": 0.9,
            "documents": [ "人工智能正在改变各行各业", "机器学习是AI的核心技术", "今天天气很好", "深度学习框架如TensorFlow很流行", "AI伦理是重要议题" ],
            "top_k": 3,
            "return_documents": true
        }')

    if [[ -z "$RESPONSE" || "$RESPONSE" == "null" ]]; then
        echo " 无效响应"
    else
        echo "模型回复: $RESPONSE"
    fi

    extracted_texts=()
    while IFS= read -r line; do
        extracted_texts+=("$line")
    done < <(echo "$RESPONSE" | jq -r '.results[].document.text')

    match=true
    for i in "${!EXPECTED_ANSWERS[@]}"; do
        if [ "${extracted_texts[$i]}" != "${EXPECTED_ANSWERS[$i]}" ]; then
            echo "位置 $i 不匹配:"
            echo "  提取结果: ${extracted_texts[$i]}"
            echo "  期望答案: ${EXPECTED_ANSWERS[$i]}"
            match=false
        fi
    done

    # --- 关闭服务 ---
    echo "----------------------------------------------------------"
    echo "关闭 vLLM 服务 (PID: $SERVER_PID)..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    sleep 5
    if [ "$match" = true ]; then
        return 1
    else
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