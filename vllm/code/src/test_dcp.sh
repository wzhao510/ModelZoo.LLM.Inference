echo "=========================================================="
echo "启动时间: $(date)"
echo "DCP+tp 测试脚本"
echo "=========================================================="

# --- 1. 参数校验 ---
if [ "$#" -lt 2 ]; then
    echo "错误: 参数错误"
    echo "用法: $0 <path_to_your_model> <the least gpu required> <gpu memory utilization>"
    echo "示例: bash test_dcp.sh /mxstorage/pde_ai/models/llm/DeepSeek/DeepSeek-V2-Lite/ 4 0.85"
    exit 1
fi

MODEL_PATH="$1"
GPUS="$2"
GPU_MEMORY_UTILIZATION="$3"

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

# --- 2. 固定问答对 ---
PROMPTS=(
    "全世界面积最大的国家是哪一个?请回答国家名"
    "144的平方根是多少?请回答数字"
    "11月有多少天？请回答数字"
    "猫的英文是什么？"
    "5的阶乘是多少?请回答数字"
)
EXPECTED_ANSWERS=(
    "俄罗斯"
    "12"
    "30"
    "cat"
    "120"
)

# --- 汇总结果数组 ---
declare -a TEST_NAMES
declare -a TEST_CORRECT
declare -a TEST_TOTAL

# --- 3. 运行单个测试用例 ---
run_test_case() {
    TP_SIZE=$1
    DCP_SIZE=$2
    DESCRIPTION=$3
    DEVICES=$(seq 0 $(($TP_SIZE - 1)) | paste -sd ",")

    echo -e "\n>>> [测试开始] ${DESCRIPTION}"
    echo "----------------------------------------------------------"
    echo "使用 GPU: ${DEVICES} (TP=${TP_SIZE}, DCP=${DCP_SIZE})"

    CUDA_VISIBLE_DEVICES=${DEVICES} vllm serve "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        -tp "$TP_SIZE" \
        -dcp "$DCP_SIZE" \
        --max-model-len 4096 \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.85}  \
        --trust-remote-code \
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
        TEST_NAMES+=("$DESCRIPTION")
        TEST_CORRECT+=(0)
        TEST_TOTAL+=(${#PROMPTS[@]})
        return
    fi

    # --- 执行问答 ---
    CORRECT_COUNT=0
    TOTAL_PROMPTS=${#PROMPTS[@]}

    for ((i=0; i<$TOTAL_PROMPTS; i++)); do
        PROMPT="${PROMPTS[$i]}"
        EXPECTED="${EXPECTED_ANSWERS[$i]}"

        echo "----------------------------------------------------------"
        echo "Prompt: $PROMPT"
        echo "期望关键词: $EXPECTED"

        RESPONSE=$(curl -s http://${HOST}:${PORT}/v1/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "'"$MODEL_PATH"'",
                "prompt": "'"$PROMPT"'",
                "max_tokens": 100,
                "temperature": 0.95
            }' | jq -r '.choices[0].text' 2>/dev/null)

        if [[ -z "$RESPONSE" || "$RESPONSE" == "null" ]]; then
            echo " 无效响应"
        else
            echo "模型回复: $RESPONSE"
            if echo "$RESPONSE" | grep -qi "$EXPECTED"; then
                echo " 匹配成功"
                ((CORRECT_COUNT++))
            else
                echo " 匹配失败"
            fi
        fi
    done

    # --- 关闭服务 ---
    echo "----------------------------------------------------------"
    echo "关闭 vLLM 服务 (PID: $SERVER_PID)..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    sleep 10

    # --- 保存结果 ---
    TEST_NAMES+=("$DESCRIPTION")
    TEST_CORRECT+=($CORRECT_COUNT)
    TEST_TOTAL+=($TOTAL_PROMPTS)

    echo " 本轮测试完成 (${CORRECT_COUNT}/${TOTAL_PROMPTS})"
}

# --- 4. 运行所有测试 ---
if [ "$GPUS" -eq 2 ]; then
    run_test_case 2 2 "TP=2, DCP=2 (2 GPU)"
elif [ "$GPUS" -eq 4 ]; then
    run_test_case 4 2 "TP=4, DCP=2 (4 GPU)"
    run_test_case 4 4 "TP=4, DCP=4 (4 GPU)"
fi
# --- 5. 最终汇总 ---
echo
echo "=========================================================="
echo " 最终测试汇总"
echo "=========================================================="
printf "%-25s | %-10s | %-10s | %-10s\n" "配置描述" "正确数" "总题数" "正确率"
echo "----------------------------------------------------------"
COUNT=0
TOTAL_PERCENT=0
for ((i=0; i<${#TEST_NAMES[@]}; i++)); do
    NAME="${TEST_NAMES[$i]}"
    CORRECT=${TEST_CORRECT[$i]}
    TOTAL=${TEST_TOTAL[$i]}
    PERCENT=$((100 * CORRECT / TOTAL))
    COUNT=$((COUNT + 1))
    TOTAL_PERCENT=$((TOTAL_PERCENT + PERCENT))
    printf "%-25s | %-10s | %-10s | %-9s%%\n" "$NAME" "$CORRECT" "$TOTAL" "$PERCENT"
done
RESULT=$((TOTAL_PERCENT/COUNT))

echo "=========================================================="
echo " 测试全部完成！"
echo "$(basename ${MODEL_PATH})正确率：${RESULT}%"
echo "=========================================================="