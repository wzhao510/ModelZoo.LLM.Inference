#/bin/bash
set -e 

# export CUDA_VISIBLE_DEVICES=0,1

### model_path  tp_size   extra config
MODELS=$(cat <<EOF
/mxstorage/pde_ai/models/llm/Qwen/Qwen3-14B 2 /pde_ai/models/llm/Qwen/Qwen3-14B_eagle3/
EOF
)


log_path="logs/daily_test"
mkdir -p ${log_path}
rm -rf ${log_path}/*

while read -r model tp draft command_rest; do
    echo "==================================================="

    log=$(basename ${model}.log)
    draftName=$(basename ${draft})
    echo "model: ${model}, tp: ${tp}, draft: ${draftName}, custom cmd: ${command_rest} ---> ${log}"
    echo "开始测试"

    # MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE=1 MACA_VLLM_USE_TN_2_NN=0 python offline.py \
    #     --model_path $model \
    #     --tensor_parallel_size $tp \
    #     --data_parallel_size 1 \
    #     --distributed_executor_backend ray ${command_rest}   > ${log_path}/${log} 2>&1

    # echo "************************************************"

    python offline_mtp.py \
        --model_path $model \
        --tensor_parallel_size $tp \
        --distributed_executor_backend ray \
        --draft_model_path $draft ${command_rest} > ${log_path}/${log} 2>&1 #
    exit_code=$?
    if [ $exit_code -eq 1 ]; then
        echo "run_ngrams 执行异常"
    elif [ $exit_code -eq 2 ]; then
        echo "run_eagle3 执行异常"
    elif [ $exit_code -eq 3 ]; then
        echo "run_draft 执行异常"
    elif [ $exit_code -eq 0 ]; then
        echo "程序正常测试完成"
    else
        echo "未知退出码: $exit_code"
    fi
    # 实时监听并提取日志中的测试结果
    sed -n '/======== ngrams 测试结果 ========/,/=================================/p' ${log_path}/${log}
    sed -n '/======== eagle3 测试结果 ========/,/=================================/p' ${log_path}/${log}
    sed -n '/======== draft 测试结果 ========/,/=================================/p' ${log_path}/${log}
done <<< "$MODELS"

echo "=========================================================="
echo "所有测试已成功完成！"
echo "=========================================================="