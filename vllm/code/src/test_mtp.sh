#/bin/bash
set -e 

MODEL_PATH="$1"
GPU_REQUIERED="$2"
DRAFT_PATH="$3"
GPU_MEMORY_UTILIZATION="$4"
MAX_NUM_SEQS="$5"

log_path="$MTP_LOG_PATH"
echo "log路径: $log_path"
mkdir -p ${log_path}
rm -rf ${log_path}/*


echo "==================================================="

model=$(basename ${MODEL_PATH})
draftName=$(basename ${DRAFT_PATH})
echo "model: ${model}, tp: ${GPU_REQUIERED}, draft: ${draftName}  ---> ${model}.log"
echo "开始测试"

python offline_mtp.py \
    --model $MODEL_PATH \
    --tensor-parallel-size $GPU_REQUIERED \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.85} \
    --max-num-seqs ${MAX_NUM_SEQS:-256} \
    --distributed-executor-backend ray \
    --draft-model-path $DRAFT_PATH  > ${log_path}/${model}.log 2>&1 #

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
sed -n '/======== ngrams 测试结果 ========/,/=================================/p' ${log_path}/${model}.log
sed -n '/======== eagle3 测试结果 ========/,/=================================/p' ${log_path}/${model}.log
sed -n '/======== draft 测试结果 ========/,/=================================/p' ${log_path}/${model}.log


echo "=========================================================="
last_line=$(tail -n 1 "${log_path}/${model}.log")
RESULT=$(echo "$last_line" | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1)
echo " 测试全部完成！"
echo "$(basename ${model})正确率：${RESULT}%"
echo "=========================================================="