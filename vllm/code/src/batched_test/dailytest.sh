SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $SCRIPT_DIR

out_dir=/workspace/dailytest-debug

# 添加长prompt测试
# --long-text-case configs/inference/long_text_case.yaml

# run C500 dailytest
python launch.py --perf --model-config $SCRIPT_DIR/configs/dailytest_models_C500.yaml --work-dir $out_dir

# run C600 dailytest（配置文件分拆为 C600-1 / C600-2）
python launch.py --perf --model-config $SCRIPT_DIR/configs/dailytest_models_C600-1.yaml --work-dir $out_dir
python launch.py --perf --model-config $SCRIPT_DIR/configs/dailytest_models_C600-2.yaml --work-dir $out_dir

# run C588 dailytest（新平台，集群就绪后取消注释）
# python launch.py --perf --model-config $SCRIPT_DIR/configs/dailytest_models_C588.yaml --work-dir $out_dir

# run C600-U dailytest（新平台，集群就绪后取消注释）
# python launch.py --perf --model-config $SCRIPT_DIR/configs/dailytest_models_C600U.yaml --work-dir $out_dir