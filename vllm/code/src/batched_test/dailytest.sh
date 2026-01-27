SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $SCRIPT_DIR

out_dir=/workspace/dailytest-debug

# run C500 dailytest
python launch.py --perf --model-config $SCRIPT_DIR/configs/dailytest_models_C500.yaml --work-dir $out_dir

# run C600 dailytest
# python launch.py --perf --model-config $SCRIPT_DIR/configs/dailytest_models_C600.yaml --work-dir $out_dir