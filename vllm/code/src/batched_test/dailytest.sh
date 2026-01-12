SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd $SCRIPT_DIR

out_dir=/workspace/dailytest-debug

python launch.py --perf --model-config $SCRIPT_DIR/configs/dailytest_models.yaml --work-dir $out_dir