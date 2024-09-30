export CUBLAS_WORKSPACE_CONFIG=:4096:16
python src/benchmark_latency.py --model /pde_ai/models/llm/Llama/Llama-2-7b-hf
unset CUBLAS_WORKSPACE_CONFIG