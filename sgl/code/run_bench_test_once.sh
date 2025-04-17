python ./bench_test.py \
--model ./models/DeepSeek-R1-Distill-Qwen-7B/ \
--dataset-path /pde_ai/datasets/ShareGPT_V3/ShareGPT_V3_unfiltered_cleaned_split.json \
--enable-ep-moe \
--ep-size 1 \
--enable-dp-attention \
--dp-size 1 \
--random-input-len 64 \
--random-output-len 32 \
--num-prompts 1 \