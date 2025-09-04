# bge_embedding.py
- 从--text-file读取--input-len长度的字符，并将其转换成词向量，记录tps与时间戳
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python bge_embedding.py --model /pde_ai/models/llm/BAAI/bge-large-zh/ --max-model-len 8192 --input-len 512 --text-file /workspace/ModelZoo.LLM.Inference/vllm/data/input_data.txt --trust-remote-code

--model                                             模型路径
--gpu-memory-utilization                            gpu利用率
--max-model-len                                     Maximum model length
--trust-remote-code                                 Trust remote code
--dtype {auto,half,float16,bfloat16,float,float32}  Model data type
--text-file                                         从此文件读取字符，根据input_len构建prompt
--input-len                                         Number of texts to read, -1 means read all texts

# bge_reranker.py
- 生成--num-queries组query list和--num-queries * --docs-per-query组doc list,为所有query list与doc list进行相关性评分，并记录tps、时间戳与相关性分数
python bge_reranker.py --model /pde_ai/models/llm/BAAI/bge-reranker-v2-m3/ --max-model-len 8192 --num-queries 100 --docs-per-query 10 --min-length 20 --max-length 50 --trust-remote-code

--model MODEL                                       模型路径
--gpu-memory-utilization                            gpu利用率
--max-model-len MAX_MODEL_LEN                       Maximum model length
--trust-remote-code                                 Trust remote code
--dtype {auto,half,float16,bfloat16,float,float32}  Model data type
--num-queries                                       生成query list的数量
--docs-per-query                                    每个query list中含有多少个文本
--min-length                                        list中单个文本最小长度
--max-length                                        list中单个文本最大长度
--seed                                              随机数种子,prompt使用随机数构建