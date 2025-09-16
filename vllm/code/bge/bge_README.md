# bge_embedding.py
- 从--text-file读取--input-len长度的字符，并将其转换成词向量，记录tps与时间戳
```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python bge_embedding.py \
--model /pde_ai/models/llm/BAAI/bge-large-zh/ --max-model-len 8192  \
--text-file /workspace/ModelZoo.LLM.Inference/vllm/data/input_data.txt \
--trust-remote-code --max-num-batched-tokens 8192 --batch-size 8

--model                                             模型路径
--gpu-memory-utilization                            gpu利用率
--max-model-len                                     Maximum model length
--trust-remote-code                                 Trust remote code
--dtype {auto,half,float16,bfloat16,float,float32}  Model data type
--max-num-seqs                                      最大并发数
--max-num-batched-tokens                            每步最大批处理token数
--batch-size                                        并发数
--text-file                                         从此文件读取字符，根据input_len构建prompt
--profile                                           使用torch.profiler抓取profile

```
# bge_reranker.py
- 生成--num-queries组query list和--num-queries * --docs-per-query组doc list,为所有query list与doc list进行相关性评分，并记录tps、时间戳与相关性分数
```bash
python bge_reranker.py --model /pde_ai/models/llm/BAAI/bge-reranker-v2-m3/ \
--max-model-len 8192 --num-queries 100 --docs-per-query 10 \
--min-length 20 --max-length 50 --trust-remote-code

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
--profile                                           使用torch.profiler抓取profile
```

# xinference
## requirements
pip install xinference[vllm]
pip install xinference[embedding]
pip install xinference[rerank]
## 启动xinferencef服务
xinference #默认端口9997  

## 加载模型
- 命令行启动
```bash
# embedding模型
 xinference launch --model-name bge-small-zh-v1.5 --model-engine vllm \ 
 --model-type embedding --model-path /pde_ai/models/llm/BAAI/bge-small-zh-v1.5 --gpu-idx 0
# rerank模型
 xinference launch --model-name bge-reranker-v2-m3 --model-engine vllm \
 --model-type rerank --model-path /pde_ai/models/llm/BAAI/bge-reranker-v2-m3 --gpu-idx 0
 ```

- python启动
```python
import os
from xinference.client import Client
client = Client("http://localhost:9997")

#model_path = "/pde_ai/models/llm/BAAI/bge-reranker-v2-m3"
model_path = "/pde_ai/models/llm/BAAI/bge-small-zh-v1.5"
print(os.listdir(model_path))

model = client.launch_model(
    #model_name="bge-reranker-v2-m3",
    model_name="bge-small-zh-v1.5",
    model_engine="vllm",
    #model_type="rerank"
    model_type="embedding",
    gpu_idx=1,
    model_path=model_path
)
```

## curl请求
```bash
# "model"字段需要与 xinference launch输出的Model uid(xinference launch指定的model name)一致
curl -X POST "http://127.0.0.1:8000/v1/embeddings" -H "Content-Type: application/json" \
-d '{
  "model": "bge-small-zh-v1.5",
  "input": "请介绍下你自己"  
}'

curl -X POST "http://127.0.0.1:9997/v1/rerank"  \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "rerank",
    "documents": ["请介绍下你自己", "ioafnwaoianorank", "nishusa"]
  }'
```
## benchmark
```bash

python bge_xinfer_benchmark_serving.py --model bge-small-zh-v1.5 \
 --model-type embedding --port 9997 --input-len 2048

--model-type  {embedding,rerank}        模型类型,embedding或rerank
--model MODEL                           Model name,需要与 xinference launch输出的Model uid一致
--port PORT                             Xinference service port
--input-len INPUT_LEN                   input length
--num-tests NUM_TESTS                   测试次数, 默认10次取平均
```
