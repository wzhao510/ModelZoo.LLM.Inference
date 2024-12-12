# Overview
[lmdeploy](https://github.com/InternLM/lmdeploy) 是一个用于大型语言模型（LLMs）和视觉-语言模型（VLMs）压缩、部署和服务的 Python 库。 其核心推理引擎包括 TurboMind 引擎和 PyTorch 引擎, 目前在沐曦平台上只适配了Pytorch引擎。
[dlinfer](https://github.com/DeepLink-org/dlinfer)提供了一套将国产硬件接入大模型推理框架的解决方案。 对上承接lmdeploy框架，对下对接沐曦算子。 
也就是说，lmdeploy要跑在沐曦平台上需要借助dlinfer提供算子适配。

# Installation
依赖c500/modelzoo.llm.vllm (0.6.2)镜像。

## 1. maca env
```
source env.sh
```
## 2. metax python pacakge dep
安装以下matax包。如果是c500/modelzoo.llm.vllm的docker镜像的话，这些依赖包应该都已经安装。
```
apex
dropout_layer_norm
flash_attn
fused_dense_lib
rotary_emb
torch
torchaudio
torchvision
triton
xentropy_cuda_lib
xformers
```
用pip list | grep metax查看。例如：
```
apex                              0.1+metax2.25.2.1
dropout_layer_norm                0.1+metax2.25.2.1torch2.1
flash_attn                        2.5.3+metax2.25.2.1torch2.1
fused_dense_lib                   2.5.3+metax2.25.2.1torch2.1
rotary_emb                        0.1+metax2.25.2.1torch2.1
torch                             2.1.2+metax2.25.2.1
torchaudio                        2.0.1+metax2.25.2.1
torchvision                       0.15.1+metax2.25.2.1
triton                            2.1.0+metax2.25.2.8
xentropy_cuda_lib                 0.1+metax2.25.2.1torch2.1
xformers                          0.0.22+metax2.25.2.1torch2.1
```

## 3. dlinfer install
```
# 内网
git clone git@pdegit.metax-internal.com:pde-ai/DeepLink-org/dlinfer.git
# 外网
git clone https://github.com/DeepLink-org/dlinfer.git (commit 8facf9f00d5352a2f2cc22c486b978d662192d8f)

cd dlinfer
rm -rf _skbuild
pip install -r requirements/maca/full.txt
DEVICE=maca python3 setup.py develop
```

## 4. lmdeploy install
```
# 内网
git clone git@pdegit.metax-internal.com:pde-ai/InternLM/lmdeploy.git
# 外网
git clone https://github.com/InternLM/lmdeploy.git (commit 4f7e50b86a1c99b706f78efa0543ce4ccc5f5628)

# TODO: 替换lmdeploy原来的requirement，因为沐曦对某些包进行了适配版本有差异。
cp runtime.txt lmdeploy/requirements
cd lmdeploy

pip install -r requirements.txt
pip install -e .
```

# Test
## 1. 模型列表
沐曦环境现在只支持pytorch后端。支持以下模型（待完善)
- <input type="checkbox" checked> Qwen2-VL-7B-Instruct
- <input type="checkbox" checked> Qwen2-VL-72B-Instruct
- <input type="checkbox" checked> MiniCPM-V 2.6
- <input type="checkbox" checked> InternVL2-8B
- <input type="checkbox" checked> cogvlm2-llama3-chinese-chat-19B


## 2. 离线测试
通过离线pipline测试特定query的回答，我们以InternVL2-8B为例。
```
cd code
python chat_prompt_demo.py --model-path /pde_ai/models/llm/Internlm/InternVL2-8B --tp 1 --block-size 16 --cache-max-entry-count 0.8 --vl True --image-url ../resources/tiger.jpeg
```
输出如下：
```
As an AI language model, I don't have feelings, but I'm functioning properly and ready to assist you. How can I help you today?

batch_0:
Q: How are you?
A: As an AI language model, I don't have feelings, but I'm functioning properly and ready to assist you with any questions or tasks you may have. How can I help you today?
batch_1:
Q: Please introduce Shanghai.
A: Shanghai is a city in eastern China. It is the largest city in China. It is also the largest city in the world. Shanghai is a very modern city. It has many tall buildings. It has many skyscrapers. Shanghai is a very busy city. It is a very important city in China. It is a very important city in the world. Shanghai is a very beautiful city. It has many parks. It has many beautiful gardens. Shanghai is a very interesting city. It has many interesting things to see and do.

session 1:  USER:
('please describe this image.', <PIL.Image.Image image mode=RGB size=278x182 at 0x7FC28FE0FA30>)
ASSISTANT:
The image depicts a majestic tiger lying on a lush green grassy area. The tiger is facing the camera, with its head slightly raised and its eyes looking directly at the viewer. The tiger's fur is predominantly orange with distinct black stripes that run vertically along its body. The stripes are particularly prominent on its face, neck, and back. The tiger's paws are visible, with one paw resting on the ground and the other slightly raised. The background consists of a well-maintained grassy field, suggesting a natural or semi-natural habitat. The lighting in the image is bright, indicating that it was taken during the daytime. The overall scene conveys a sense of calm and serenity.

session 2:  USER:
('please describe this image.', <PIL.Image.Image image mode=RGB size=278x182 at 0x7FC28FE0FA30>)
ASSISTANT:
The image depicts a majestic tiger lying on a lush green grassy area. The tiger is facing the camera, with its head slightly raised and its eyes looking directly at the viewer. The tiger's fur is predominantly orange with distinct black stripes that run vertically along its body. The stripes are particularly prominent on its face, neck, and back. The tiger's paws are visible, with one paw resting on the ground and the other slightly raised. The background consists of a well-maintained grassy field, suggesting a natural or semi-natural habitat. The lighting in the image is bright, indicating that it was taken during the daytime. The overall scene conveys a sense of calm and serenity.
USER:
What is the woman doing?
ASSISTANT:
The image does not show a woman. It depicts a tiger lying on a grassy area. The tiger is facing the camera, with its head slightly raised and its eyes looking directly at the viewer. The tiger's fur is orange with black stripes, and it is resting on a lush green grassy field.
```


## 3. 静态推理性能测试
我们把推理引擎在固定 batch、固定输入输出 token 数量的前提下的推理，称之为静态推理。评测脚本是 profile_generation.py，采用dummpy data，即随机数。
我们以[internlm/internlm-7b](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm-7b)为例，介绍测试 LMDeploy pytorch 推理引擎的静态推理性能测试方法。

```
python profile_generation.py /pde_ai/models/llm/Internlm/internlm2-chat-7b --backend pytorch -c 1 -pt 256 -ct 128 --tp 1 --cache-block-seq-len 16 --dtype float16
```
输出如下：
```
--------------------------------------------------
total time: 5.51s
concurrency: 1, test_round: 3
input_tokens: 256, output_tokens: 128
first_token latency(min, max, ave): 0.045s, 0.047s, 0.046s
total_token latency(min, max, ave): 1.833s, 1.846s, 1.839s
token_latency percentiles(50%,75%,95%,99%)(s): [0.014, 0.014, 0.02, 0.021]
throughput(output): 69.68 token/s
throughput(total): 209.04 token/s
--------------------------------------------------
```
测量指标：
首字延迟 = TTFT = first_token latency
每个输出token平均时延（不包括首个token) = TPOT = 1000 / throughput(output)
token 吞吐量 （包括首个token）= tokens/s = throughput(total)


## 4. 服务化请求吞吐量性能测试
在真实应用中，用户输入的 prompt 长度以及模型回复的 token 数量是动态变化的。而静态推理能力不足以反映推理引擎对动态输入输出的处理能力。
所以需要使用真实对话数据，评测推理引擎的动态推理能力。下载[ShareGPT_V3_unfiltered_cleaned_split.json](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json)。下面将介绍如何在 localhost 采用server api上测试 LMDeploy 的动态推理性能。我们仍然以[internlm/internlm-7b](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm-7b)为例。


```
# 启动模型服务
lmdeploy serve api_server --server-port 23333 --tp 1 --backend pytorch --max-batch-size 256 /pde_ai/models/llm/Internlm/internlm2-chat-7b --dtype float16 --device maca --cache-block-seq-len 16
```

输出如下：
```
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
HINT:    Please open http://0.0.0.0:23333 in a browser for detailed api usage!!!
INFO:     Started server process [1384373]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:23333 (Press CTRL+C to quit)
```

```
# 吞吐测试
python profile_restful_api.py --port 23333 --backend lmdeploy --dataset-path ../resources/ShareGPT_V3_unfiltered_cleaned_split.json
```

输出如下：
```
============ Serving Benchmark Result ============
Backend:                                 lmdeploy  
Traffic request rate:                    inf       
Successful requests:                     1000      
Benchmark duration (s):                  96.23     
Total input tokens:                      228316    
Total generated tokens:                  195534    
Total generated tokens (retokenized):    181775    
Request throughput (req/s):              10.39     
Input token throughput (tok/s):          2372.62   
Output token throughput (tok/s):         2031.95   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   43531.86  
Median E2E Latency (ms):                 42845.51  
---------------Time to First Token----------------
Mean TTFT (ms):                          26513.87  
Median TTFT (ms):                        24519.39  
P99 TTFT (ms):                           62724.06  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          110.65    
Median TPOT (ms):                        95.38     
P99 TPOT (ms):                           535.69    
---------------Inter-token Latency----------------
Mean ITL (ms):                           91.67     
Median ITL (ms):                         61.98     
P99 ITL (ms):                            724.82    
==================================================
```

测量指标：
首字延迟 = TTFT = Mean TTFT
每个输出token平均时延（不包括首个token) = TPOT = Mean TPOT
token 吞吐量 （包括首个token）= tokens/s = Input token throughput (tok/s) + Output token throughput (tok/s)

## 5. OpenCompass测评
OpenCompass是一个专门用于语言大模型与多模态大模型的评测工具。

### 安装 OpenCompass (tag: 0.3.6)
```
# 内网
git clone git@pdegit.metax-internal.com:pde-ai/open-compass/opencompass.git
# 外网
git clone https://github.com/open-compass/opencompass.git (commit ff831b153e3f81f80ac84a56a254dbb4cbad95c9)

cd opencompass
git apply opencompass.patch
pip install --use-pep517 nltk==3.8
pip install -e .
```

### 下载数据集
```
wget https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip
unzip OpenCompassData-core-20231110.zip
```

### 执行评测任务
```
# opencompass 根目录下，
python run.py {LMDEPLOY_MODELZOO}/models/internlm2-chat-7b/eval_lmdeploy_internlm2-chat-7b.py
```

输出如下：
```
| dataset | version | metric | mode | internlm2-chat-7b-pytorch |
|----- | ----- | ----- | ----- | -----|
| lukaemon_mmlu_college_biology | 0eddf6 | accuracy | ppl | 71.53 |
| lukaemon_mmlu_college_chemistry | 9e2300 | accuracy | ppl | 48.00 |
| lukaemon_mmlu_college_computer_science | 5a1625 | accuracy | ppl | 57.00 |
| lukaemon_mmlu_college_mathematics | 13e9be | accuracy | ppl | 33.00 |
| lukaemon_mmlu_college_physics | f05705 | accuracy | ppl | 41.18 |
| lukaemon_mmlu_electrical_engineering | 87e5d9 | accuracy | ppl | 60.69 |
| lukaemon_mmlu_astronomy | 69352a | accuracy | ppl | 69.74 |
| lukaemon_mmlu_anatomy | a367fc | accuracy | ppl | 60.74 |
...
```


