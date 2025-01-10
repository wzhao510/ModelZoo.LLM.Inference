## 目录结构及说明

```
.
├── code
│   ├── bench_test.py 
│   ├── lm-eval
│   │   ├── c-eval.py
│   │   ├── mmlu.py
│   │   └── mmmu.py
│   └── src
│       ├── benchmark_throughput_llm.py
│       ├── benchmark_throughput_multimodal.py
│       ├── run_offline_inference_llm_demo.py
│       ├── run_offline_inference_multimodal_demo.py
|       └──torch_profile_utils.py
│   └── utils
│       ├── __init__.py     
│       └── utils.py
│   └── tools
│       └── statistics_csv.py
├── data
│   └── demo.jpg
└── README.md
```

## lm-eval框架下数据集精度测试
依赖环境
```shell
lm-eval=0.4.5
```

- 大语言模型支持ceval、mmlu数据集测试
- 多模型模型支持mmmu数据集测试

- 脚本中use_cmd=True
    - 表示使用命令行方式运行
    - 否则使用subprocess方式运行，采用subprocess方式会自动保存结果到./model/performance.json

### C-Eval
1. 下载数据集
如果是本地运行需要修改下路径，否则会一直尝试网络下载：
将下载好的数据集放到./lm_eval/ceval/ceval-exam
进入./lm_eval/ceval/ceval-exam/ceval-exam.py +38
    将_URL=r"https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip" 修改为 _URL=r"Your ceval zip path"

2. 修改数据集路径
vi ./lm_eval/tasks/ceval/_default_ceval_yaml 
将数据集路径修改为步骤一数据集所在路径：eg: /opt/conda/lib/python3.8/site-packages/lm_eval/ceval/ceval-exam

3. 执行方式
python code/lm-eval/c-eval.py  ./models/xxxx  

### MMLU
1. 下载数据集
如果是本地运行需要修改下路径，否则会一直尝试网络下载：
将下载好的数据集放到./lm_eval/mmlu_no_train
进入./lm_eval/mmlu_no_train/mmlu_no_train.py +38
    将_URL=r"https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/mmludata.zip" 修改为 _URL=r"Your data zip path"
    eg./opt/conda/lib/python3.8/site-packages/lm_eval/mmlu_no_train/data.tar

2. 修改数据集路径
vi ./lm_eval/tasks/mmul/default/_default_template_yaml 
将数据集路径修改为步骤一数据集所在路径：eg: /opt/conda/lib/python3.8/site-packages/lm_eval/mmlu_no_train

3. 执行方式
python code/lm-eval/mmlu.py  ./models/xxxx 

### MMMU
1. 下载数据集
如果是本地运行需要修改下路径，否则会一直尝试网络下载：
将下载好的数据集放到./lm_eval/MMMU

2. 修改数据集路径
vi ./lm_eval/tasks/mmul/_template_yaml
将数据集路径修改为步骤一数据集所在路径：eg: /opt/conda/lib/python3.8/site-packages/lm_eval/MMMU

3. 执行方式
python code/lm-eval/mmmu.py  ./models/xxxx 

## benchmark throughput 执行
```
python code/bench_test.py  --model ./models/xxxx --num-prompts 24                                         # 跑24条数据测试 ，默认： 输入长度1024 输出长度1024
python code/bench_test.py  --model ./models/xxxx --num-prompts 1024 --input-len 512 --output-len 128      # 跑1024条数据测试，设置: 输入长度512 输出长度 128
```
当前支持参数列表如下：
 - model:&nbsp;&nbsp;原始模型路径，必须设置
 - num-prompts:&nbsp;&nbsp;测试prompts数量，可以类比为batchsize， 默认为32
 - input-len:&nbsp;&nbsp;测试输入长度， 默认为1024
 - output-len:&nbsp;&nbsp;测试输出长度， 默认为1024
 - tensor-parallel-size:&nbsp;&nbsp;tensor-parallel-size，默认为config的配置


### 新增环境变量 `MX_VLLM_ENABLE_PROFILE`
* 使能 `MX_VLLM_ENABLE_PROFILE`环境变量后将会在 `./mx_profiler/` 文件夹通过torch_profiler 工具生成csv原始文件（如果跑35个case的话，目前只统计 input_len=256,output_len=128 以及 input_len=1024,output_len=1024 数据）

生成的对应文件夹路径下的csv 可以通过 以下脚本完成 kernel 汇总（注：需要 安装openxl包： `pip install openxl`）
```shell
python ./vllm/code/tools/statistics_csv.py ./mx_profiler/
```

## 大语言模型推理
## 1、本地推理脚本run_offline_inference_llm_demo.py
    脚本内需要根据使用模型情况修改模型所在目录，如/pde_ai/models/llm/Llama/Llama-2-7b-hf/
    可根据需要修改prompts

## 2、benchmark_throughput
##   Benchmark offline inference throughput.
    脚本内需要根据使用模型情况修改模型所在目录，如/pde_ai/models/llm/Llama/Llama-2-7b-hf/
    可根据测试需要在脚本内修改input-len、output-len、batch-size等参数，详细参数信息见/code/src/benchmark_throughput_llm.py

## 多模态模型推理
建议测试model文件夹下的llava、Qwenvl系列模型
## 1、本地推理脚本run_offline_inference_multimodal_demo.py
    脚本内需要根据使用模型情况修改模型所在目录，如/pde_ai/models/llm/LLaVa/llava-v1___6-vicuna-13b-hf
    可根据需要修改prompts

## 2、benchmark_throughput
##   Benchmark offline inference throughput.
    脚本内需要根据使用模型情况修改模型所在目录，如/pde_ai/models/llm/LLaVa/llava-v1___6-vicuna-13b-hf
    可根据测试需要在脚本内修改input-len、output-len、batch-size等参数，详细参数信息见/code/src/benchmark_throughput_multimodal.py

