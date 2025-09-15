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

## 数据集精度测试 (基于 lm-eval 框架)
### 依赖环境
```shell
lm-eval=0.4.5
```

#### 支持的数据集
- 大语言模型: c-eval, mmlu
- 多模态模型: mmmu

#### 脚本说明
- `use_cmd=True`: 使用命令行方式运行。
- `use_cmd=False`: 使用 subprocess 方式运行，结果会自动保存到 `./model/performance.json`。

### C-Eval
1. **下载数据集**
   - 如果是本地运行需要修改路径，否则会一直尝试网络下载。
   - 下载好的数据集文件应包含以下结构：
     ```
    ├── ceval-exam
    │   ├── ceval-exam.py 
    │   ├── ceval-exam.zip 
    │   ├── dev
    │   ├── test
    │   └── val
     ```

2. **修改 `ceval-exam.py` 文件**
   - 打开 `ceval-exam.py` 文件，找到第38行，将 `_URL` 修改为 `data.zip` 所在路径：
     ```python
     # 将 _URL=r"https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip" 修改为
     _URL=r"Your ceval-exam zip path"
     ```

3. **修改数据集路径**
   - 编辑 `/opt/conda/lib/python3.8/site-packages/lm_eval/tasks/ceval/_default_ceval_yaml` 文件：
    - 如果是 Python 3.10，则将路径中的 `python3.8` 替换为 `3.10`。
     ```yaml
     # 将 dataset_path: ./ceval-exam 修改为本地 ceval-exam 存放路径
     dataset_path: /path/to/your/ceval-exam
     ```

4. **执行方式**
   - 运行以下命令来评估模型：
     ```bash
     python code/lm-eval/c-eval.py ./models/xxxx
     ```

### MMLU
1. **下载数据集**
   - 如果是本地运行需要修改路径，否则会一直尝试网络下载。
   - 下载好的数据集文件应包含以下结构：
     ```
     .
     ├── mmlu_no_train
     │   ├── mmlu_no_train.py
     │   ├── data.zip
     │   ├── all
     │   └── README.md
     ```

2. **修改 `mmlu_no_train.py` 文件**
   - 打开 `mmlu_no_train.py` 文件，找到第37行，将 `_URL` 修改为 `data.zip` 所在路径：
     ```python
     # 将 _URL=r"https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/mmludata.zip" 修改为
     _URL=r"Your data zip path"
     ```

3. **修改数据集路径**
   - 编辑 `/opt/conda/lib/python3.8/site-packages/lm_eval/tasks/mmlu/default/_default_template_yaml` 文件：
    - 如果是 Python 3.10，则将路径中的 `python3.8` 替换为 `3.10`。
     ```yaml
     # 将 dataset_path: ./mmlu_no_train 修改为本地 mmlu_no_train 存放路径
     dataset_path: /path/to/your/mmlu_no_train
     ```

4. **执行方式**
   - 运行以下命令来评估模型：
     ```bash
     python code/lm-eval/mmlu.py ./models/xxxx
     ```

### MMMU
1. **下载数据集**
   - 如果是本地运行需要修改路径，否则会一直尝试网络下载。
   - 下载好的数据集文件应包含以下结构：
     ```
     .
     ├── MMMU
     │   ├── Accounting
     │   ├── Agriculture
     │   ├── Agriculture_and_Engineering
     │   ├── Art
     ....
     │   ├── Sociology
     │   └── README.md
     ```


2. **修改数据集路径**
   - 编辑 `/opt/conda/lib/python3.8/site-packages/lm_eval/tasks/mmmu/_template_yaml` 文件：
    - 如果是 Python 3.10，则将路径中的 `python3.8` 替换为 `3.10`。
     ```yaml
     # 将 dataset_path: ./MMMU 修改为本地 MMMU 存放路径
     dataset_path: /path/to/your/MMMU
     ```

3. **执行方式**
   - 运行以下命令来评估模型：
     ```bash
     python code/lm-eval/mmmu.py ./models/xxxx
     ```

## throughput 性能基准测试

### 执行方式
```
python code/bench_test.py --model ./models/xxxx --num-prompts 24      # 默认输入长度1024 输出长度1024
python code/bench_test.py --model ./models/xxxx --num-prompts 1024 --input-len 512 --output-len 128   # 输入长度512 输出长度 128
```

### 参数列表
- `model`: 模型路径 (必填)
- `num-prompts`: 测试 prompts 数量 (默认 32)
- `input-len`: 测试输入长度 (默认 1024)
- `output-len`: 测试输出长度 (默认 1024)
- `tensor-parallel-size`: tensor-parallel-size (默认为配置中的设置)

### # bge_vl_large测试脚本
```bash
python code/src/bge_vl_large.py --model /pde_ai/models/llm/BAAI/BGE-VL-large/ --image-path ./data/demo.jpg
# 参数列表
--model-dir MODEL_DIR               model path
--num-iterations NUM_ITERATIONS     test rounds
--input-text INPUT_TEXT             input text
--image-path IMAGE_PATH             path to image
```

## 大语言模型推理
- 需要根据使用模型实际路径修改模型所在的绝对路径

### 1. 本地推理脚本 (`run_offline_inference_llm_demo.py`)
- 根据需求修改 prompts。

### 2. 通过put 性能基准测试 (`benchmark_throughput_llm.py`)
- 根据需求修改 `input-len`, `output-len`, `batch-size` 等参数。

## 多模态模型推理

### 1. 本地推理脚本 (`run_offline_inference_multimodal_demo.py`)
- 根据模型路径修改脚本中的路径。
- 根据需求修改 prompts。

### 2. 性能基准测试 (`benchmark_throughput_multimodal.py`)
- 根据模型路径修改脚本中的路径。
- 根据需求修改 `input-len`, `output-len`, `batch-size` 等参数。

---

希望以上文档能更好地帮助您理解和使用本项目。如有任何问题，欢迎联系项目维护者。