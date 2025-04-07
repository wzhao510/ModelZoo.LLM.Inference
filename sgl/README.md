## throughput 目录结构
```
sgl
├── code
│   ├── sglang
│   │   ├── bench_offline_throughput.py
│   ├── utils
│   ├── bench_test.py
│   ├── run_bench_test_batched.sh
│   ├── run_bench_test_once.sh
├── data
├── models
└── README.md
```


## throughput 执行
* 需要准备ShareGPT_V3_unfiltered_cleaned_split.json数据，请从https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json下载
* 跑看护的35个case性能数据:在code目录下运行run_bench_test_batched.sh
* 跑单次throughput在code目录下运行run_bench_test_batched.sh
* 脚本内需要根据使用模型情况修改模型所在目录
* 可根据测试需要在脚本内修改input-len、output-len、batch-size、dataset-path等参数


