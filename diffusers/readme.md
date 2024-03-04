运行描述文件

## 目录结构及说明

```
    ├── code                     #  Contains test code.  
    │   ├── utils                #  conformer 替代 onnx 实现核心代码
    |   |── sample_onnx.py 
    │                                                                                         
    ├── readme.md                 #  任务模型运行说明
    ├── models
        ├── models                   #  模型文件夹 
        │   └── config.json          #  模型参数   
        │                                          
        o
        o── other models
        o   └── config.json 
```

## environment build 

# 第一步 git clone modelZoo 部分机子下载不了需要找IT帮忙看下  在gerrit找ssh下载链接
```
cd your_work_path
git clone "ssh://***@mercury.metax-tech.com:29418/PDE/AI/ModelZoo.LLM.Inference"
```

# 第二步 加载docker (可选)  
参见xwiki   待建

# 第三步 进入docker工作目录 开ssh 服务  （可选）
待补充

# 第四步 获取数据集  通过软连接获取
待补充



## 运行
python code/sample_onnx_multithreads.py your/onnx/model/path 1 fp16 norm test maca
例如：
```python
python code/sample_onnx.py ./models/ox_sd_15/  1 fp16 norm test maca # 执行fp16 C500推理. Demo 测试跑1条数据
python code/start.py ./models/ox_sd_15/  1 fp16 norm test maca 16 # 执行fp16 C500推理. 全量多线程 
python code/start.py ./models/ox_sd_15/  1 fp32 norm test cpu  16 # 执行fp32 CPU推理. 全量多线程
```
参数
# modelpath：  需测试的模型路径, 内部包含config.json模型参数文件已配置好
# batchsize:   推理的batchsize, 可设置1, 暂时只支持bs1
# precision:   推理精度包含, fp32：使用ort-fp32精度, fp16：使用ort-fp16精度(暂未支持), int8：暂未支持, qdq：暂未支持
# task:        测试任务，当前暂未用上
# test_data:   test 表示读取test数据， dev 表示读取 dev数据。见 data
# EP:          设置onnxruntime运行的EP, 默认为maca, 可选EP有cpu,gpu,trt, 需根据环境中存在的EP进行设置
# num_thr:     测算性能时，开启的线程数量 默认8