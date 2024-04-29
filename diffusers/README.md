运行描述文件
## requirement
* 需要 onnxsim=0.4.36 
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

## 运行测试
./runtest.sh ./models/ox_sd_15/ 1 fp16 norm test maca
./runtest.sh ./models/ox_sd_21/ 1 fp16 norm test maca
./runtest.sh ./models/ox_sd_21_base/ 1 fp16 norm test maca


## 其它例子：
```python
python code/sample_onnx.py ./models/ox_sd_15/  1 fp16 norm test maca # 执行fp16 C500推理. Demo 测试跑2条数据
python code/sample_onnx.py ./models/ox_sd_15_bs/  1 fp16 norm test maca # 执行fp16 C500推理. Demo 测试跑2条数据 (动态shape和动态batch支持，默认出图照片为 960x960)

```

参数
# modelpath：  需测试的模型路径, 内部包含config.json模型参数文件已配置好
# batchsize:   推理的batchsize, 可设置1, 暂时只支持bs1
# precision:   推理精度包含, fp32：使用ort-fp32精度, fp16：使用ort-fp16精度(暂未支持), int8：暂未支持, qdq：暂未支持
# task:        测试任务，当前暂未用上
# test_data:   test 表示读取test数据， dev 表示读取 dev数据。见 data
# EP:          设置onnxruntime运行的EP, 默认为maca, 可选EP有cpu,gpu,trt, 需根据环境中存在的EP进行设置
# num_thr:     测算性能时，开启的线程数量 默认8
