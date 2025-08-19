运行描述文件
## requirement
* 需要 onnxsim=0.4.36 
* sample_onnx、sample_sd3_onnx、sample_sdxl增加精度测试功能
* 使用ViT-H-14计算精度，需要挂载数据集，默认路径 /external/ai/models/llm/CLIP/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
* 数据集获取路径https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main
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


## demo 示例：
```python
python code/sample_onnx.py ./models/ox_sd_15/  1 fp16 norm test maca # 执行fp16 C500推理. Demo 测试跑10组数据

python code/sample_onnx.py ./models/ox_sd_15_bs/  1 fp16 norm test maca # 执行fp16 C500推理. Demo 测试跑10组数据 (动态shape和动态batch支持，默认出图照片为 960x960)
python code/sample_onnx.py ./models/ox_sd_21_bs/  1 fp16 norm test maca # 执行fp16 C500推理. Demo 测试跑10组数据 (动态shape和动态batch支持，默认出图照片为 960x960)

python code/sample_onnx.py ./models/ox_sd_15_bs/  2 fp16 norm test maca 512 # 执行fp16 C500推理. Demo 测试跑10组数据 (batchsize=2，出图照片为 512x512)
python code/sample_onnx.py ./models/ox_sd_21_bs/  2 fp16 norm test maca 512 # 执行fp16 C500推理. Demo 测试跑10组数据 (batchsize=2，出图照片为 512x512)

python code/sample_onnx.py ./models/ox_sd_15_static/  1 fp16 norm test maca 512 # 执行fp16 C500推理(静态shape). Demo 测试跑10组数据 (batchsize=1，出图照片为 512x512)
python code/sample_onnx.py ./models/ox_sd_21_static/  8 fp16 norm test maca 768 # 执行fp16 C500推理(静态shape). Demo 测试跑10组数据 (batchsize=8，出图照片为 768x768)

```

## 数据集测试
本测试需要使用PartiPrompts数据集，请在测试前从https://huggingface.co/datasets/nateraw/parti-prompts/blob/main/PartiPrompts.tsv下载，并放至./data路径。
```python
python code/sample_onnx_multithreads.py ./models/ox_sd_15/  1  fp16 norm test maca 8 # 执行fp16 C500推理. 以batch=1，8个线程运行。模型不带_bs 只支持 bs=1 运行
python code/sample_onnx_multithreads.py ./models/ox_sd_15_bs/  2  fp16 norm test maca 8  # 执行fp16 C500推理. 以batch=2，8个线程运行
python code/sample_onnx_multithreads.py ./models/ox_sd_15_static/  2  fp16 norm test maca 8  # 执行fp16 C500推理(静态shape). 以batch=2，8个线程运行
```

参数
# modelpath：  需测试的模型路径, 内部包含config.json模型参数文件已配置好
# batchsize:   推理的batchsize, 可设置1, 暂时只支持bs1
# precision:   推理精度包含, fp32：使用ort-fp32精度, fp16：使用ort-fp16精度(暂未支持), int8：暂未支持, qdq：暂未支持
# task:        测试任务，当前暂未用上
# test_data:   test 表示读取test数据， dev 表示读取 dev数据。见 data
# EP:          设置onnxruntime运行的EP, 默认为maca, 可选EP有cpu,gpu,trt, 需根据环境中存在的EP进行设置
# num_thr:     测算性能时，开启的线程数量 默认8


## sd1.5/2.1 demo 示例：
```python
python code/sample_onnx.py ./models/ox_sd_15_bs/ 1 fp16 norm test maca 512 0 10 # 执行fp16 C500推理. Demo 测试跑10组数据，不跳过模型转换 (动态shape和动态batch支持，默认出图照片为 512x512)
```
参数
# modelpath：  需测试的模型路径, 内部包含config.json模型参数文件已配置好
# batchsize:   推理的batchsize, 可设置1, 暂时只支持bs1
# precision:   推理精度包含, fp32：使用ort-fp32精度, fp16：使用ort-fp16精度(暂未支持), int8：暂未支持, qdq：暂未支持
# task:        测试任务，当前暂未用上
# EP:          设置onnxruntime运行的EP, 默认为maca, 可选EP有cpu,maca
# size:        生成图片尺寸 (size*size)
# skip_convert 是否跳过转化fp16模型，转化后的模型保存在./temp_model中，默认0，不跳过转化
# test_round   测试轮次，测试test_round次，结果取平均，默认10，测试十次取平均

输出示例
average score: 0.380
StableDiffusion_ox_sd_15_bs_bs1_precfp16 FPS : 0.283, latency : 3532.173ms, memory usage: 7.686 GB
StableDiffusion_ox_sd_15_bs_bs1_precfp16 Avg Score : 0.380

## sdxl demo 示例：
(需要配置环境变量MACART_OP_KEEP_ONNX_PRECISION=ON 与CUDA_VISIBLE_DEVICES=0,1 现需要两张卡，单独吧unet放到一张卡上，测试环境ort版本为mxc500-onnxruntime-20240930)
```python 
MACART_OP_KEEP_ONNX_PRECISION=ON CUDA_VISIBLE_DEVICES=0,1 python code/sample_sdxl_onnx.py ./models/ox_sd_xl/ 50 1 maca 512 # 执行C500推理，step=50，出图尺寸为512*512，每个prompt生成1张图，使用0卡与1卡
```
参数
# modelpath：               需测试的模型路径, 内部包含config.json模型参数文件已配置好
# step:                     推理step
# num_images_per_prompt:    每个prompt生成图片数量
# EP:                       设置onnxruntime运行的EP, 默认为maca, 可选EP有cpu,maca
# size:                     生成图片尺寸 (size*size)
# device_id:                使用gpu编号(改用CUDA_VISIBLE_DEVICES指定  此参数无效)

输出示例(2.32.0.7)
average score: 0.356
Output 2 images, inference cost 34.618 seconds
StableDiffusion_ox_sd_xl_step50_images_per_prompt1 FPS : 0.058, latency : 17309.060ms, memory usage: 44.446 GB
StableDiffusion_ox_sd_xl_step50_images_per_prompt1 Avg Score : 0.356


## sd3 demo 示例：
(需要配置环境变量MACART_OP_KEEP_ONNX_PRECISION=ON，测试环境python3.8, diffusers==0.29.2)
```python
MACART_OP_KEEP_ONNX_PRECISION=ON python code/sample_sd3_onnx.py ./models/ox_sd_3/ 50 1 maca 512 0 # 执行C500推理，step=50，出图尺寸为512*512，每个prompt生成1张图，使用0卡
```
参数
# modelpath：               需测试的模型路径, 内部包含config.json模型参数文件已配置好
# step:                     推理step
# num_images_per_prompt:    每个prompt生成图片数量
# EP:                       设置onnxruntime运行的EP, 默认为maca, 可选EP有cpu,maca
# size:                     生成图片尺寸 (size*size)
# device_id:                使用gpu编号

输出示例(2.32.0.7)
average score: 0.390
Output 2 images, inference cost 29.569 seconds
StableDiffusion_ox_sd_3_step50_images_per_prompt1 FPS : 0.068, latency : 14784.265ms, memory usage: 43.989 GB
StableDiffusion_ox_sd_3_step50_images_per_prompt1 Avg Score : 0.390

## sd3.5 medium demo 示例：
```python
python code/sample_sd35_medium.py ./models/sd35_medium/ 50 1 512 0 # 执行PyTorch推理，step=50，出图尺寸为512*512，每个prompt生成1张图，使用0卡
```
参数
# modelpath:                需测试的模型路径, 内部包含config.json模型参数文件已配置好
# step:                     推理step
# images_per_prompt:        每个prompt生成图片数量
# output_size:              生成图片尺寸 (size*size)
# device_id:                使用gpu编号

输出示例
```
Output 2 images, inference cost 8.060 seconds
StableDiffusion_sd35_medium_step50_images_per_prompt1 FPS : 0.248, latency : 4029.977ms, memory usage: 24.563 GB
StableDiffusion_sd35_medium_step50_images_per_prompt1 Avg Score : 0.386
```