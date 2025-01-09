这一页说明如何推理SD模型，包括sd1.5和sd2.1。建议使用onnxruntime Docker环境，添加模型目录映射：-v /pde_ai/models/llm/StableDiffusion/stable-diffusion-onnx:/pde_ai/models/llm/StableDiffusion/stable-diffusion-onnx

## requirement
需要安装运行依赖包: onnxsim=0.4.36 diffusers==0.19.3 onnx==1.12.0 Pillow==10.0.0 transformers==4.31.0

执行:
    
    pip install -r requirement.txt

## 目录结构及说明
```
.
|-- code
|   |-- common.py
|   |-- sample_onnx.py
|   |-- sample_onnx_multithreads.py
|   |-- sd_model
|   `-- utils
|-- data
|   `-- PartiPrompts.tsv
|-- models
|   |-- ox_sd_15
|   |-- ox_sd_15_bs
|   |-- ox_sd_21
|   |-- ox_sd_21_base
|   |-- ox_sd_21_base_bs
|   `-- ox_sd_21_bs
|-- requirement.txt
`-- runtest.sh
```
code包括示例代码；data下存放Parti数据集；models下存放sd模型配置文件，其中以"_bs"结尾的模型是支持动态batch和动态shape，否则不支持；requirment.txt中包含运行依赖；runtest.sh为运行脚本。

## 运行示例(在diffusers目录下执行)
```
runtest.sh $modelname, $batchsize, $precision, $task, $model_path, $EP, $th_num, $device_id
或
python code/sample_onnx.py $modelname, $batchsize, $precision, $task, $model_path, $EP, $th_num, $device_id

参数说明：
$modelpath：需测试的模型路径, 内部包含config.json模型参数文件已配置好, 可选模型包括: ./models/ox_sd_15/, ./models/ox_sd_15_bs/, ./models/ox_sd_21/, ./models/ox_sd_21_base/, ./models/ox_sd_21_base_bs/, ./models/ox_sd_21_bs/；
$precision: 推理精度包含, fp32：使用ort-fp32精度, fp16：使用ort-fp16精度(暂未支持), int8：暂未支持, qdq：暂未支持
$task:      测试任务，当前暂未用上，但不能省略
$test_data: test 表示读取test数据， dev 表示读取 dev数据。见 data
$EP:        设置onnxruntime运行的EP, 默认为maca, 可选EP有cpu,gpu,trt, 需根据环境中存在的EP进行设置
$num_thr:   测算性能时，开启的线程数量 默认8
$device_id: 设备id，默认为0
```
```
单batch测试：
./runtest.sh ./models/ox_sd_15/ 1 fp16 norm test maca
./runtest.sh ./models/ox_sd_21/ 1 fp16 norm test maca
./runtest.sh ./models/ox_sd_21_base/ 1 fp16 norm test maca
```
```
python code/sample_onnx.py ./models/ox_sd_15/ 1 fp16 norm test maca  # 执行fp16 batchsize=1 C500推理. Demo 测试跑2条数据
python code/sample_onnx.py ./models/ox_sd_15_bs/ 2 fp16 norm test maca  # 执行fp16 batchsize=2 C500推理. Demo 测试跑2条数据 (动态shape和动态batch支持，默认出图照片为 960x960)
```
如果要修改输出图像尺寸，修改config.json中outputs_size字段，例如当前输出尺寸是960x960：
```
{
    "//": "以下配置的模型名字为FP32路径模型名字, FP16名字将会在转换统一命名 model_sim_fp16.onnx",
    "model_config":{
        "ori_path":"/pde_ai/models/llm/StableDiffusion/stable-diffusion-onnx/sd-2-1_fp32_bs",
        "fp16_path":"/pde_ai/models/llm/StableDiffusion/stable-diffusion-onnx/sd-2-1_fp16_bs",
        "dataset":"./data/PartiPrompts.tsv",
        "seed": 21,
        "num_inference_steps": 50,
        "text_encoder":"model_sim.onnx",
        "unet": "model_fix_sim.onnx",
        "vae_decoder":"model_sim.onnx",
        "vae_encoder":"model.onnx",
        "outputs_size":"960#960",
        "scheduler":"DDIM"
    }
}
```
## 数据集测试(在diffusers目录下执行)
```python
python code/sample_onnx_multithreads.py ./models/ox_sd_15/ 1 fp16 norm test maca 8 0  # 执行fp16 C500推理. 使用0号卡，以batch=1，8个线程运行。模型不带_bs 只支持 bs=1 运行
python code/sample_onnx_multithreads.py ./models/ox_sd_15_bs/ 2 fp16 norm test maca 8 1  # 执行fp16 C500推理. 使用1号卡，以batch=2，8个线程运行
```

## Flux
```python
python code/sample_flux.py --model /models/Flux/FLUX.1-schnell/ --batchsize 1 --offload false --resolution 1024x1024
```

参数说明：
model：需测试的模型路径,可以输入模型的绝对路径或模型名称；
    a.当输入是模型名称时，代码中将从 models/$modelname/config.json 文件中去读取模型的路劲。
    b.当输入是模型绝对路径时，如：/models/Flux/FLUX.1-dev/  程序将 “FLUX.1-dev”作为模型名称，并在models/FLUX.1-dev/config.json 读取模型配置信息。
    c.当前支持 FLUX.1-dev 和 FLUX.1-schnell
batchsize：目前单卡最大只支持16
save-iamge: 可选项，默认保存图片，如不保存设置为 0
offload 是否使用cpu 加载;默认 true，--offload false 则全部用显存加载模型
resolution  指定生成图像的分辨率，wxh：1024x1024 

demo 输出：
输出csv和图像；
csv：
csv 文件命名方式为：模型名称_分辨率_bs{n}_output_时间.csv   如：FLUX.1-schnell_1024x1024_bs1_output_2025-01-06_02-24-10.csv
内容：
promt,bs,hw,e2e time
a tiny astronaut hatching from an egg on the moon,1,1024x1024,3273.81
最后1列是 e2e time,单位毫秒；

## OmniGen

安装官方OmniGen依赖包
```python
git clone https://github.com/VectorSpaceLab/OmniGen.git
cd OmniGen
#注意:将 requirement.txt 中的 torch 注释掉，使用maca torch
pip install -e .
```

参数说明
--model 指定模型路径，一般在 /pde_ai/models/llm/OmniGen-v1 
--batchsize 指定 batch 数量，此模型默认效果好的情况下 steps=50 ，时间比较长，建议不用测试多batch 性能
--infertype 推理方式，T2I:文生图，  I2I：图生图，当前demo 是将2张图像中的 两个人合在一起生成另外一张图像的。
--resolution 生成图像分辨率 默认值 1024x1024
--offload true 使用cpu 加载权重，false 使用GPU 加载权重； 默认值：false
--save_image 保存图像，可选值：true 、false；默认值 true

运行文生图
python  code/sample_omnigen.py --model /models/OmniGen-v1/ --batchsize 1 --infertype T2I

运行图生图
#图生图 --batchsize 只能是1
python  code/sample_omnigen.py --model /models/OmniGen-v1/ --batchsize 1 --infertype I2I

输出：
目录：output/omnigen/
csv: {infertype}_{resolution}_bs{batchsize}_output_{current_time}.csv,内容如：
promt,bs,hw,e2e time
"Realistic photo. A young woman sits on a sofa, holding a book and facing the camera. She wears delicate silver hoop earrings adorned with tiny, sparkling diamonds that catch the light,         with her long chestnut hair cascading over her shoulders. Her eyes are focused and gentle, framed by long, dark lashes. She is dressed in a cozy cream sweater, which complements her warm, inviting smile.        Behind her, there is a table with a cup of water in a sleek, minimalist blue mug. The background is a serene indoor setting with soft natural light filtering through a window, adorned with tasteful art and flowers, creating a cozy and peaceful ambiance. 4K, HD.",1,1024x1024,43661.36

e2e time 时间单位为毫秒

图像：{infertype}{infertype}_bs{batchsize}_image_{i}_{w}x{h}.png