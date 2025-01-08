这一页说明如何推理FLUX模型，FLUX.1-dev。建议使用diffusers Docker环境，添加模型目录映射：-v /pde_ai:/pde_ai

## 运行示例(在diffusers目录下执行)
```
python code/sample_flux.py --model $modelname, --batchsize $batchsize --save-image 0

参数说明：
model：需测试的模型路径,可以是模型的绝对路径如：/models/Flux/FLUX.1-dev/  也可以是 FLUX.1-dev：
    a.当只输入模型名称时候将去 代码陌路 models/$modelname/config.json 中去读取模型的默认路劲。
    b./models/Flux/FLUX.1-dev/ 这种绝对路径，程序会在此路劲去加载模型权重，并在 models/FLUX.1-dev/config.json 去读取配置
    c。当前支持 FLUX.1-dev 和 FLUX.1-schnell
batchsize：目前单卡最大只支持16
save-iamge: 可选项，默认保存图片，如不保存设置为0
offload 是否使用cpu 加载默认 true，--offload false 则全部用显存加载模型
resolution  指定生成图像的分辨率，wxh：1024x1024 
例：
python code/sample_flux.py --model /models/Flux/FLUX.1-schnell/ --batchsize 1 --offload false --resolution 1024x1024
```
运行结果保存在当前目录 output/下，图像和csv 性能数据；
csv 文件命名方式为：模型名称_分辨率_bs{n}_output_时间.csv   如：FLUX.1-schnell_1024x1024_bs1_output_2025-01-06_02-24-10.csv
内容：
promt,bs,hw,e2e time
a tiny astronaut hatching from an egg on the moon,1,1024x1024,3273.81
最后1列是 e2e time,单位毫秒；



2025.1.2 修改：
flux 模型下config.json 字段修改：
    "outputs_size":["1024#1024","2048#2048"] #改为数组可输入多个分辨率
    "max_sequence_length":512,#增加字段
    "guidance_scale":3.5  #新增加字段

2025.1.6
   修改只记录第二次的数据，第一次运行的数据作为warmup，将不进行性能比较
   将生成图像分辨率修改为从参数输入，之前从配置文件里面读取。