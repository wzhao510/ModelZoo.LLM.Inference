这一页说明如何推理FLUX模型，FLUX.1-dev。建议使用diffusers Docker环境，添加模型目录映射：-v /pde_ai:/pde_ai

## 运行示例(在diffusers目录下执行)
```
python code/sample_flux.py --model $modelname, --batchsize $batchsize --save-image 0

参数说明：
model：需测试的模型路径, 内部包含config.json模型参数文件已配置好, 目前仅支持: ./models/flux_1_0_dev
batchsize：目前单卡最大只支持16
save-iamge: 可选项，默认保存图片，如不保存设置为0
例：
python code/sample_flux.py --model ./models/flux_1_0_dev --batchsize 1 
```
