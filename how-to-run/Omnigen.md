简介
本页将说明如何运行OmniGen模型，推荐torch 2.4.0，torch 2.1 版本也可以跑。

安装官方OmniGen 的包
cd  OmniGen
pip install -e .

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
#图生图 --batchsize 只能是1，其他值无效
python  code/sample_omnigen.py --model /models/OmniGen-v1/ --batchsize 1 --infertype I2I

输出：
目录：output/omnigen/
csv: {infertype}_{resolution}_bs{batchsize}_output_{current_time}.csv,内容如：
promt,bs,hw,e2e time
"Realistic photo. A young woman sits on a sofa, holding a book and facing the camera. She wears delicate silver hoop earrings adorned with tiny, sparkling diamonds that catch the light,         with her long chestnut hair cascading over her shoulders. Her eyes are focused and gentle, framed by long, dark lashes. She is dressed in a cozy cream sweater, which complements her warm, inviting smile.        Behind her, there is a table with a cup of water in a sleek, minimalist blue mug. The background is a serene indoor setting with soft natural light filtering through a window, adorned with tasteful art and flowers, creating a cozy and peaceful ambiance. 4K, HD.",1,1024x1024,43661.36

e2e time 时间单位为毫秒

图像：bs{args.batchsize}_image_{i}_{w}x{h}.png