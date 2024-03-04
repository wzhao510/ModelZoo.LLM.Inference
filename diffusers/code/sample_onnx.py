import sys
import numpy as np
from utils.utils import get_params
from common import Text2ImgModel
import time

# generator = np.random
def main(modelname,batchsize,precision, task="normal", modelfile="./",EP="maca", th_num="8"):
    params = get_params(modelname)
    model = Text2ImgModel(params)
    prompt = "a photo of an astronaut riding a horse on mars"
    print("Warm up")
    image = model.infer(prompt, params)
    print("Start Inference")
    start = time.time()
    image = model.infer(prompt, params)
    print(f"Cost time: {time.time() - start}")
    image.save(f"generated_image.png")



if __name__ == '__main__':
    modelname = sys.argv[1]
    batchsize = sys.argv[2]
    precision = sys.argv[3]

    task = sys.argv[4]      
    model_path = sys.argv[5] if len(sys.argv) > 5 else "./"
    EP = sys.argv[6] if len(sys.argv) > 6 else "maca"
    th_num = sys.argv[7] if len(sys.argv) > 7 else "16"
    main(modelname,batchsize,precision,task,model_path, EP, th_num) 
