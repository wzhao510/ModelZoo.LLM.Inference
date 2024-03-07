import struct
import numpy as np
import os
import sys
import glob


def numpy_load(file_name):
    '''
    try:
        value = np.load(file_name)
    except ValueError:
        value = np.load(file_name, allow_pickle=True)
    '''
    value = np.fromfile(file_name, np.float32)
    return value

def get_cosine(gpu_array, cpu_array):
    from math import sqrt
    gpu_array = gpu_array.astype('float64')
    cpu_array = cpu_array.astype('float64')
    x = np.square(gpu_array)
    #print(x)
    x = np.sum(x) 
    #print(x)
    x = sqrt(x)
    #print(x)

    y = np.square(cpu_array)
    #print(y)
    y = np.sum(y)
    #print(y)
    y = sqrt(y)
    #print(y)

    z = gpu_array * cpu_array
    #print(z)
    z = np.sum(z)
    #print(z)

    cosine = ((z)) / ((x * y) + 1e-12) # eps
    if cosine > 1.0:
        cosine = 1.0

    return cosine
#'''
def get_mse(diff_array):
    x = np.square(diff_array)
    mse = np.mean(x)
    return mse  

def get_snr(diff_array, cpu_array):
    x = np.square(diff_array)
    x = np.sum(x)

    y = np.square(cpu_array)
    y = np.sum(y) 

    snr = (x) / (y + 1e-7)

    snr = np.mean(snr)

    return snr


fail_str = """
_____            _   _              _ 
|  ___|   __ _  (_) | |   ___    __| |
| |_     / _` | | | | |  / _ \  / _` |
|  _|   | (_| | | | | | |  __/ | (_| |
|_|      \__,_| |_| |_|  \___|  \__,_|
                                    
"""

succ_str = """

                                                                    _ 
 _ __ ___     __ _    ___    __ _     ___   _   _    ___    ___    | |
| '_ ` _ \   / _` |  / __|  / _` |   / __| | | | |  / __|  / __|   | |
| | | | | | | (_| | | (__  | (_| |   \__ \ | |_| | | (__  | (__    |_|
|_| |_| |_|  \__,_|  \___|  \__,_|   |___/  \__,_|  \___|  \___|   (_)
                                                                    
                        
"""

if __name__ == "__main__":
    STEP = sys.argv[1]
    OUT_DIR = sys.argv[2]
    GLODEN_DIR = sys.argv[3]
    # STEP = int(STEP)+1
    # golden_path=f"/home/yjchen/maca_PPL_LLM/ppl.pmx-master/model_zoo/llama/facebook/data/gt_dump/rank_0/step{STEP}_logits-4_32000-fp32.bin"

    filename_pattern = f'{GLODEN_DIR}/step{STEP}_logits-*.bin'
    file_list = glob.glob(filename_pattern)

    golden_path = file_list[0]
    ouput_path = os.path.join(OUT_DIR, "pplnn_output-logits.dat")
    print(f"\ngolden_path: ---> {golden_path}")
    print(f"ouput_path --> {ouput_path}")
    print("...........Start comapring ................")

    # data_gt = np.fromfile(golden_path, dtype=np.float32)
    # data_pt = np.fromfile(ouput_path, dtype=np.float32)

    data_gt = numpy_load(golden_path)
    data_pt = numpy_load(ouput_path)
    print(f'---------------- compare info:   ------------------')
    #print(f'get_cosine: {get_cosine(data_gt, data_pt)}')
    cos_value = get_cosine(data_gt, data_pt)
    print("get_cosine: %3.10f" % (cos_value))
    diff = data_gt - data_pt
    #print(f'get_mse: {get_mse(diff)}')
    print("get_mse: %3.10f" % (get_mse(diff)))
    #print(f'get_snr: {get_snr(diff, data_gt)}')
    print("get_snr: %3.10f" % (get_snr(diff, data_gt)))
    print(f'diff max: {np.max(diff)} @gpu: {data_gt.reshape(-1)[np.argmax(diff.reshape(-1))]}')
    print("."*50)


    if cos_value > 0.9999:
        print(succ_str)
    else:
        print(fail_str)
    print("Note: we use COS_VALUE > 0.9999 to judge. ")
