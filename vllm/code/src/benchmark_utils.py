from typing import List, Tuple
import itertools
import argparse


MAINSTREAM_MODEL=["DeepSeek","Qwen3","Glm4.5"]

class VllmParamLibrary(object):
    def _common_params():
        batches = (1,8,16,32,64)
        input_lengths = (256, 512, 1024)
        output_lengths = (128, 512, 1024)
        return list(itertools.product(batches,input_lengths,output_lengths))
    
    def _mainstream_params():
        batches = (1,8,16,32,64,128,256,512,1024)                
        input_lengths = (128, 2048, 3072)
        output_lengths= (128, 2048, 1024)
        return list(itertools.product(batches,input_lengths,output_lengths))

    def _highlight_params():
        batches = (1,16,32,128,512)
        input_lengths = (128, 2048, 3072)
        output_lengths = (128, 2048, 1024)
        return list(itertools.product(batches,input_lengths,output_lengths))

    PARAM_DICT = {
        "highlight":_highlight_params,
        "mainstream":_mainstream_params,
        "common": _common_params,
    }
    
    def params_filter(self, params: List[Tuple[int,int,int]], is_mainstreams_model:bool) ->List[Tuple[int,int,int]]:
        def condition1(batch, input_len, output_len):
            return input_len == 128 and output_len != 128
        
        def condition2(batch, input_len, output_len):
            return input_len == 2048 and output_len != 2048
        
        def condition3(batch, input_len, output_len):
            return input_len == 3072 and output_len != 1024
        
        def condition4(batch, input_len, output_len):
            return (input_len == 3072 or input_len == 2048) and batch > 64
        
        def condition5(batch, input_len, output_len):
            return input_len == 1024 and output_len != 1024
        
        mainstream_contidions = [condition1, condition2, condition3, condition4]
        rest_conditions = [condition5]

        filtered_params = []
        if is_mainstreams_model:
            for (batch, input_len, output_len) in params:
                if any(condition(batch, input_len, output_len) for condition in mainstream_contidions):
                    continue
                else:
                    filtered_params.append((batch, input_len, output_len))
        else:
            for (batch, input_len, output_len) in params:
                if any(condition(batch, input_len, output_len) for condition in rest_conditions):
                    continue
                else:
                    filtered_params.append((batch, input_len, output_len))
        return filtered_params


    def generates(self, model:str, is_highlight:bool) -> List[Tuple]:
        if any(m in model for m in MAINSTREAM_MODEL):
            is_mainstream_model = True
        else: 
            is_mainstream_model = False
        print(f"is_Mainstream_model: {is_mainstream_model}\n")
        print(f"is_highlight: {is_highlight}\n")
        if is_mainstream_model and is_highlight:
            params = self.PARAM_DICT['highlight']()
        elif is_mainstream_model:
            params = self.PARAM_DICT['mainstream']()
        else:
            params = self.PARAM_DICT['common']()
        # print(f"params are:\n {params}")
        filterd_params = self.params_filter(params, is_mainstream_model)
        print(f"final test set is {filterd_params}")
        return filterd_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        required=True,
                        help="model name"
                        )
    parser.add_argument("--highlight",
                        action='store_true',
                        help="using highlight params or not."
                        )
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="output params file path.")
    args = parser.parse_args()
    vpl = VllmParamLibrary()
    params_set = vpl.generates(args.model, args.highlight)
    print(params_set)
    with open(args.output, 'w') as o:
        o.write(repr(params_set))