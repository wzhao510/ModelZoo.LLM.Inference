import json
import onnx
from onnxsim import simplify
import os
from onnx import helper

Jsonfile = "./%s/config.json"

def get_params(modelname):
    with open(Jsonfile%(modelname), 'r') as f:
        param_all = json.load(f)
        params = param_all["model_config"]
    return params

def get_input_shape_info(model_name, input_size, batch_size, encoder_hidden_dim, is_dynamic_bs=False):
    if is_dynamic_bs:
        if model_name == "text_encoder":
            input_shape_dict = {"input_ids":[-1,77]}
        elif model_name == "unet":
            input_shape_dict = {"sample":[-1,4,int(input_size[0]/8),int(input_size[1]/8)],
                                "timestep":[1],
                                "encoder_hidden_states":[-1,77,encoder_hidden_dim]}
        elif model_name == "vae_decoder":
            input_shape_dict = {"latent_sample":[-1,4,int(input_size[0]/8),int(input_size[1]/8)]}
        else:
            raise ValueError("unkown model name %s"%model_name)
    else:
        if model_name == "text_encoder":
            input_shape_dict = {"input_ids":[batch_size,77]}
        elif model_name == "unet":
            input_shape_dict = {"sample":[batch_size*2,4,int(input_size[0]/8),int(input_size[1]/8)],
                            "timestep":[1],
                            "encoder_hidden_states":[batch_size*2,77,encoder_hidden_dim]}
           
        elif model_name == "vae_decoder":
            input_shape_dict = {"latent_sample":[1,4,int(input_size[0]/8),int(input_size[1]/8)]}
        else:
            raise ValueError("unkown model name %s"%model_name)
    return input_shape_dict


def remove_value_info(model):
    # 创建一个新的 graph，将原模型中的所有节点、输入、输出和 initializer 复制过来，但不包括 value_info
    new_graph = helper.make_graph(
        nodes=model.graph.node,
        name=model.graph.name,
        inputs=model.graph.input,
        outputs=model.graph.output,
        initializer=model.graph.initializer,
        sparse_initializer=model.graph.sparse_initializer,
        doc_string=model.graph.doc_string
    )
    
    # 将原模型的 opset_import 复制过来
    new_model = helper.make_model(
        new_graph, 
        producer_name=model.producer_name,
        opset_imports=model.opset_import,
        # metadata_props=model.metadata_props
    )

    # 保留原模型的 ir_version、producer_version、domain 和 model_version 等信息
    new_model.ir_version = model.ir_version
    new_model.producer_version = model.producer_version
    new_model.domain = model.domain
    new_model.model_version = model.model_version

    return new_model

def fixShape(input_model_path, output_model_path="/home/dhe/temp.onnx", input_shape_dict={}):
    model = onnx.load(input_model_path)
    # del model.graph.value_info[:]
    model = remove_value_info(model)
    model_simplified, check = simplify(model, overwrite_input_shapes=input_shape_dict,perform_optimization=False, skip_constant_folding=True, skip_fuse_bn=True, skip_shape_inference=False)

    # 验证简化后的模型是否有效
    assert check, "Simplified ONNX model could not be validated"

    # 删除已有的model_weights.data
    model_dir = os.path.split(output_model_path)[0]
    os.system(f'rm -f {model_dir}/model_weights.data')
    # 保存简化后的模型
    onnx.save(model_simplified, output_model_path, save_as_external_data=True, all_tensors_to_one_file=True, location="model_weights.data")

def macaOptFp16(input_model_path, output_model_path, dynamic_batch):
    if dynamic_batch:
        print('\n\n ################### dynamic_batch true ################# \n\n')
        os.system("python -m maca_converter --model_path %s --output %s --model_type onnx --fp32_to_fp16 1 --simplify 2 --dynamic_batch 1"%(input_model_path,output_model_path))
    else:        
        print('\n\n ################### dynamic_batch false ################# \n\n')
        os.system("python -m maca_converter --model_path %s --output %s --model_type onnx --fp32_to_fp16 1 --simplify 2 --dynamic_batch 0"%(input_model_path,output_model_path))

def convertModel(input_model_path,output_model_path, input_shape_dict={}, dynamic_batch=False):
    temp_path = output_model_path+"_bak"
    
    fixShape(input_model_path,output_model_path=temp_path, input_shape_dict=input_shape_dict)
    macaOptFp16(temp_path,output_model_path, dynamic_batch)
    # os.remove(temp_path)