import os
import subprocess
import json
import sys
import tempfile
import signal

def ConvertWeightToPmx(config_str):
    config = json.loads(config_str).get('convert_to_pmx')
    model_type = json.loads(config_str).get('model_type')
    if config['enable_using_safetensors']:
        convert_cmd = 'python ./src/{}/ConvertWeightToPMX.py --input_dir {} --output_dir {} --use_safetensors True'.format(model_type, config['origin_model_dir'], config['pmx_model_output_dir'])
    else:
        convert_cmd = 'python ./src/{}/ConvertWeightToPMX.py --input_dir {} --output_dir {}'.format(model_type, config['origin_model_dir'], config['pmx_model_output_dir'])
    ret = subprocess.Popen(convert_cmd, shell=True, stdout=None, stderr=None, encoding='utf-8')
    ret.wait()

def SplitPmxModel(config_str):
    config = json.loads(config_str).get('split_pmx_model')
    model_type = json.loads(config_str).get('model_type')
    split_cmd = 'python ./src/{}/Split.py --input_dir {} --num_shards {} --output_dir {}'.format(model_type, config['pmx_model_dir'], config['number_of_shards'], config['split_model_output_dir'])
    ret = subprocess.Popen(split_cmd, shell=True, stdout=None, stderr=None, encoding='utf-8')
    ret.wait()

def MergePmxModel(config_str):
    config = json.loads(config_str).get('merge_pmx_model')
    model_type = json.loads(config_str).get('model_type')
    merge_cmd = 'python ./src/{}/Merge.py --input_dir {} --num_shards {} --output_dir {}'.format(model_type, config['split_model_dir'], config['num_of_shards'], config['merged_model_output_dir'])
    ret = subprocess.Popen(merge_cmd, shell=True, stdout=None, stderr=None, encoding='utf-8')
    ret.wait()

def PmxModelTest(config_str):
    config = json.loads(config_str).get('pmx_model_test')
    model_type = json.loads(config_str).get('model_type')
    if len(config['dump_steps'].split(",")) == 1 and config['dump_steps'][-1] != ",":
        config['dump_steps'] += ","
    test_cmd = 'OMP_NUM_THREADS=1 torchrun --nproc_per_node {} ./src/{}/Demo.py --ckpt_dir {} --tokenizer_path {}/tokenizer.model --fused_qkv 1 --fused_kvcache 1\
                --auto_causal 1 --quantized_cache 1 --dynamic_batching 1 --seqlen_scale_up {} --max_gen_len {} --dump_steps {} --dump_tensor_path {} --batch {}\
                --cache_layout {}'.format(config['num_gpu'], model_type, config['pmx_model_dir'], config['origin_model_tokenizer_dir'], config['seqlen_scale_up'],\
                config['max_gen_len'], config['dump_steps'], config['dump_tensor_path'], config['batch_size'], config['cache_layout'])
    ret = subprocess.Popen(test_cmd, shell=True, stdout=None, stderr=None, encoding='utf-8')
    ret.wait()

def ConvertPmxToOnnx(config_str):
    config = json.loads(config_str).get('convert_to_onnx')
    model_type = json.loads(config_str).get('model_type')
    convert_cmd = 'OMP_NUM_THREADS=1 torchrun --nproc_per_node {} ./src/{}/Export.py --ckpt_dir {} --fused_qkv 1\
                     --fused_kvcache 1 --auto_causal 1 --quantized_cache 1 --dynamic_batching 1 --export_path {} --cache_layout {}'.format(config['num_gpu'],\
                     model_type, config['pmx_model_dir'], config['onnx_model_output_dir'], config['cache_layout'])
    if model_type != 'chatglm':
        convert_cmd += ' --tokenizer_path {}/tokenizer.model'.format(config['origin_model_tokenizer_dir'])                     
    ret = subprocess.Popen(convert_cmd, shell=True, stdout=None, stderr=None, encoding='utf-8')
    ret.wait()

def OnnxModelAccuracyTest(config_str):
    config = json.loads(config_str).get('onnx_accuracy_test')
    step = config['step']
    num_gpu = config['num_gpu']
    if num_gpu > 1:
        test_cmd = '{}/ompi/bin/mpirun -np {} bash ./src/benchmark.sh {} {} {} {} {}'.format(config['maca_path'], num_gpu, step, config['onnx_model_dir'], config['out_put_dir'], config['test_data_dir'], config['ppl_serving_dir'])
    else:
        test_cmd = 'bash ./src/benchmark.sh {} {} {} {} {}'.format(step, config['onnx_model_dir'], config['out_put_dir'], config['test_data_dir'], config['ppl_serving_dir'])
    ret = subprocess.Popen(test_cmd, shell=True, stdout=None, stderr=None, encoding='utf-8')
    ret.wait()

def OnnxModelPerformanceTest(config_str):
    config = json.loads(config_str).get('onnx_performance_test')
    batch_size_list = config['batch_size_list'].split(",")
    input_token_list = config['input_token_list'].split(",")
    output_token_list = config['output_token_list'].split(",")
    model_log_dir = '{}/{}'.format(config['log_path'], config['model_name'])
    result_json = config['output_result_json_path']
    if result_json[-5:] != '.json':
        print('Please set correct json file for \'output_result_json_path\'')
        sys.exit()
    else:
        print('Result json file is : {}'.format(result_json))
    result_benchmark = []
    for batch_size in batch_size_list:
        for input_token in input_token_list:
            for output_token in output_token_list:
                if input_token == 2048 and output_token != 240:
                    continue
                if output_token == 240 and input_token != 2048:
                    continue
                if input_token == 1024 and output_token < 1024:
                    continue
                input_file = '{}/{}_{}'.format(config['input_file_dir'], config['input_file_base'], input_token)
                cur_log = '{}/input_{}_output_{}_batch_{}.log'.format(model_log_dir, input_token, output_token, batch_size)
                if config['do_tracer']:
                    llama_benchmark_cmd = 'mcTracer --name Tracers/Tracer_{}_bs{}_input{}_output{}_new_lib {}/benchmark_llama --model-type llama --model-dir {} --model-param-path {} \
                    --tensor-parallel-size {} --top-p {} --top-k {} --temperature {} --warmup-loops {} --generation-len {} \
                    --benchmark-loops {} --input-file {} --batch-size {} 2>&1'.format(
                        config['model_name'], batch_size, input_token, output_token, config['ppl_serving_dir'], config['onnx_model_dir'],
                        config['onnx_model_param_path'], config['tensor_parallel_size'], config['top_p'], config['top_k'], config['temperature'],
                        config['warmup_loops'], output_token, config['benchmark_loops'], input_file, batch_size
                    )
                    if config['enable_output_logs']:
                        llama_benchmark_cmd += ' | tee {}'.format(cur_log)
                else:
                    llama_benchmark_cmd = '{}/benchmark_llama --model-type llama --model-dir {} --model-param-path {} \
                    --tensor-parallel-size {} --top-p {} --top-k {} --temperature {} --warmup-loops {} --generation-len {} \
                    --benchmark-loops {} --input-file {} --batch-size {} 2>&1'.format(
                        config['ppl_serving_dir'], config['onnx_model_dir'], config['onnx_model_param_path'],
                        config['tensor_parallel_size'], config['top_p'], config['top_k'], config['temperature'],
                        config['warmup_loops'], output_token, config['benchmark_loops'], input_file, batch_size
                    )
                    if config['enable_output_logs']:
                        llama_benchmark_cmd += ' | tee {}'.format(cur_log)
                while(True):
                    ret = subprocess.run(llama_benchmark_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
                    print(ret.stdout)
                    result_metric = {}
                    if (ret.stdout.find('Minimum size of grid & block dimension is 1') > 0):
                        continue
                    if ret.stdout.find('failed: VersionRange') > 0:
                        continue
                    if ret.stdout.find('failed: out of memory') > 0:
                        result_metric['status'] = 'Failed'
                        result_metric['type'] = 'OOM'
                        break
                    start_pos = ret.stdout.find('CSV format output')
                   
                    result_metric['model'] = config['model_name']
                    result_metric['batch'] = batch_size
                    result_metric['input_token'] = input_token
                    result_metric['output_token'] = output_token
                    if start_pos < 0:
                        result_metric['status'] = 'Failed'
                    else:
                        # 'CSV format output:44.822,14.7243,14.9594,66.8476,14.3148'
                        # CSV format output:2962.29,55.1702,67.2832,475.602,39.4693
                        result_line = ret.stdout[start_pos:-1]
                        result_line = result_line.strip()
                        try:
                            result_infos = result_line.split(':')[1].split(',')
                            result_metric['prefill'] = float(result_infos[0])
                            result_metric['avg_decode_latency'] = float(result_infos[1])
                            result_metric['avg_step_latency'] = float(result_infos[2])
                            result_metric['tps'] = float(result_infos[3])
                            result_metric['memory'] = float(result_infos[4][:6])
                            result_metric['status'] = 'Passed'
                        except:
                            print(result_line)
                            break
                    break

                result_benchmark.append(result_metric)
                # exit()
        # save result per batch
        result_dir = os.path.dirname(result_json)
        if not os.path.exists(result_dir):  
            os.makedirs(result_dir)
        with open(result_json, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_benchmark, indent=4))

def StartServer(config_str, config_file):
    processes = []

    server_dir = json.loads(config_str).get('ppl_serving_dir')
    general_config = json.loads(config_str).get('server_config')
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=True, prefix='server_config_', dir=os.path.dirname(os.path.abspath(config_file))) as tmp_file:
        json.dump(general_config, tmp_file)
        tmp_file.seek(0)
        start_server_cmd = '{}/ppl_llm_server {}'.format(server_dir, tmp_file.name)
        ret = subprocess.Popen(start_server_cmd, shell=True, stdout=None, stderr=None, encoding='utf-8')
        
        processes.append(ret)

        if json.loads(config_str).get('enable_http_server'):
            start_server_cmd = 'python ./src/http/http_to_grpc.py --host {} --port {} --grpc_server {}:{}'.format(json.loads(config_str).get('http_server_config')['host'], json.loads(config_str).get('http_server_config')['port'], json.loads(config_str).get('server_config')['host'], json.loads(config_str).get('server_config')['port'])
            ret = subprocess.Popen(start_server_cmd, shell=True, stdout=None, stderr=None, encoding='utf-8')
            processes.append(ret)

        for process in processes:
            process.wait()


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print('use like this:\n python Start.py COMMAND config_file_path')
        sys.exit()

    model_cmd = sys.argv[1]
    with open(sys.argv[2], 'r') as f:
        config_str = f.read()
   
    if model_cmd != 'start_llm_server':
        model_type = json.loads(config_str).get('model_type')
        if model_type != 'llama' and model_type != 'chatglm' and model_type != 'internlm' and model_type != 'baichuan':
            print('unknown model type: %s, please use llama / chatglm / internlm / baichuan' %(model_type))
            sys.exit()

    if model_cmd == 'convert_to_pmx':
        ConvertWeightToPmx(config_str)
    elif model_cmd == 'split_pmx_model':
        SplitPmxModel(config_str)
    elif model_cmd == 'merge_pmx_model':
        MergePmxModel(config_str)
    elif model_cmd == 'pmx_model_test':
        PmxModelTest(config_str)
    elif model_cmd == 'convert_to_onnx':
        ConvertPmxToOnnx(config_str)
    elif model_cmd == 'onnx_accuracy_test':
        OnnxModelAccuracyTest(config_str)
    elif model_cmd == 'onnx_performance_test':
        OnnxModelPerformanceTest(config_str)
    elif model_cmd == 'start_llm_server':
        StartServer(config_str, sys.argv[2])
    else:
        print('Unknown command: %s' %(model_cmd))
        sys.exit()
