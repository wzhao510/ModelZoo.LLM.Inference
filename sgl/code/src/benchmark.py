from typing import Optional, List, Dict, Any, Union
import re


from utils.utils import *



class Benchmark:
    def __init__(self, name, client_id, launch_mode) -> None:
        self.client_id = client_id
        self.cmd_list = []
        self.name = name
        self.launch_mode = launch_mode
        self.type = BenchmarkType.perf.value
        self.envs = []

    @classmethod
    def parse_config(cls, name, config, client_id, common_envs, launch_mode = TaskLaunchMode.online):
        if 'type' not in config.keys() or config['type'] == BenchmarkType.perf.value:
            print(name)
            return PerfBenchmark(name, config, client_id, common_envs, launch_mode)
        
        if config['type'] in [BenchmarkType.mmlu.value, BenchmarkType.ceval.value]:
            return AccBenchmark(name, config, client_id, common_envs, launch_mode)
        assert (False, f"benchmark type {config['type']} not support !")

    def get_input_output_str(self, command):
        data_set_match = re.search(r"--dataset-name\s+(\S+)", command)
        if not data_set_match:
            return "--random-input-len", "--random-output-len"

        return f"--{data_set_match.group(1).strip()}-input-len", f"--{data_set_match.group(1).strip()}-output-len"
    
    def get_benchmark_args(self, command):
        input_str, output_str = self.get_input_output_str(command)
        input_len_match = re.search(rf"{input_str}\s+(\d+)", command)
        output_len_match = re.search(rf"{output_str}\s+(\d+)", command)
        num_prompt_match = re.search(r"--num-prompts?\s+(\d+)", command)
        max_con_match = re.search(r"--max-concurrency?\s+(\d+)", command)

        input_len = input_len_match.group(1) if input_len_match else "0"
        output_len = output_len_match.group(1) if output_len_match else "0"
        num_prompt = num_prompt_match.group(1) if num_prompt_match else "0"
        max_con = max_con_match.group(1) if max_con_match else "0"

        return input_len, output_len, num_prompt, max_con
    
    def get_benchmark_args_str(self, command):
        input_len, output_len, num_prompt ,max_con = self.get_benchmark_args(command)
        bench_args_str = f"in{input_len}_out{output_len}_prompt{num_prompt}"
        if max_con != '0':
            bench_args_str += f'_maxcon{max_con}'
        return f'{self.type}_{self.name}_{bench_args_str}'

    def get_end_id(self):
        return self.client_id + len(self.cmd_list)
    
    def extract_metrics_from_file(self, file_path):
        pass

    def get_output_str(self, command):
        pass

    def get_result_ext(self):
        return '.txt'
    
    def get_csv_result_str(self):
        return 'perf'
    
    def merge_envs(self, common_envs, benchmark_envs):
        if len(benchmark_envs) == 0:
            return
        new_env = []
        for env_value in benchmark_envs:
            if env_value in common_envs.keys():
                new_env.extend(common_envs[env_value])
                continue
            new_env.append(env_value)
        return convert_str_to_env_dict(new_env)
    

class PerfBenchmark(Benchmark):
    def __init__(self, name, config, client_id, common_envs, launch_mode = TaskLaunchMode.online) -> None:
        super().__init__(name, client_id, launch_mode)
        if config is not None:
            self.envs = self.merge_envs(common_envs, get_json_config_default(config, 'environment', []))
            self._init_benchmark(config)

    def _init_benchmark(self, config) -> None:
        command_base = get_json_config_default(config, 'command_base', '')
        max_concurrency = get_json_config_default(config, 'max_concurrency', None)
        num_prompt_times = get_json_config_default(config, 'num_prompt_times', None)
        num_prompts = get_json_config_default(config, 'num_prompt', None)
        input_str, output_str = self.get_input_output_str(command_base)
        assert (
            max_concurrency is None or isinstance(max_concurrency, list), f'invalid max_concurrency, must be list'
        )
        for input_output in config['input_output_len']:
            input_len, output_len = input_output.split('/')
            if input_len == '':
                command_base_str = f" {command_base} {output_str} {output_len} "
            else:
                command_base_str = f" {command_base} {input_str} {input_len} {output_str} {output_len} "
            if num_prompt_times is None:
                assert (
                    num_prompts is not None and isinstance(num_prompts, list), f'when num_prompt_times not set, num_prompt must be set and be list!'
                )
                
                for num_prompt in num_prompts:
                    if max_concurrency is None:
                        self.cmd_list.append(f"{command_base_str} --num-prompts {num_prompt}")
                        continue
                    for concurrency in max_concurrency:
                        self.cmd_list.append(f"{command_base_str} --num-prompts {num_prompt} --max-concurrency {concurrency}")

                continue

            for index, concurrency in enumerate(max_concurrency):
                cur_num_prompt = None
                if isinstance(num_prompt_times, list):
                    if index < len(num_prompt_times):
                        cur_num_prompt = int(num_prompt_times[index])
                    else:
                        cur_num_prompt = 1
                else:
                    cur_num_prompt = int(num_prompt_times)  

                if cur_num_prompt is None or cur_num_prompt <= 0:
                    self.cmd_list.append(f"{command_base_str} --num-prompts {concurrency}")
                    continue
                cur_num_prompt = cur_num_prompt * int(concurrency)
                self.cmd_list.append(f"{command_base_str} --num-prompts {cur_num_prompt} --max-concurrency {concurrency}")
            

        if len(config['input_output_len']) == 0:
            self.cmd_list.append(f" {command_base}")

    def extract_metrics_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        command_match = re.search(r'^Command:\s*(.+)$', content, re.MULTILINE)
        bench_args = {
                        'benchmark-name':[self.name],
                        'in-out':[],
                        'num_prompt':[],
                    }
        if command_match:
            command = command_match.group(1)
        else:
            command = ' '
        input_len,output_len,num_prompt,_ = self.get_benchmark_args(command)
        bench_args['in-out'].append(f'{input_len}-{output_len}')
        bench_args['num_prompt'].append(f'{num_prompt}')

        extract_field = lambda pattern, text: (
            match.group(1).strip()
            if (match := re.search(pattern, text, re.IGNORECASE)) 
            else None
        )
        patterns = {
            'Infer Backend': r'Backend:\s+(.+)',
            'Traffic request rate': r'Traffic request rate:\s+(.+)',
            'Successful requests': r'Successful requests:\s+(\d+)',
            'Benchmark duration (s)': r'Benchmark duration \(s\):\s+(\d+\.\d+|\d+)',
            'Total input tokens': r'Total input tokens:\s+(\d+)',
            'Total generated tokens': r'Total generated tokens:\s+(\d+)',
            'Total generated tokens (retokenized)': r'Total generated tokens \(retokenized\):\s+(\d+)',
            'Request throughput (req/s)': r'Request throughput \(req/s\):\s+(\d+\.\d+|\d+)',
            'Input token throughput (tok/s)': r'Input token throughput \(tok/s\):\s+(\d+\.\d+|\d+)',
            'Output token throughput (tok/s)': r'Output token throughput \(tok/s\):\s+(\d+\.\d+|\d+)',
            'Total token throughput (tok/s)': r'Total token throughput \(tok/s\):\s+(\d+\.\d+|\d+)',
            'Concurrency': r'Concurrency:\s+(\d+\.\d+|\d+)',
            'Accept length': r'Accept length:\s+(\d+\.\d+|\d+)',
            'Mean E2E Latency (ms)': r'Mean E2E Latency \(ms\):\s+(\d+\.\d+|\d+)',
            'Median E2E Latency (ms)': r'Median E2E Latency \(ms\):\s+(\d+\.\d+|\d+)',
            'Mean TTFT (ms)': r'Mean TTFT \(ms\):\s+(\d+\.\d+|\d+)',
            'Median TTFT (ms)': r'Median TTFT \(ms\):\s+(\d+\.\d+|\d+)',
            'P50 TTFT (ms)': r'P50 TTFT \(ms\):\s+(\d+\.\d+|\d+)',
            'P90 TTFT (ms)': r'P90 TTFT \(ms\):\s+(\d+\.\d+|\d+)',
            'P95 TTFT (ms)': r'P95 TTFT \(ms\):\s+(\d+\.\d+|\d+)',
            'P99 TTFT (ms)': r'P99 TTFT \(ms\):\s+(\d+\.\d+|\d+)',
            'P100 TTFT (ms)': r'P100 TTFT \(ms\):\s+(\d+\.\d+|\d+)',
            'Mean TPOT (ms)': r'Mean TPOT \(ms\):\s+(\d+\.\d+|\d+)',
            'Median TPOT (ms)': r'Median TPOT \(ms\):\s+(\d+\.\d+|\d+)',
            'P50 TPOT (ms)': r'P50 TPOT \(ms\):\s+(\d+\.\d+|\d+)',
            'P90 TPOT (ms)': r'P90 TPOT \(ms\):\s+(\d+\.\d+|\d+)',
            'P95 TPOT (ms)': r'P95 TPOT \(ms\):\s+(\d+\.\d+|\d+)',
            'P99 TPOT (ms)': r'P99 TPOT \(ms\):\s+(\d+\.\d+|\d+)',
            'P100 TPOT (ms)': r'P100 TPOT \(ms\):\s+(\d+\.\d+|\d+)',
            'Mean ITL (ms)': r'Mean ITL \(ms\):\s+(\d+\.\d+|\d+)',
            'Median ITL (ms)': r'Median ITL \(ms\):\s+(\d+\.\d+|\d+)',
            'P50 ITL (ms)': r'P50 ITL \(ms\):\s+(\d+\.\d+|\d+)',
            'P90 ITL (ms)': r'P90 ITL \(ms\):\s+(\d+\.\d+|\d+)',
            'P95 ITL (ms)': r'P95 ITL \(ms\):\s+(\d+\.\d+|\d+)',
            'P99 ITL (ms)': r'P99 ITL \(ms\):\s+(\d+\.\d+|\d+)',
            'P100 ITL (ms)': r'P100 ITL \(ms\):\s+(\d+\.\d+|\d+)',
            'Max ITL (ms)': r'Max ITL \(ms\):\s+(\d+\.\d+|\d+)',
        }

        metrics = {
            key: [value]
            for key, pattern in patterns.items()
            if (value := extract_field(pattern, content)) is not None
        }

        bench_result_data = {**bench_args, **metrics}

        return bench_result_data
    
    def get_output_str(self, command):
        if self.launch_mode == TaskLaunchMode.offline:
            return "", "--result-filename"
        if 'sglang.bench_serving' in command:
            return "", "--output-file"
        else:
            return "", ""


class AccBenchmark(Benchmark):
    def __init__(self, name, config, client_id, common_envs, launch_mode = TaskLaunchMode.online) -> None:
        super().__init__(name, client_id, launch_mode)
        self.envs = self.merge_envs(common_envs, get_json_config_default(config, 'environment', []))
        self._init_benchmark(config)

    def _init_benchmark(self, config) -> None:
        command_base = get_json_config_default(config, 'command_base', '')
        self.type = config['type']
        if self.type == BenchmarkType.mmlu.value:
            self.cmd_list.append(command_base)
            return
        elif self.type == BenchmarkType.ceval.value:
            if 'random' in config:
                for random in config['random']:
                    self.cmd_list.append(f"{command_base} {random}")
            else:
                self.cmd_list.append(f"{command_base}")

    def _get_acc_mmlu_metrics(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        try:
            data = json.loads(content)
            accuracy = data.get("accuracy")
            nsub = data.get("other").get("nsub")
            if accuracy is not None:
                accuracy = float(accuracy)
        except json.JSONDecodeError:
            accuracy = None
            nsub = None
        metrics = {'batch_size':None, 'random_seed':None, 'random_num':None, 'dataset':'mmlu', 'nsub': nsub, 'Accuracy': [accuracy],}
        return metrics

    def _get_acc_ceval_metrics(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        matches = re.findall(r'Accuracy\s*:\s+(\d+\.\d+|\d+)', content)
        last_accuracy = matches[-1] if matches else None
        pattern = r'bs(\d+)_seed(\d+)_num(\d+)'
        match = re.search(pattern, file_path)
        bs = int(match.group(1)) if match else None
        seed = int(match.group(2)) if match else None
        num = int(match.group(3)) if match else None

        metrics = {'batch_size':bs, 'random_seed':seed, 'random_num':num, 'dataset':'ceval', 'nsub': None, 'Accuracy': [float(last_accuracy) if last_accuracy else None]}
        return metrics

    def extract_metrics_from_file(self, file_path):
        metrics = None
        if self.type == BenchmarkType.mmlu.value:
            metrics = self._get_acc_mmlu_metrics(file_path)
        elif self.type == BenchmarkType.ceval.value:
            if 'seed' in file_path and 'num' in file_path:
                metrics = self._get_acc_ceval_metrics(file_path)
        return metrics
    
    def get_benchmark_args_str(self, command):
        if self.type == BenchmarkType.ceval.value:
            bs_str = re.search(r"--batch_size?\s+(\d+)", command)
            seed_str = re.search(r"--random_seed?\s+(\d+)", command)
            num_str = re.search(r"--random_num?\s+(\d+)", command)

            bs = bs_str.group(1) if bs_str else "0"
            seed = seed_str.group(1) if seed_str else "0"
            num = num_str.group(1) if num_str else "0"
            return f'acc_{self.name}_bs{bs}_seed{seed}_num{num}'
        else:
            return f'acc_{self.name}'

    def get_output_str(self, command):
        if self.type == BenchmarkType.ceval.value:
            return "--save_dir", ""
        else:
            return "--save_dir", "--result-file"
        
    def get_result_ext(self):
        return '.jsonl' if self.type == BenchmarkType.mmlu.value else '.txt'
    
    def get_csv_result_str(self):
        return 'acc'
    