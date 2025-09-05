import argparse
import json
import csv
import os
from datetime import datetime


daily_test = ["./benchmark_daily.json"]
weekly_test = ["./benchmark_weekly.json", "./acc_weekly.json"]

daily_test_master = ["./dailytest/benchmark_daily.json"]
weekly_test_master = ["./dailytest/benchmark_weekly.json", "./dailytest/acc_weekly.json"]

def replace_path(args, task_list, is_reverse = False):
    for task in task_list:
        content = ''
        with open(task, "r", encoding='utf-8') as f:
            content = f.read()

        if not is_reverse:
            content = content.replace("{--model-path}", args.model_path)
            content = content.replace("{--dataset-path}", args.dataset_path)
            content = content.replace("{--ceval-path}", args.ceval_path)
            content = content.replace("{--mmlu-path}", args.mmlu_path)
        else:
            content = content.replace(args.dataset_path, "{--dataset-path}")
            content = content.replace(args.ceval_path, "{--ceval-path}")
            content = content.replace(args.mmlu_path, "{--mmlu-path}")
            content = content.replace(args.model_path, "{--model-path}")

        content_out = json.loads(content)
        with open(task, "w", encoding="utf-8") as f:
            json.dump(content_out, f, indent=4, ensure_ascii=False)


def delete_files_except_a_json(directory):
    # 遍历指定目录下的所有子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查是否为a.json文件
            if file != "benchmark_result.json" and file != "acc_mmlu_result.json" and file != "acc_ceval_result.json":
                # 构建文件的完整路径
                file_path = os.path.join(root, file)
                # 删除文件
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")


def delete_unnecessary_content(file_data, file_name):
    data_list = []
    for content in file_data:
        data = {}
        if file_name == "benchmark_result.csv":
            data = {
                "Att Backend": "flashinfer",
                "batch-size": "1",
                "in-out": "128-64",
                "Request throughput (req/s)": "0",
                "Input token throughput (tok/s)": "0",
                "Output token throughput (tok/s)": "0",
                "Total token throughput (tok/s)": "0",
                "Concurrency": "0",
                "Mean E2E Latency (ms)": "0",
                "Median E2E Latency (ms)": "0",
                "Mean TTFT (ms)": "0",
                "Median TTFT (ms)": "0",
                "P99 TTFT (ms)": "0",
                "Mean TPOT (ms)": "0",
                "Median TPOT (ms)": "0",
                "P99 TPOT (ms)": "0",
                "Mean ITL (ms)": "0",
                "Median ITL (ms)": "0",
                "P99 ITL (ms)": "0",
                "P95 ITL (ms)": "0",
                "Max ITL (ms)": "0",
                "Decoding TPS(toks/s)": "0",
                "TGS(toks/GPU/s)": "0",
                "Interactivity(toks/User/s)": "0"
            }
            for key, val in content.items():
                if key in data.keys():
                    data[key] = content[key]
        elif file_name == "acc_mmlu_result.csv":
            data = {"dataset":'mmlu', "Accuracy": "0.0"}
            data["Accuracy"] = content["Accuracy"]
        elif file_name == "acc_ceval_result.csv":
            data = {"dataset":'ceval', "Accuracy": "0.0"}
            data["Accuracy"] = content["Accuracy"]
        data_list.append(data)
    return data_list


def csv_to_json(output_path):
    # 遍历指定目录下的所有子目录
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if not file.endswith("benchmark_result.csv") and not file.endswith("acc_mmlu_result.csv") and not file.endswith("acc_ceval_result.csv"):
                continue

            # 打开 CSV 文件并读取数据
            file_path = root + '/' + file
            with open(file_path, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                data = list(csv_reader)

            # 处理json文件输出结果
            data = delete_unnecessary_content(data, file)
            # 将数据转换为 JSON 格式并写入文件
            file_name = os.path.splitext(os.path.basename(file))
            file_path = root + '/' + file_name[0] + ".json"
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)


def process_result_file(output_path):
    # 获取当前目录下的所有文件夹
    folders = [f for f in os.listdir('../../outputs/') if os.path.isdir(os.path.join('../../outputs/', f))]

    # 创建一个字典来存储文件夹名和它们的最后修改时间
    folder_times = {}

    for folder in folders:
        folder_path = os.path.join('../../outputs/', folder)
        last_modified = os.path.getmtime(folder_path)
        folder_times[folder] = datetime.fromtimestamp(last_modified)

    # 找到最后修改时间最晚的文件夹
    latest_folder = max(folder_times, key=folder_times.get)
    print("最新的文件夹是:", latest_folder)

    # 将result目录迁移到当前指定的目录下
    print("最新的输出目录是:", output_path)

    # 迁移 benchmark 相关的result
    result_path = "../../outputs/" + latest_folder + "/dailytest/benchmark/result/online/ "
    cmds = f"cp -r " + result_path + output_path + "result"
    # 迁移 acc 相关的result
    result_path = "../../outputs/" + latest_folder + "/dailytest/acc/result/online/ceval/* "
    cmds = cmds + "; cp -r " + result_path + output_path + "result"
    result_path = "../../outputs/" + latest_folder + "/dailytest/acc/result/online/mmlu/* "
    cmds = cmds + "; cp -r " + result_path + output_path + "result"
    cmds = cmds + "; rm -rf " + output_path + "result/online"
    print(cmds)
    os.system(cmds)

    # 将csv文件转换为json文件
    csv_to_json(output_path + "result")
    # 删除多余文件
    delete_files_except_a_json(output_path + "result")

        
def process_command_line_param(args):
    if args.output_path[-1] != "/":
        args.output_path += "/"

    if args.model_path[-1] != "/":
        args.model_path += "/"
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--model-path", type=str, default="",help="model path")
    parser.add_argument("--dataset-path", type=str, default="",help="dataset-path")
    parser.add_argument("--ceval-path", type=str, default="",help="acc ceval path")
    parser.add_argument("--mmlu-path", type=str, default="",help="acc mmlu path")
    parser.add_argument("--output-path", type=str, default="./",help="result output path")
    parser.add_argument("--benchmark-name", type=str, default="daily",help="daily or weekly test")
    args = parser.parse_args()
    args = process_command_line_param(args)

    task_list = []
    if args.benchmark_name == "weekly":
        task_list.append(weekly_test)
        task_list.append(weekly_test_master)
    else:
        task_list.append(daily_test)
        task_list.append(daily_test_master)

    replace_path(args, task_list[0])

    task_cmd = f" --task "
    for task in task_list[1]:
        task_cmd += task
        task_cmd += " "
    benchmark_cmd = f"cd ..;python3 -m src.master --output-path ../outputs/ " + task_cmd
    os.system(benchmark_cmd)

    replace_path(args, task_list[0], True)

    process_result_file(args.output_path)
