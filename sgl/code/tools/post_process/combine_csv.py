'''
    在测试还没全都结束的时候使用，生成csv
'''

import os
import pandas as pd

def merge_result_csvs(root_dir, output_file="merged_result.csv"):
    """
    遍历 root_dir 下所有一级子文件夹中的 result 文件夹，
    并递归遍历 result 下所有子文件夹，合并其中所有 CSV 文件
    
    Args:
        root_dir (str): 根路径（要遍历的起始文件夹）
        output_file (str): 合并后的输出文件路径
    """
    all_data = []
    processed_files = 0

    # 第一步：获取 root_dir 下的所有一级子文件夹（仅一级）
    first_level_folders = [
        os.path.join(root_dir, folder)
        for folder in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, folder))
    ]

    # 第二步：遍历每个一级子文件夹，处理其中的 result 文件夹
    for folder_path in first_level_folders:
        result_root = os.path.join(folder_path, "result")
        if not os.path.exists(result_root) or not os.path.isdir(result_root):
            print(f"跳过：未找到 result 文件夹 - {result_root}")
            continue

        # 第三步：递归遍历 result 文件夹下的所有层级，找所有 CSV
        for dirpath, _, filenames in os.walk(result_root):
            for filename in filenames:
                if filename.lower().endswith(".csv"):  # 兼容大写 .CSV
                    csv_path = os.path.join(dirpath, filename)
                    try:
                        # 读取 CSV（根据实际情况调整 header/encoding）
                        df = pd.read_csv(
                            csv_path,
                            encoding="utf-8",       # 中文乱码改 gbk
                            # errors="ignore",        # 忽略编码错误
                            skip_blank_lines=True,  # 跳过空行
                            header=0                # 有表头用0，无表头用None
                        )
                        # 可选：添加溯源列，方便定位数据来源
                        # df["source_folder"] = os.path.basename(folder_path)  # 一级子文件夹名
                        # df["source_csv"] = csv_path                          # CSV 完整路径
                        all_data.append(df)
                        processed_files += 1
                        print(f"成功读取：{csv_path}")
                    except Exception as e:
                        # 单个文件失败不中断，仅打印错误
                        print(f"读取失败 {csv_path}：{str(e)}")

    # 第四步：合并所有 CSV 数据并保存
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        # 保存（utf-8-sig 兼容 Windows 记事本）
        merged_df.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )
        # 输出统计信息
        print("\n" + "-"*50)
        print(f"合并完成！")
        print(f"共处理一级子文件夹数：{len(first_level_folders)}")
        print(f"共读取 CSV 文件数：{processed_files}")
        print(f"合并后总行数：{len(merged_df)}")
        print(f"合并文件路径：{os.path.abspath(output_file)}")
    else:
        print("\n未找到任何符合条件的 CSV 文件！")

# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 替换为你的实际根路径（比如 /root/mxlog/umd 或 D:/data）
    ROOT_DIRECTORY = "/external/ai/share/m01305/ModelZoo.LLM.Inference/sgl/daily_result/2026-03-01-two-machine-acc/20260301_214827"
    # 调用合并函数
    merge_result_csvs(
        root_dir=ROOT_DIRECTORY,
        output_file="merged_all_results.csv"
    )