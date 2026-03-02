import re
import os
import csv
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def process_log_file(log_dir):
    """处理指定目录下的.log文件，生成Task_info.csv在脚本所在目录"""
    print("="*50)
    print("模型信息提取工具 (脚本所在目录生成)")
    print("="*50)
    
    # 获取脚本所在目录（用于生成Task_info.csv）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 在指定目录下查找一级 .log 文件
    log_files = []
    for item in os.listdir(log_dir):
        item_path = os.path.join(log_dir, item)
        # 检查是否是文件且以.log结尾
        if os.path.isfile(item_path) and item.lower().endswith('.log'):
            log_files.append(item)
    
    if not log_files:
        print(f"\n错误: 在目录 '{log_dir}' 中未找到任何 .log 文件。")
        return None
    
    # 取第一个找到的.log文件
    log_file = os.path.join(log_dir, log_files[0])
    print(f"\n找到 .log 文件: {log_file}")
    
    # 读取文件内容
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except Exception as e:
        print(f"\n错误: 读取文件时出错 - {str(e)}")
        print("可能原因:")
        print("  1. 文件编码问题 (尝试修改为 'gbk' 或 'latin-1')")
        print("  2. 文件正在被其他程序使用")
        return None
    
    print(f"\n正在处理文件: {log_file}")
    print("提取中...")

    # 从日志内容中提取模型信息
    model_info = []
    lines = log_content.split('\n')
    
    for i, line in enumerate(lines):
        # 检查是否包含 "Start Task"，如果包含则退出循环
        if "Start Task" in line:
            print("\nFound 'Start Task', stopping extraction.")
            break
            
        # 检查是否是模型名称行（以 |— 开头）
        if "|——" in line:
            task_name = line.split("|——")[1].strip()
            
            # 检查下一行是否是启动命令
            if i + 1 < len(lines) and "|___" in lines[i + 1]:
                start_command = lines[i + 1].split("|___")[1].strip()
                
                # 检查下下一行是否是环境变量
                if i + 2 < len(lines) and "|___" in lines[i + 2]:
                    env_vars = lines[i + 2].split("|___")[1].strip()
                    
                    model_info.append({
                        "task_name": task_name,
                        "start_command": start_command,
                        "env_vars": env_vars
                    })
    
    # 显示结果
    if not model_info:
        print("\n未找到任何模型信息。")
        return None

    print(f"\n共找到 {len(model_info)} 个模型配置:")
    for i, info in enumerate(model_info, 1):
        print(f"  {i}. {info['task_name']}")
    
    # === 生成文件在脚本所在目录 ===
    output_file = os.path.join(script_dir, "task_info.csv")
    
    try:
        # 按模型名排序
        sorted_result = sorted(model_info, key=lambda x: x['task_name'].lower())
        
        # 写入CSV文件（覆盖式生成）
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['TASK_NAME', 'CMD', 'ENV'])
            for info in sorted_result:
                writer.writerow([
                    info['task_name'],
                    info['start_command'],
                    info['env_vars']
                ])
        
        print(f"\n✅ 结果已自动保存到: {output_file}")
        print("CSV文件已按模型名排序生成（覆盖式保存）")
        return output_file
    except Exception as e:
        print(f"\n❌ 错误: 保存文件时出错 - {str(e)}")
        return None

def process_csv_files(csv_dir):
    """处理指定目录下的一级CSV文件（仅处理merge_perf_result.csv和merge_acc_result.csv），调整列顺序并保存到脚本所在目录（文件名不变）"""
    print("\n" + "="*50)
    print("正在处理CSV文件（仅处理merge_perf_result.csv和merge_acc_result.csv）")
    print("="*50)
    
    # 获取脚本所在目录（用于保存新CSV文件）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义要处理的两个文件及其列顺序要求
    files_to_process = [
        {
            "filename": "merge_perf_result.csv",
            "required_columns": ["TaskName", "in-out", "Parallelism", "Max request concurrency", "mem_frac", "available_mem"]
        },
        {
            "filename": "merge_acc_result.csv",
            "required_columns": ["dataset", "TaskName", "Parallelism", "Accuracy"]
        }
    ]
    
    for file_info in files_to_process:
        target_filename = file_info["filename"]
        required_columns = file_info["required_columns"]
        
        # 检查目录下是否存在该文件（不区分大小写）
        found = False
        for filename in os.listdir(csv_dir):
            if filename.lower() == target_filename.lower():
                csv_path = os.path.join(csv_dir, filename)
                print(f"\n正在处理 CSV 文件: {csv_path}")
                found = True
                break
        if not found:
            print(f"  ❌ 文件 {target_filename} 未在目录中找到，跳过处理。")
            continue

        try:
            # 读取csv文件
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # 检查列名是否存在
            columns = reader.fieldnames
            valid_columns = []
            for col in required_columns:
                found_col = None
                for c in columns:
                    if c.lower() == col.lower():
                        found_col = c
                        break
                if found_col is None:
                    print(f"❌ 错误：csv文件 {target_filename} 缺少列 '{col}'，跳过处理。")
                    break
                valid_columns.append(found_col)
            else:
                # 如果所有列都存在
                # 保留其他列（如果有）在最后
                other_columns = [col for col in columns if col not in valid_columns]
                new_columns = valid_columns + other_columns

                # --- 新增：多级排序逻辑 ---
                # 配置：指定排序到第几列（从1开始计数，这里用索引，所以前k列是new_columns[:k]）
                # 例如：排序到第2列，就设置 sort_up_to = 2
                sort_columns = new_columns[:4]
                # 2. 新增：额外要排序的列（自定义，需确保列名存在）
                extra_sort_columns = ["Torch Compile","Cache", "speculative-algorithm"]  # 替换成你要额外排序的列名，如["延迟", "成功率"]
                
                # 校验额外列是否存在，避免KeyError
                valid_extra_columns = []
                for col in extra_sort_columns:
                    found_col = None
                    for c in new_columns:
                        if c.lower() == col.lower():
                            found_col = c
                            break
                    if found_col:
                        valid_extra_columns.append(found_col)
                    else:
                        print(f"⚠️ 警告：额外排序列 '{col}' 不存在，已跳过。")

                def get_sort_key(row):
                    """生成排序用的key：基础列 + 额外列，自动处理数字类型"""
                    key_parts = []
                    # 第一步：基础排序列（sort_columns）
                    for col in sort_columns:
                        val = row[col]
                        # 尝试转换为数字类型，避免字符串排序问题
                        try:
                            key_parts.append(int(val))
                        except (ValueError, TypeError):
                            try:
                                key_parts.append(float(val))
                            except (ValueError, TypeError):
                                key_parts.append(val)
                    
                    # 第二步：额外排序列（valid_extra_columns）
                    for col in valid_extra_columns:
                        val = row[col]
                        # 同样处理数字类型
                        try:
                            key_parts.append(int(val))
                        except (ValueError, TypeError):
                            try:
                                key_parts.append(float(val))
                            except (ValueError, TypeError):
                                key_parts.append(val)
                    
                    return tuple(key_parts)


                # 执行多级排序
                sorted_rows = sorted(rows, key=get_sort_key)

                # 构建新文件路径（脚本所在目录，文件名不变）
                new_csv_path = os.path.join(script_dir, target_filename)
                # 写入新CSV文件
                with open(new_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=new_columns)
                    writer.writeheader()
                    # 遍历排序后的行
                    for row in sorted_rows:
                        new_row = {col: row[col] for col in new_columns}
                        writer.writerow(new_row)

                print(f"✅ 已保存调整列顺序并排序后的CSV到：{new_csv_path}")

        except Exception as e:
            print(f"❌ 错误：处理文件 {csv_path} 时出错 - {str(e)}")

def generate_template_csv(col_csv_path, row_csv_path, output_csv_path, fill_value=np.nan):
    """
    自适应生成新CSV（无固定行列数，完全按输入CSV的实际内容）：
    - 列标题：取自col_csv_path的所有列名
    - 行标题：取自row_csv_path第一列的所有行值
    - 最终格式：(行标题数+1)行 × (列标题数+1)列（+1为标题行/列）
    - 空值填充：默认NaN，可自定义（如0、""等）
    
    Args:
        col_csv_path: 提供列标题的CSV文件路径
        row_csv_path: 提供行标题的CSV文件路径
        output_csv_path: 生成的新CSV路径
        fill_value: 空单元格填充值（默认NaN）
    
    Returns:
        bool: 是否生成成功
    """
    try:
        # 1. 读取列标题CSV：提取所有列名（自适应列数）
        col_df = pd.read_csv(col_csv_path)
        col_headers = col_df.columns.tolist()  # 取所有列名作为数据列标题
        col_count = len(col_headers)
        if col_count == 0:
            print("❌ 错误：列CSV无有效列标题")
            return False
        print(f"📌 从列CSV读取到 {col_count} 列标题：{col_headers}")

        # 2. 读取行标题CSV：提取第一列所有行值作为行标题（自适应行数）
        row_df = pd.read_csv(row_csv_path)
        if row_df.empty or row_df.shape[1] == 0:
            print("❌ 错误：行CSV无有效数据")
            return False
        # 取第一列所有非空行值作为行标题（转字符串避免类型问题）
        row_headers = row_df.iloc[:, 0].dropna().astype(str).tolist()
        row_count = len(row_headers)
        if row_count == 0:
            print("❌ 错误：行CSV第一列无有效行标题")
            return False
        print(f"📌 从行CSV读取到 {row_count} 行标题：{row_headers}")

        # 3. 动态构造新CSV结构
        # 列结构：标题列 + 所有数据列
        new_columns = col_headers
        # 行结构：标题行 + 所有数据行
        # 创建空数据矩阵（行标题数 × 列标题数）
        data_matrix = np.full((row_count, col_count-1), fill_value)
        # 组合行标题和数据
        new_data = [ [row_header] + list(row_data) for row_header, row_data in zip(row_headers, data_matrix) ]
        # 插入标题行
        new_data.insert(0, new_columns)

        # 4. 转换为DataFrame并保存（自适应行列数）
        new_df = pd.DataFrame(new_data[1:], columns=new_data[0])
        new_df.to_csv(output_csv_path, index=False, encoding='utf-8')

        # 打印结果统计
        final_rows = new_df.shape[0]
        final_cols = new_df.shape[1]
        print(f"\n✅ 新CSV生成成功！路径：{output_csv_path}")
        print(f"📊 最终格式：{final_rows}行（含标题行） × {final_cols}列（含标题列）")
        print(f"   - 标题行1行 + 数据行{row_count}行 = {final_rows}行")
        print(f"   - 标题列1列 + 数据列{col_count}列 = {final_cols}列")
        return True

    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件 - {e.filename}")
        return False
    except Exception as e:
        print(f"❌ 生成失败：{str(e)}")
        return False

def main():
    """主函数：依次调用日志处理和CSV处理函数"""
    # 交互式获取目录路径
    while True:
        dir_path = input("\n请输入要处理的目录路径: ").strip()
        
        # 检查目录是否存在
        if os.path.isdir(dir_path):
            break
        else:
            print(f"错误: 目录 '{dir_path}' 不存在！")
            print("请检查路径是否正确，或使用绝对路径。")
            print("示例路径:")
            print("  Linux/MacOS: /var/log")
            print("  Windows: C:\\logs")
    
    # 1. 处理日志文件
    log_result = process_log_file(dir_path)
    
    # 2. 处理CSV文件
    process_csv_files(dir_path)

    #3. 生成模版csv
    # generate_template_csv("merge_perf_result.csv","task_info.csv","template_perf_result.csv")
    # generate_template_csv("merge_acc_result.csv","task_info.csv","template_acc_result.csv")
    print("\n" + "="*50)
    print("所有处理已完成")
    print("="*50)

if __name__ == "__main__":
    main()