from dataclasses import dataclass
import glob
import math
import os
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union
import pandas as pd
import sqlite3
import argparse
from tabulate import tabulate
from sqlalchemy import JSON, Integer, and_, create_engine, Column, String, Float, DateTime, func, or_, select, over, desc, asc, tuple_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import OperationalError
import logging



MODEL_DATA_FILE_NAME = "summary.csv"

LOCAL_DB_NAME = "model_best_performance.db"

# 远程 DB（MySQL，使用 ORM）
#    格式: mysql+pymysql://用户名:密码@主机:端口/数据库名
REMOTE_DB_URL = "mysql+pymysql://pdeai-readonly:XeGGx3pg6UzMnnPa@sw-db-prod.metax-internal.com:3306/aone"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class TaskInfo:
    model_name: 'str'
    tp: 'int'
    pp: 'int'
    dp: 'int'


# ----------------------------- ORM 模型定义 -----------------------------
# 本地最优记录模型
BaseLocal = declarative_base()
class LocalBestPerformance(BaseLocal):
    __tablename__ = "best_performance"

    model_name = Column(String, primary_key=True)
    tp = Column(Integer, primary_key=True)
    pp = Column(Integer, primary_key=True)
    dp = Column(Integer, primary_key=True)
    concurrency = Column(Integer, primary_key=True)
    input = Column(Integer, primary_key=True)
    output = Column(Integer, primary_key=True)

    device_type = Column(String)
    best_tps = Column(Float, nullable=False)
    update_date = Column(DateTime, nullable=False)

# 远程每日记录模型（只需用于查询 MAX，不需要完整结构）
BaseRemote = declarative_base()
class DailyPerformance(BaseRemote):
    __tablename__ = "test_record"
    id = Column(Integer, primary_key=True)

    model_name = Column(String)
    concurrency = Column(Integer)
    input_len = Column(Integer)
    output_len = Column(Integer)

    tps = Column(Float)
    test_date = Column(DateTime)
    meta_data = Column(JSON)
    device_type = Column(String)
    task_type = Column(String)
    test_cycle_type = Column(String)
    evaluation_framework = Column(String)


def create_engines(local_db_url: 'str'):
    logging.info(f"Database URL: {local_db_url}")

    engine_local = create_engine(local_db_url, echo=False, future=True)
    # 远程 URL 请根据实际情况修改
    engine_remote = create_engine(
        REMOTE_DB_URL,
        echo=False, future=True, pool_pre_ping=True
    )

    return engine_local, engine_remote


def create_sessions(engine_local, engine_remote):
    SessionLocal = scoped_session(sessionmaker(bind=engine_local))
    SessionRemote = scoped_session(sessionmaker(bind=engine_remote))
    return SessionLocal, SessionRemote


def find_summary_csv(root: 'Path') -> 'Path':
    matches = []
    
    for path in root.rglob(MODEL_DATA_FILE_NAME):
        if not path.is_file():
            continue

        matches.append(path.resolve())
    
    if len(matches) > 1:
        print(f"Warning: found multiple {MODEL_DATA_FILE_NAME}. The first one will be extracted.")

    if not matches:
        return None
    return matches[0]


def get_performence_dir(base_path: 'Path') -> 'Path':
    pattern = re.compile(r'^\d{8}_\d{4}$')
    
    # 获取第一层子目录（仅目录，不含文件）
    matched_dirs = [
        p for p in base_path.iterdir() 
        if p.is_dir() and pattern.match(p.name)
    ]
    
    # 验证数量
    if len(matched_dirs) == 0:
        raise ValueError(f"No directory with format 'YYYYMMDD_HHMM' found in {base_path}")
    elif len(matched_dirs) > 1:
        names = [d.name for d in matched_dirs]
        raise ValueError(f"Multiple matching directories found (expected exactly one): {names}")
    
    return matched_dirs[0]


def get_summary_df(csv_dir: 'Path') -> 'pd.DataFrame':
    if not csv_dir.exists():
        raise FileNotFoundError(f"The summary csv is not existed! file: {csv_dir.resolve()}")
    
    df_summary = pd.read_csv(csv_dir)

    rename_map = {
        col: 'input_len' if 'input_len' in col else 'output_len'
        for col in df_summary.columns
        if 'input_len' in col or 'output_len' in col
    }

    df_summary['date'] = pd.to_datetime(df_summary['date'], format='%Y%m%d-%H%M%S')
    df_summary['start_time'] = df_summary['date'] - pd.to_timedelta(df_summary['duration'], unit='s')

    # 执行重命名
    df_summary.rename(columns=rename_map, inplace=True)
    return df_summary


def query_local_best(keys: 'List[Tuple]', SessionLocal) -> 'Dict[Tuple, float]':
    with SessionLocal() as sess_local:
        results: 'List[LocalBestPerformance]' = sess_local.query(LocalBestPerformance).filter(
            tuple_(
                LocalBestPerformance.device_type, LocalBestPerformance.model_name, LocalBestPerformance.tp,
                LocalBestPerformance.pp, LocalBestPerformance.dp,
                LocalBestPerformance.concurrency, LocalBestPerformance.input,
                LocalBestPerformance.output
            ).in_(keys)
        ).all()
        
        return {
            (r.model_name, r.tp, r.pp, r.dp, r.concurrency, r.input, r.output): (r.best_tps, r.update_date)
            for r in results
        }


def query_remote_best(keys: 'List[Tuple]', SessionRemote) -> 'Dict[Tuple, float]':
    if not keys:
        return {}
    
    # 分组列名（根据你的表结构调整）
    group_cols = [
        'device_type', 'model_name', 'tp', 'pp', 'dp',
        'concurrency', 'input', 'output'
    ]
    
    json_conditions = []
    for device_type, model_name, tp, pp, dp, concurrency, input_tokens, output_tokens in keys:
        json_conditions.append(
            and_(
                DailyPerformance.device_type == device_type,
                DailyPerformance.model_name == model_name,
                DailyPerformance.concurrency == concurrency,
                DailyPerformance.input_len == input_tokens,
                DailyPerformance.output_len == output_tokens,
                func.json_extract(DailyPerformance.meta_data, '$.tp') == tp,
                func.json_extract(DailyPerformance.meta_data, '$.pp') == pp,
                func.json_extract(DailyPerformance.meta_data, '$.dp') == dp,
                DailyPerformance.task_type == "LLM-Inference",
                DailyPerformance.evaluation_framework == "vLLM",
                DailyPerformance.test_cycle_type == "daily"
            )
        )
    
    with SessionRemote() as sess_remote:
        # 批量查询匹配的所有行
        results = sess_remote.query(
            DailyPerformance.device_type,
            DailyPerformance.model_name,
            func.json_extract(DailyPerformance.meta_data, '$.tp').label('tp'),
            func.json_extract(DailyPerformance.meta_data, '$.pp').label('pp'),
            func.json_extract(DailyPerformance.meta_data, '$.dp').label('dp'),
            DailyPerformance.concurrency,
            DailyPerformance.input_len,
            DailyPerformance.output_len,
            DailyPerformance.tps,
            DailyPerformance.test_date
        ).filter(
            or_(*json_conditions)
        ).all()
        
        if not results:
            logging.info(f"{model_name}-tp{tp}-pp{pp}-dp{dp} doesn't have any data!")
            return {}
        
        # 转 Pandas
        df = pd.DataFrame(results, columns=group_cols + ['tps', 'test_date'])
        
        # 1. 先按分组取 max tps
        max_tps = df.groupby(group_cols)['tps'].max().reset_index()
        
        # 2. 合并回原 df，筛选出 tps == max_tps 的行
        df_with_max = df.merge(max_tps, on=group_cols + ['tps'], how='inner')
        
        # 3. 在 max tps 行中取最早 test_date
        best = df_with_max.loc[
            df_with_max.groupby(group_cols)['test_date'].idxmin()
        ]
        
        # 构建返回 dict
        best_dict = {
            tuple(row[group_cols]): (row['tps'], row['test_date'])
            for _, row in best.iterrows()
        }
        
        return best_dict
    

def update_local_best(best_list: 'List[LocalBestPerformance]', SessionLocal):
    if not best_list:
        return
    
    with SessionLocal() as sess_local:
        for ins in best_list:
            sess_local.merge(ins)
        sess_local.commit()


# def _parse_time(t: 'Union[datetime, str]') -> 'datetime':
#     if isinstance(t, str):
#         dt = datetime.strptime(t, "%Y%m%d-%H%M%S")
#         tz_beijing = timezone(timedelta(hours=8))
#         dt_aware = dt.replace(tzinfo=tz_beijing)
    
#         return dt
#     else:
#         logging.info(f"parse time: {type(t)}")
#         return t


def process_performance(model_info: 'TaskInfo', df: 'pd.DataFrame', threshold: 'float', device_type: 'str', SessionLocal, SessionRemote) -> 'Tuple[bool, str, List[int]]':
    model_name = model_info.model_name
    tp, pp, dp = (model_info.tp, model_info.pp, model_info.dp)

    keys = [
        (device_type, model_name, tp, pp, dp, r['max_concurrency'], r['input_len'], r['output_len'])
        for _, r in df.iterrows()
    ]

    local_best = query_local_best(keys, SessionLocal)
    missing_local_keys = [k for k in keys if k not in local_best]
    remote_best = query_remote_best(missing_local_keys, SessionRemote)
    # missing_keys = [k for k in missing_local_keys if k not in remote_best]
    # new_best = {
    #     k: None
    #     for k in missing_keys
    # }

    history_best = local_best.copy()
    history_best.update(remote_best)

    to_merge = []
    failed_indices = []
    for idx, r in df.iterrows():
        key = (model_name, tp, pp, dp, r['max_concurrency'], r['input_len'], r['output_len'])
        current_tps = r['output_throughput']
        history = history_best.get(key, (-math.inf, r['date']))
        history_tps = history[0]

        final_best = max(current_tps, history_tps)

        # 数据不存在或者是远程存在，但是本地没有，都需要更新到本地
        if final_best >= history_tps or remote_best.get(key, None):
            the_best: 'LocalBestPerformance' = LocalBestPerformance(
                model_name = model_name, tp = tp, pp = pp, dp = dp,
                concurrency = r['max_concurrency'], input = r['input_len'],
                output = r['output_len'], best_tps = final_best,
                update_date = r['date']
            )
            to_merge.append(the_best)
        
        if current_tps < final_best * (1 - threshold):
            failed_indices.append(idx)

    update_local_best(to_merge, SessionLocal)

    total_cases = df.size
    pass_count = total_cases - len(failed_indices)
    cases_status = f"{pass_count}/{total_cases} (pass/total)"

    return pass_count == total_cases, cases_status, failed_indices



def info_complement(row: 'pd.Series', threshold: 'float', device_type: 'str', SessionLocal, SessionRemote) -> 'pd.Series':
    new_cols = {}

    # 处理model_name
    task_name: 'str' = row["task_name"]
    model_name = task_name.rsplit('_', 3)[0]

    match = re.search(r'_tp(\d+)_pp(\d+)_dp(\d+)', task_name)
    tp = int(match.group(1)) if match and match.group(1) else 1
    pp = int(match.group(2)) if match and match.group(2) else 1
    dp = int(match.group(3)) if match and match.group(3) else 1

    new_cols["model_name"] = model_name
    new_cols["tp"] = tp
    new_cols["pp"] = pp
    new_cols["dp"] = dp

    # 处理summary.csv
    log_dir = Path(row['log_dir'])
    root = log_dir.parent / log_dir.stem.rsplit('_', 1)[0]
    new_cols["summary_dir"] = find_summary_csv(root)

    assert new_cols["summary_dir"] is not None or row["status"] != "success", f"Only the task [{row['task_name']}]'s status is success can a summary.csv be found!"

    # 处理性能
    if new_cols["summary_dir"]:
        task_info = TaskInfo(model_name=model_name, tp=tp, pp=pp, dp=dp)
        df_summary = get_summary_df(new_cols["summary_dir"])

        ispass, cases_status, failed_indices = process_performance(task_info, df_summary, threshold, device_type, SessionLocal=SessionLocal, SessionRemote=SessionRemote)

        new_cols["cases_status"] = cases_status
        new_cols["failed_indices"] = failed_indices
        new_cols["benchmark_start"] = df_summary["start_time"].min()
        new_cols["benchmark_end"] = df_summary["date"].max()
        new_cols["actual_duration (s)"] = df_summary["duration"].sum()
        new_cols["whole_duration (s)"] = (new_cols['benchmark_end'] - new_cols['benchmark_start']).total_seconds()

        if ispass:
            new_cols["note"] = "pass"
        else:
            new_cols["note"] = "performance degradation"
    else:
        new_cols["cases_status"] = "Runtime Error"
        new_cols["failed_indices"] = None
        new_cols["note"] = "Runtime Error"

    return pd.Series(new_cols)


# ----------------------------- 主逻辑 -----------------------------
def main(res_dir: Path, local_db_url: 'str', device_type: 'str', threshold: 'float'):
    # 创建 engine 和 session
    engine_local, engine_remote = create_engines(local_db_url)
    SessionLocal, SessionRemote = create_sessions(engine_local, engine_remote)

    # 创建本地数据库表，如果存在则跳过
    BaseLocal.metadata.create_all(engine_local)
    logging.debug("The tables of local DB are already checked/created.")

    df_bench_result = pd.read_csv(res_dir / "performance" / "bench_tasks_result.csv")
    total_tasks_len = df_bench_result["status"].size
    runtime_pass_tasks_len = (df_bench_result["status"] == "success").sum()

    new_cols = df_bench_result.apply(info_complement, axis=1, args=(threshold, device_type, SessionLocal, SessionRemote))
    df_bench_result[new_cols.columns] = new_cols

    performence_pass_num = (df_bench_result["note"] == "pass").sum()
    df_not_pass = df_bench_result[df_bench_result["note"] != "pass"][["model_name", "note", "cases_status", "failed_indices"]]
    
    logging.info("+" * 100)
    logging.info("=" * 100)
    logging.info(f"Execute succefully: {performence_pass_num} | {runtime_pass_tasks_len} | {total_tasks_len} (pass | run success | total)")
    if len(df_not_pass) > 0:
        logging.info("tasks not pass:")
        print(tabulate(
            df_not_pass,
            headers='keys',
            tablefmt='psql',
            showindex=False,
            stralign='left',
            numalign='left',
            missingval='None'
        ), flush=True)
    logging.info("=" * 100)
    logging.info("+" * 100)


    report_file = res_dir / "report.csv"
    df_bench_result.to_csv(report_file, index=False, encoding='utf-8')


def create_parser():
    parser = argparse.ArgumentParser(
        description="Modelzoo.llm.vllm Daily test analyser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--res-dir",
        type=Path,
        required=True,
        help="The director of daily test result."
    )

    parser.add_argument(
        "--db-dir",
        type=Path,
        required=True,
        help="The directory of local .db file to record the best performance.",
    )

    parser.add_argument(
        "--device-type",
        default="C500",
        type=str,
        help="The device type of current dailytest"
    )

    parser.add_argument(
        "--threshold",
        default=0.1,
        type=float,
        help="The threshold at which performance rollback is allowed."
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    res_dir = get_performence_dir(args.res_dir)
    db_dir: Path = args.db_dir

    

    if not res_dir.exists() or not db_dir.exists():
        raise FileNotFoundError(f"{res_dir.resolve()} or {db_dir.resolve()} is not found!")
    if not db_dir.is_dir():
        raise ValueError(f"--db-dir can only accept a directory not a file. provided: {db_dir}")
    
    db_url = "sqlite:///" + str(db_dir.resolve() / LOCAL_DB_NAME)
    main(res_dir, db_url, args.device_type, args.threshold)