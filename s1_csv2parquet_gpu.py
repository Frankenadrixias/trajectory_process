# -*-coding:utf-8 -*-
"""
@File    :   s1_csv2parquet.py
@Time    :   2026/01/13 16:37
@Author  :   huangsh, FriedrichXR
@Version :   2.0
@Contact :   1126456109@qq.com，2249307370@qq.com
@Desc    :   将原始的解压文件并行处理转换为排序好的parquet文件，同时提取用户信息表
"""
import gc
import logging
import warnings
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from typing import Tuple

"""
将原始的解压文件（data_v2）转换为排序好的parquet文件（data_v3）
输入：data_v2
输出：data_v3

输入目录结构要求：
F:/data_v2/北京24/
    ├── SceneReco/
    │   ├── 2024-01-01.csv
    │   └── 2024-01-02.parquet
    ├── Timing/
    ├── WifiConnect/
    └── WifiStable/

输出目录结构：
F:/data_v3/北京24/
    ├── SceneReco/
    │   ├── 2024-01-01.parquet
    │   └── 2024-01-02.parquet
    ├── Timing/
    ├── WifiConnect/
    └── WifiStable/
"""


import os
import logging
import warnings
import gc
from pathlib import Path
from typing import Tuple, List, Set
from time import time

# ================= 关键库替换 =================
try:
    import cudf  # 核心库：GPU DataFrame
    import cupy  # 辅助库：通用GPU计算

    GPU_AVAILABLE = True
except ImportError:
    # 如果没有GPU环境，回退到Pandas（为了代码鲁棒性，但建议在GPU环境运行）
    import pandas as cudf

    GPU_AVAILABLE = False
    print("警告：未检测到cudf，将使用CPU运行（速度较慢）")

# ================= 配置数据 =================
config = {
    "input_root": Path(r"G:\data_v3\北京24_1-3"),
    "output_root": Path(r"G:\data_v3\北京24_1-3"),
    # 注意：GPU模式下忽略进程数配置，强制串行以保护显存
    "datatypes": {"SceneReco": 8, "Timing": 1, "WifiConnect": 12, "WifiStable": 3},
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def process_file_gpu(args: Tuple[Path, Path, str]) -> Tuple[bool, str, str, Set]:
    """
    单个文件处理逻辑 (GPU版)
    """
    file_path, output_dir, category = args
    unique_users = set()

    try:
        output_file = output_dir / f"{file_path.stem}.parquet"

        # 检查是否存在
        if output_file.exists():
            logger.info(f"文件已存在，跳过: [{category}] {output_file.name}")
            # 如果需要后续合并用户，这里可能需要读取现存文件提取用户，
            # 若不需要可直接返回空集合
            return True, file_path.stem, category, set()

        logger.info(f"GPU正在处理: [{category}] {file_path.name}")
        t_start = time()

        # 1. 读取数据 (GPU I/O)
        if file_path.suffix.lower() == ".csv":
            # read_csv 在 cudf 中非常快
            df = cudf.read_csv(file_path)
        else:
            df = cudf.read_parquet(file_path)

        # 2. 列重命名
        # 假设列顺序严格一致，否则建议使用 rename 字典映射更安全
        if len(df.columns) >= 6:
            df.columns = ["Userid", "lng", "lat", "pid", "starttime", "endtime"] + list(df.columns[6:])
        else:
            logger.warning(f"列数不足: {file_path.name}")
            return False, "", "", set()

        # 3. 去重 (GPU Parallel)
        # drop_duplicates 在 GPU 上是并行的，极快
        df.drop_duplicates(subset=["Userid", "starttime", "endtime"], inplace=True)

        # 4. 排序 (GPU Radix Sort)
        # 移除 kind='mergesort'，让 cudf 自动选择最优 GPU 排序算法
        df = df.sort_values("Userid")

        # 5. 提取公共用户 (Task 1 需求)
        # 在 GPU 上计算 unique，然后将结果转回 CPU (to_pandas) 再转为 set
        # 这样不会占用显存保留结果
        extracted_users = df["Userid"].unique()

        # 兼容性处理：cudf Series 转 numpy/pandas
        if hasattr(extracted_users, "to_pandas"):
            unique_users = set(extracted_users.to_pandas())
        elif hasattr(extracted_users, "values"):
            unique_users = set(extracted_users.values.get())  # cupy -> numpy

        # 6. 保存 (GPU I/O)
        df.to_parquet(output_file, compression="snappy", index=False)

        t_end = time()
        logger.info(f"完成: {output_file.name} | 耗时: {t_end - t_start:.2f}s | 用户数: {len(unique_users)}")

        # 7. 显存清理
        del df
        del extracted_users
        # 强制垃圾回收，防止显存碎片
        gc.collect()

        return True, file_path.stem, category, unique_users

    except Exception as e:
        logger.error(f"处理失败 {file_path.name}: {str(e)}")
        # 出错时尝试清理显存
        gc.collect()
        return False, "", "", set()


def process_batch_gpu(input_dir: Path, output_dir: Path, category: str):
    """
    GPU 批处理模式 - 串行执行以节省显存
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in (".csv", ".parquet")]

    if not files:
        logger.info(f"无文件: {input_dir}")
        return []

    logger.info(f"开始处理 {len(files)} 个文件: [{category}] (GPU串行模式)")

    results = []
    # 移除多进程 Pool，改为直接循环
    for f in files:
        res = process_file_gpu((f, output_dir, category))
        results.append(res)

    success_count = sum(1 for r in results if r[0])
    logger.info(f"类别 [{category}] 处理完成: {success_count}/{len(files)}")
    return results


def main():
    logger.info(f"运行环境: {'GPU (cuDF)' if GPU_AVAILABLE else 'CPU (Pandas)'}")

    # 用于存储所有日期的所有用户，用于提取公共用户
    # 结构示例: { "2024-01-01": {"user1", "user2"}, ... }
    # 这里我们简化为收集所有的处理结果，你可以根据需要做交集(intersection)
    all_daily_users = {}

    for datatype, _ in config["datatypes"].items():
        input_dir = config["input_root"] / datatype
        output_dir = config["output_root"] / datatype

        if input_dir.exists():
            # 这里忽略了 config 中的 processes 参数
            batch_results = process_batch_gpu(input_dir, output_dir, datatype)

            # 聚合用户数据（示例逻辑：按日期归档）
            for success, date_str, cat, users in batch_results:
                if success and users:
                    # 注意：date_str 目前是文件名stem，如果是类似 "20240101_Part1" 这种
                    # 你可能需要解析出纯日期。假设文件名即日期或包含日期。
                    # 这里简单演示合并到总列表
                    pass
        else:
            logger.warning(f"目录不存在: {input_dir}")

    logger.info("所有任务处理完成！")


if __name__ == "__main__":
    main()
