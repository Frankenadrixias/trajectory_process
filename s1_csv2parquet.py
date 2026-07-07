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
import numpy as np
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

# ================= 配置数据 =================
# 配置参数：指定输入输出目录和每一类数据开启进程数
config = {
    "input_root": Path(r"G:\厦门市(202207-202306)"),
    "output_root": Path(r"H:\data_v3\厦门市"),
    # "datatypes": {"SceneReco": 16, "Timing": 2, "WifiConnect": 20, "WifiStable": 6},
    "datatypes": {"WifiStable": 1},
}

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
# ==========================================


# 处理单个文件：读取、处理、保存为Parquet，同时收集用户数据
# 参数: args - 元组包含(文件路径, 输出目录, 类别名称)
# 返回: (成功标志, 日期字符串, 类别名称, 用户集合)
def process_file(args: Tuple[Path, Path, str]) -> Tuple[bool, str, str]:
    file_path, output_dir, category = args
    try:
        # 智能判断输出文件名
        if file_path.parent.name == category:
            output_filename = f"{file_path.stem}.parquet"
        else:
            output_filename = f"{file_path.parent.name}.parquet"
        output_file = output_dir / output_filename

        # 检查输出文件是否已存在
        if output_file.exists():
            logger.info(f"文件已存在，跳过: [{category}] {output_file.name}")
            return True, file_path.stem, category

        logger.info(f"处理: [{category}] {file_path.name}")

        # 读取文件（支持CSV、XLS和Parquet）
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path, engine='python', dtype_backend="pyarrow",
                             encoding='utf-8', on_bad_lines="skip")
        elif file_path.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(file_path, dtype_backend="pyarrow")
        else:  # 包括.parquet
            df = pd.read_parquet(file_path, engine="pyarrow", dtype_backend="pyarrow")

        # 原始数据列名：脱敏ID，经度，纬度，p_id，开始时间，结束时间
        if "index" in df.columns:
            df.drop(columns=["index"], inplace=True)

        # 2023年7月以前的数据：包含15列
        if len(df.columns) > 10:
            df = df.iloc[:, :15]
            df.columns = ["Userid", "lng", "lat", "starttime", "endtime", "duration",
                          "scene_name", "brand_name", "floor",  "province", "city",  "address",
                          "category_1", "category_2", "category_3"]

            float_cols = ["lng", "lat", "duration"]
            datetime_cols = ["starttime", "endtime"]
            string_cols = ["Userid", "scene_name", "brand_name", "floor", "province",
                           "city", "address", "category_1", "category_2", "category_3"]

            # 强制执行 PyArrow 类型的格式转换
            try:
                for col in string_cols:  # 转换为 PyArrow string 字符串格式
                    df[col] = df[col].astype("string[pyarrow]")

                for col in float_cols:  # 经纬度转换为 PyArrow float32 浮点格式
                    df[col] = df[col].astype("float32[pyarrow]")

                time_fmt = "%Y-%m-%d %H:%M:%S"  # 转换为 PyArrow timestamp 日期时间格式
                for col in datetime_cols:
                    df[col] = pd.to_datetime(df[col], format=time_fmt, errors="coerce")
                    df[col] = df[col].astype("timestamp[ns][pyarrow]")

            # 如果转换失败，程序继续运行，但保持原类型
            except Exception as trans_e:
                logger.warning(f"类型转换警告 {file_path.name}: {trans_e}")

        # 2023年7月以后的数据：包含6列
        else:
            df = df.iloc[:, :6]
            df.columns = ["Userid", "lng", "lat", "pid", "starttime", "endtime"]

            # 强制执行 PyArrow 类型的格式转换
            try:
                df["Userid"] = df["Userid"].astype("string[pyarrow]")  # ID 转为 PyArrow 字符串
                df["lng"] = df["lng"].astype("float32[pyarrow]")  # 经纬度转 float32
                df["lat"] = df["lat"].astype("float32[pyarrow]")
                df["pid"] = df["pid"].fillna(-1).astype("int32[pyarrow]")  # pid 转 int32
                time_fmt = "%Y-%m-%d %H:%M:%S"
                df["starttime"] = pd.to_datetime(df["starttime"], format=time_fmt, errors="coerce")  # 时间转换
                df["endtime"] = pd.to_datetime(df["endtime"], format=time_fmt, errors="coerce")
                df["starttime"] = df["starttime"].astype("timestamp[ns][pyarrow]")
                df["endtime"] = df["endtime"].astype("timestamp[ns][pyarrow]")

            # 如果转换失败，程序继续运行，但保持原类型
            except Exception as trans_e:
                logger.warning(f"类型转换警告 {file_path.name}: {trans_e}")

        # 处理数据：排序，去重
        df.drop_duplicates(subset=["Userid", "starttime", "endtime"], inplace=True)
        df.sort_values("Userid", kind="mergesort", inplace=True)

        # 保存为Parquet
        df.to_parquet(output_file, engine="pyarrow", compression='zstd', index=False)
        logger.info(f"完成: [{category}] {output_file.name}")

        # 释放内存
        del df
        gc.collect()
        return True, file_path.stem, category

    except Exception as e:
        logger.error(f"处理失败 {file_path.name}: {str(e)}")
        if "df" in locals():
            del df
            gc.collect()
        return False, "", ""


# 批量处理目录中的所有文件
# 参数: input_dir 输入目录；output_dir 输出目录；category 数据类别；processes 进程数
# 返回: 处理结果列表，每个元素为 (成功标志, 日期字符串, 类别名称, 用户集合)
def process_batch(input_dir: Path, output_dir: Path, category: str, processes: int = 4):
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    # 使用 rglob 递归获取所有子目录中的 CSV 和 Parquet 文件
    files = [f for f in input_dir.rglob("*") if f.is_file() and f.suffix.lower() in (".csv", ".parquet")]

    if not files:
        logger.info(f"无文件需要处理: {input_dir}")
        return []

    logger.info(f"开始处理 {len(files)} 个文件: {input_dir} [{category}]")

    # 准备参数列表
    args_list = [(f, output_dir, category) for f in files]

    # 使用进程池并行处理
    with Pool(processes=processes) as pool:
        results = pool.map(process_file, args_list)

    success_count = sum(1 for r in results if r[0])
    logger.info(f"处理完成: {success_count}/{len(files)} 个文件成功 [{category}]")
    return results


# 批量处理目录中的所有文件
# 仅针对2022-23上半年数据
def process_batch_old(input_dir: Path, output_dir: Path, category: str, processes: int = 4):
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    # 1. 获取 input_dir 下的所有日期子目录（如 2022-07-01）
    date_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if not date_dirs:
        logger.info(f"无日期目录需要处理: {input_dir}")
        return []

    # 准备参数列表
    args_list = []
    for date_dir in date_dirs:
        # 获取该目录下唯一的 CSV 文件
        csv_files = list(date_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"目录 {date_dir} 下未找到 CSV 文件，跳过")
            continue

        csv_file = csv_files[0]  # 取唯一的一个 CSV 文件
        # 目标 parquet 的完整路径，例如：output_dir/SceneReco/2022-07-13.parquet
        target_parquet = output_dir / f"{date_dir.name}.parquet"
        args_list.append((csv_file, target_parquet, category))

    if not args_list:
        logger.info(f"未找到可处理的 CSV 文件: {input_dir}")
        return []

    logger.info(f"开始处理 {len(args_list)} 个文件: {input_dir} [{category}]")

    # 使用进程池并行处理
    with Pool(processes=processes) as pool:
        results = pool.map(process_file, args_list)

    success_count = sum(1 for r in results if r[0])
    logger.info(f"处理完成: {success_count}/{len(date_dirs)} 个目录成功 [{category}]")
    return results


# 主函数
def main():
    # 收集所有处理结果
    all_results = []

    # 处理每种数据类型
    for datatype, processes in config["datatypes"].items():
        input_dir = config["input_root"] / datatype
        output_dir = config["output_root"] / datatype

        if input_dir.exists():
            results = process_batch(input_dir, output_dir, datatype, processes)
            all_results.extend(results)
        else:
            logger.warning(f"目录不存在: {input_dir}")

    logger.info("所有处理任务完成！")


if __name__ == "__main__":
    main()
