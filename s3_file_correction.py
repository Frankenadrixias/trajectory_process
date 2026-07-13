# -*- coding: utf-8 -*-
"""
@File    :   s3_file_correction.py
@Time    :   2026/07/10 16:16
@Author  :   FriedrichXR
@Version :   1.0
@Contact :   2249307370@qq.com
@Desc    :   data_v4: Merge_v1文件修正
"""
import logging
import warnings
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm

# ================ 配置数据 =================
BASE_DIR = r"H:\data_v4\厦门市\Merge_v1"  # 输入路径
START_DATE = "2022-07-01"  # 起始日期
END_DATE = "2022-12-31"  # 结束日期
MAX_CPU_USAGE = 0.9  # 最大cpu使用比例 (90%)
NUM_WORKERS = int(MAX_CPU_USAGE * cpu_count())  # 根据 CPU 核心数和内存设定进程数
priority_map = {"wifi": 2, "scene": 3, "timing": 1}  # 数据类型权重表
# ========== 一般只需要修改以上内容 ==========

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# 处理单个 parquet 文件的函数
def process_group(file_path):

    try:
        # 读取数据
        df = pd.read_parquet(file_path, engine='pyarrow', dtype_backend='pyarrow')

        # 添加 priority 列
        df["priority"] = df["ftype"].map(priority_map).fillna(0).astype("int16[pyarrow]")

        # 删除经纬度异常数据
        df = df.dropna(subset=['lng', 'lat'])
        df = df[(df['lng'] != 0) & (df['lat'] != 0)]

        # 过滤非法起始结束时间
        df = df.dropna(subset=['starttime'])
        df["endtime"] = df["endtime"].fillna(df["starttime"])
        mask = df['endtime'] < df['starttime']
        df.loc[mask, ['starttime', 'endtime']] = df.loc[mask, ['endtime', 'starttime']].values

        # 针对 Timing 数据，单独赋予 1 分钟默认持续时长
        timing_mask = df["ftype"] == "timing"
        df.loc[timing_mask, "endtime"] = df.loc[timing_mask, "starttime"] + pd.Timedelta(minutes=1)

        # 写回同一文件
        df.to_parquet(file_path, engine="pyarrow", compression="zstd", index=False)

        return file_path, True, None

    except Exception as e:
        return file_path, False, str(e)


# 并行处理主函数（天内并行，天间串行）
# - input_dir: 输入根目录 (如 H:\data_v4\city\Merge_v1)
# - start_date: 起始日期 (YYYY-MM-DD)，可选
# - end_date: 结束日期 (YYYY-MM-DD)，可选
# - num_workers: 最大并行进程数，None 表示使用 CPU 核心数
def run_parallel(input_dir, start_date=None, end_date=None, num_workers=None):

    # 获取并排序日期文件夹
    input_root = Path(input_dir)
    all_day_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])

    # 日期范围筛选
    day_dirs = []
    for d in all_day_dirs:
        day_str = d.name
        if start_date and day_str < start_date:
            continue
        if end_date and day_str > end_date:
            continue
        day_dirs.append(d)

    if not day_dirs:
        logger.info(f"未在指定范围 [{start_date} 至 {end_date}] 内找到日期文件夹。")
        return

    logger.info(f"待处理日期: {day_dirs[0].name} 至 {day_dirs[-1].name}，共 {len(day_dirs)} 天")
    logger.info(f"使用 {num_workers} 个线程并行处理")
    total_success = 0
    total_fail = 0

    # 天间串行
    for day_dir in day_dirs:
        day_name = day_dir.name

        # 扫描当天所有 parquet 文件
        all_files = list(day_dir.glob("*.parquet"))

        if not all_files:
            logger.info(f"日期 {day_name}: 没有找到 parquet 文件，跳过。")
            continue

        logger.info(f"开始处理 {day_name}，共 {len(all_files)} 个文件")

        # 天内并行
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_group, all_files),
                total=len(all_files),
                desc=f"  {day_name} 进度",
                leave=True
            ))

        # 统计结果
        day_success = sum(1 for _, s, _ in results if s)
        day_fail = sum(1 for _, s, _ in results if not s)
        total_success += day_success
        total_fail += day_fail

        logger.info(f"{day_name} 完成: 成功 {day_success} 个, 失败 {day_fail} 个")

        # 打印失败详情
        if day_fail > 0:
            for file_path, _, error in results:
                if error:
                    logger.warning(f"  {file_path} -> {error}")

    logger.info(f"全部完成! 总计成功: {total_success}, 失败: {total_fail}")


if __name__ == "__main__":
    run_parallel(BASE_DIR, START_DATE, END_DATE, NUM_WORKERS)
