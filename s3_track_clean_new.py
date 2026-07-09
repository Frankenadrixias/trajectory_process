# -*- coding: utf-8 -*-
"""
@File    :   s3_track_clean_new.py
@Time    :   2026/01/15 21:48
@Author  :   huangsh, FriedrichXR
@Version :   2.0
@Contact :   1126456109@qq.com，2249307370@qq.com
@Desc    :   对同一天用户的轨迹进行空间聚类、按聚类进行合并，保留最长区间
"""
import os
import h3
import time
import logging
import warnings
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from pyproj import Transformer, CRS
from sklearn.cluster import DBSCAN
from multiprocessing import Pool

"""
输入：data_v4/Merge_v0
输出：data_v4/Merge_v1

输入目录结构要求：
F:/data_v4/北京24/Merge_v0/
    ├── 2024-01-01/
    │   ├── group_0.parquet
    │   ├── group_1.parquet
    │   └── rest.parquet
    ├── 2024-01-02/
    │   └── ...
    └── ...

输出目录结构：
F:/data_v4/北京24/Merge_v1/
    ├── 2024-01-01/
    │   ├── group_0.parquet
    │   └── group_1.parquet
    ├── 2024-01-02/
    │   └── ...
    └── ...
    
性能：8进程并行，平均480s/文件，20w用户；较原始代码提速150%
"""

# ================= 配置数据 =================
# 输入输出路径配置
INPUT_DIR = r"H:\data_v4\厦门市\Merge_v0"
OUTPUT_DIR = r"H:\data_v4\厦门市\Merge_v1"
START_DATE = "2022-12-01"  # 起始日期
END_DATE = "2022-12-31"  # 结束日期

# 参数配置
NUM_WORKERS = 11  # 进程数
GROUP_METHOD = "dbscan"  # 可选 "dbscan"（通过DBSCAN聚类plabel合并） 或 "h3"（直接按H3网格合并）
H3_RESOLUTION = 9  # H3分辨率，可根据需要调整
DBSCAN_MODEL = DBSCAN(eps=100, min_samples=2, algorithm="ball_tree", n_jobs=-1)  # DBSCAN参数
priority_map = {"wifi": 2, "scene": 3, "timing": 1}  # 数据类型权重表
# =========== 一般仅需要修改以上内容 ===========

# 指定 Albers 等积投影坐标系
albers_crs = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=105 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(albers_crs), always_xy=True)

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# 创建目录（如果不存在）
def create_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# 优化后的时间段合并函数（O(n)复杂度）
# 输入参数：含有位置属性列（如聚类类别、所属网格等）的df、位置属性列名、数据优先级属性、容差时间
# 输出参数：记录合并后的df
def merge_time(df: pd.DataFrame, col_label: str, priority_col: str = "priority", time_gap: int = 5) -> pd.DataFrame:
    if df.empty:
        return df

    # 一次性排序：group + time，确保每个组内的时间是顺序的
    df = df.sort_values([col_label, "starttime"], kind="mergesort").reset_index(drop=True)
    group_vals = df[col_label].to_numpy(copy=False)
    starts = df["starttime"].to_numpy(copy=False)
    ends = df["endtime"].to_numpy(copy=False)
    priorities = df[priority_col].to_numpy(copy=False)  # 优先级数组
    length = len(df)
    gap_threshold = np.timedelta64(time_gap, 'm')

    # 预分配结果数组
    out_idx = np.empty(length, dtype=np.int32)
    merged_starts = np.empty_like(starts)
    merged_ends = np.empty_like(ends)

    # 初始化当前区间
    current_group = group_vals[0]
    current_start = starts[0]
    current_end = ends[0]
    out_count = 0  # 对应保存新区间次数
    anchor_idx = 0  # 对应原始数据的行id

    # 单次遍历合并区间
    for i in range(1, length):
        # group 发生变化，必须强制封口
        if group_vals[i] != current_group:
            out_idx[out_count] = anchor_idx
            merged_starts[out_count] = current_start
            merged_ends[out_count] = current_end
            out_count += 1

            # 重置到新 group
            current_group = group_vals[i]
            current_start = starts[i]
            current_end = ends[i]
            anchor_idx = i
            continue

        # 同一 group，执行区间合并逻辑（含容差判定）
        if starts[i] <= current_end + gap_threshold:
            # 扩展当前区间
            if ends[i] > current_end:
                current_end = ends[i]
            # 合并时，如果当前记录的定位源优先级更高，则将属性指向该行
            if priorities[i] > priorities[anchor_idx]:
                anchor_idx = i
        else:
            # 保存当前区间
            out_idx[out_count] = anchor_idx
            merged_starts[out_count] = current_start
            merged_ends[out_count] = current_end
            out_count += 1

            # 开始新区间
            current_start = starts[i]
            current_end = ends[i]
            anchor_idx = i

    # 收尾，添加最后一个区间
    out_idx[out_count] = anchor_idx
    merged_starts[out_count] = current_start
    merged_ends[out_count] = current_end
    out_count += 1

    # 构造结果 DataFrame
    result = df.iloc[out_idx[:out_count]].reset_index(drop=True)
    result["starttime"] = merged_starts[:out_count]
    result["endtime"] = merged_ends[:out_count]

    return result


# 处理单个文件的函数
def process_single_file(args):

    input_file, output_dir = args
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    if os.path.exists(output_file):
        return

    # 读取数据
    df = pd.read_parquet(input_file, engine='pyarrow', dtype_backend='pyarrow',
                         columns=['Userid', 'lng', 'lat', 'starttime', 'endtime', 'ftype'])
    df["priority"] = df["ftype"].map(priority_map).fillna(0).astype("int16[pyarrow]")

    # 删除经纬度异常数据
    df = df.dropna(subset=['lng', 'lat'])
    df = df[(df['lng'] != 0) & (df['lat'] != 0)]

    # 过滤非法起始结束时间
    df = df.dropna(subset=['starttime'])  # 删除 starttime 为空的行
    df["endtime"] = df["endtime"].fillna(df["starttime"])  # endtime 为空则用 starttime 填充
    mask = df['endtime'] < df['starttime']  # 确保所有数据的 endtime 都不小于 starttime
    df.loc[mask, ['starttime', 'endtime']] = df.loc[mask, ['endtime', 'starttime']].values

    # 针对 Timing 数据，单独赋予 1 分钟的默认持续时长
    timing_mask = df["ftype"] == "timing"
    df.loc[timing_mask, "endtime"] = df.loc[timing_mask, "starttime"] + pd.Timedelta(minutes=1)

    # 向量化坐标转换
    lons = df["lng"].to_numpy(copy=False)
    lats = df["lat"].to_numpy(copy=False)
    xs, ys = transformer.transform(lons, lats)  # 经度在前，纬度在后
    df["x"] = xs.astype(np.float32)  # 使用float32减少内存
    df["y"] = ys.astype(np.float32)

    # 如果选择 "h3" 分组方式，提前在循环外进行全局唯一坐标 H3 转换
    if GROUP_METHOD == "h3":
        coords = df[["lat", "lng"]].drop_duplicates()
        coords["h3_grid"] = [
            h3.latlng_to_cell(lat, lng, H3_RESOLUTION) 
            for lat, lng in zip(coords["lat"].values, coords["lng"].values)
        ]
        df = df.merge(coords, on=["lat", "lng"], how="left")

    # 处理每个用户
    results = []
    for uid, user_data in df.groupby("Userid", sort=False):

        if GROUP_METHOD == "dbscan":  # DBSCAN聚类
            coords = user_data[["x", "y"]].values
            if len(coords) < DBSCAN_MODEL.min_samples:
                labels = np.arange(len(coords), dtype=np.int32)
            else:
                try:
                    labels = DBSCAN_MODEL.fit_predict(coords)
                except Exception:  # 如果失败，给所有点分配唯一标签
                    labels = np.arange(len(coords), dtype=np.int32)
            labels = labels.astype(np.int32)

            # 高效处理噪声点（向量化操作）
            noise_mask = labels == -1
            num_noise = np.count_nonzero(noise_mask)  # 使用向量化操作一次性生成所有标签
            if num_noise:
                # 为噪声点分配唯一负值
                labels[noise_mask] = np.arange(-1, -num_noise - 1, -1, dtype=np.int32)
            user_data = user_data.assign(plabel=labels)  # 赋值标签

            # 对同一聚类内的数据进行结果合并：保留最长区间
            ws_merged = merge_time(user_data, "plabel", "priority", 5)
        
        else:  # 直接按 H3 网格合并
            ws_merged = merge_time(user_data, "h3_grid", "priority", 5)

        results.append(ws_merged)

    if results:
        # 合并结果并排序
        final_df = pd.concat(results, ignore_index=True)
        final_df = final_df.sort_values(by=["Userid", "starttime"], kind="mergesort")
    else:
        # 如果没有处理结果，创建空DataFrame
        final_df = pd.DataFrame(columns=df.columns)

    # 保存结果
    final_df.to_parquet(str(output_file), engine="pyarrow", compression='zstd', index=False)
    return


# 处理单日的所有轨迹数据文件
# 参数: day (str): 日期字符串（如 "2024-01-01"）
# 返回: str: 处理后的日期字符串
def process_single_day(day):
    # 1. 构建输入/输出目录路径，获取当天的所有文件
    day_input_dir = os.path.join(INPUT_DIR, day)
    day_output_dir = create_dir(os.path.join(OUTPUT_DIR, day))

    # 2. 获取当天的公共用户parquet文件
    input_files = glob(os.path.join(day_input_dir, "group_*.parquet"))

    # 3. 添加进度条处理所有文件
    progress_bar = tqdm(
        total=len(input_files),  # 总文件数
        desc=f"Day {day}",  # 进度条描述（显示日期）
        unit="file",  # 单位（文件）
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",  # 自定义格式
        colour="green",  # 进度条颜色
        leave=True,  # 处理完成后保留进度条（设为False则清除）
    )

    # 4. 处理每个文件
    for file_path in input_files:
        task = (file_path, day_output_dir)  # 准备任务参数
        process_single_file(task)  # 处理单个文件
        progress_bar.update(1)  # 更新进度条
        progress_bar.set_postfix_str(f"Last: {os.path.basename(file_path)}")  # 显示最后处理的文件名

    # 5. 关闭进度条
    progress_bar.close()
    return day


if __name__ == "__main__":
    start_time = time.time()
    create_dir(OUTPUT_DIR)

    # 获取所有日期目录
    date_dirs = glob(os.path.join(INPUT_DIR, "*"))
    date_dirs = [d for d in date_dirs if os.path.isdir(d)]

    all_dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D").strftime("%Y-%m-%d").tolist()
    batch_dates = [os.path.basename(d) for d in all_dates]
    batch_dates.sort()  # 排序日期

    logger.info(f"Found {len(batch_dates)} days to process")
    logger.info(f"Using {NUM_WORKERS} processes")

    # 使用多进程池处理
    with Pool(processes=NUM_WORKERS) as pool:
        # 使用tqdm显示进度
        list(tqdm(pool.map(process_single_day, batch_dates), total=len(batch_dates), desc="Processing Days"))

    # 输出执行时间
    t = time.time() - start_time
    logger.info(f"Execution completed in {int(t // 3600)}h {int((t % 3600) // 60)}m {int(t % 60)}s")
