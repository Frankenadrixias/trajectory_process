# -*- coding: utf-8 -*-
"""
@File    :   s4_track_merge_new.py
@Time    :   2026/01/20 17:13
@Author  :   huangsh, FriedrichXR
@Version :   2.0
@Contact :   1126456109@qq.com，2249307370@qq.com
@Desc    :   对同一天用户的轨迹进行空间聚类、按聚类进行合并，保留最长区间
"""
import h3
import warnings
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from pyproj import Transformer, CRS
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

"""
输入：data_v4/Merge_v1
输出：data_v4/Merge_v1a、Merge_v2a、Merge_v2b

输入目录结构要求：
F:/data_v4/北京24/Merge_v1/
    ├── 2024-01-01/
    │   ├── group_0.parquet
    │   ├── group_1.parquet
    │   └── rest.parquet
    ├── 2024-01-02/
    │   └── ...
    └── ...

输出目录结构：
F:/data_v4/北京24/Merge_v1a(Merge_v2a、Merge_v2b)/
    ├── 2024-01-01/
    │   ├── group_0.parquet
    │   └── group_1.parquet
    ├── 2024-01-02/
    │   └── ...
    └── ...
"""

# ================ 配置数据 =================
# 输入输出路径配置
INPUT_DIR = r"H:\data_v4\厦门市\Merge_v1"
OUTPUT_ROOT = r"H:\data_v4\厦门市"
START_DATE = "2022-10-03"  # 起始日期
END_DATE = "2022-10-03"  # 结束日期
# ========== 一般只需要修改以上内容 ==========

# 参数配置
H3_RESOLUTION = 9
MAX_CPU_USAGE = 0.8  # 最大cpu使用比例 (90%)
NUM_WORKERS = min(int(MAX_CPU_USAGE * cpu_count()), 12)  # 根据 CPU 核心数和内存设定进程数
priority_map = {"wifi": 2, "scene": 3, "timing": 1}  # 数据类型权重表

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# 指定 Albers 等积投影坐标系
albers_crs = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=105 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(albers_crs), always_xy=True)


# 异常判别类
@dataclass
class JumpFilterParams:
    speed_thresh_ms: float = 100  # 异常速度阈值 = 360km/h
    dist_thresh_m: float = 50000  # 位移阈值 = 50km
    far_thresh_m: float = 100000  # 远离中心阈值 = 100km
    max_far_hours: float = 6  # 异常段最大持续时间 = 6h


# 返回 (start, end) 内部的整点时间列表
# 跨整点 -> 返回空列表
def get_split_points(start, end):
    if pd.isna(end) or start >= end:
        return []

    first = start.ceil("H")
    if first >= end:
        return []

    return list(pd.date_range(first, end, freq="H", inclusive="left"))


# 基于运动特征 + 距离市中心的规则，按 Userid 识别并清洗剔除轨迹异常跳跃点
# 返回：
# - cleaned_df: 去除异常点后的DataFrame
# - df: 含is_outlier标记的完整DataFrame
def clean_user_track(df, params: JumpFilterParams) -> tuple[pd.DataFrame, pd.DataFrame]:

    # 1. 计算位移和速度
    group_obj = df.groupby("Userid", sort=False)
    df["dist"] = np.sqrt(group_obj["x"].diff() ** 2 + group_obj["y"].diff() ** 2)
    dt = group_obj["starttime"].diff().dt.total_seconds()
    valid_dt_mask = (dt > 0).fillna(False)
    df["speed"] = np.where(valid_dt_mask, df["dist"] / dt, np.inf)

    # 2. 点级异常判定
    df["point_outlier"] = (df["speed"] > params.speed_thresh_ms) | (df["dist"] > params.dist_thresh_m)
    df.loc[df["dist"].isna(), "point_outlier"] = False  # 起始点通常没有 diff，不作为异常

    # 3. 常住地距离判定
    home = df.groupby("Userid", sort=False)[["x", "y"]].transform("median")
    df["dist_center"] = np.sqrt((df["x"] - home["x"]) ** 2 + (df["y"] - home["y"]) ** 2)
    df["is_far"] = df["dist_center"] > params.far_thresh_m

    # 4. 远离段逻辑判定
    df["far_seg"] = df["is_far"].ne(df["is_far"].shift()).cumsum()
    far_mask = df["is_far"]  # 仅对 is_far 为 True 的段进行统计
    if not far_mask.any():
        # 情况 A: 如果全天都没有远离市中心的点
        df["far_outlier"] = False
    else:
        # 情况 B: 存在远离点，进行段聚合计算
        seg_stats = (
            df[far_mask]
            .groupby("far_seg")
            .agg(
                start_time=("starttime", "first"),
                end_time=("starttime", "last"),
                max_speed=("speed", "max"),
                max_dist=("dist", "max"),
            )
        )
        seg_stats["duration"] = (seg_stats["end_time"] - seg_stats["start_time"]).dt.total_seconds()

        # 远离段异常判别，映射回原 df
        seg_stats["far_outlier"] = (seg_stats["duration"] < params.max_far_hours * 3600) & (
            seg_stats["max_speed"] > params.speed_thresh_ms
        )
        df["far_outlier"] = df["far_seg"].map(seg_stats["far_outlier"].fillna(False))

    # 5. 综合异常标签
    df["is_outlier"] = (df["point_outlier"] | df["far_outlier"]).fillna(False)

    # 返回清洗结果
    drop_cols = ["dist", "speed", "point_outlier", "dist_center", "is_far", "far_seg", "far_outlier", "is_outlier"]
    cleaned_df = df.loc[~df["is_outlier"]].drop(columns=drop_cols, errors="ignore")

    return cleaned_df, df


# 生成按小时打断后的明细记录
# 功能：仅在跨整点时拆分时间段，不跨整点的记录完整保留
def hour_break(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 生成每行的“内部整点切分点”
    df["split_points"] = [get_split_points(s, e) for s, e in zip(df["starttime"], df["endtime"])]

    # 2. 构造每行的区间端点：[start, ..., end]，不跨整点 -> [start, end]后面只生成 1 行
    df["bounds"] = [[s] + pts + [e] for s, pts, e in zip(df["starttime"], df["split_points"], df["endtime"])]

    # 3. 生成区间对 (start_i, end_i)
    df["intervals"] = [list(zip(b[:-1], b[1:])) for b in df["bounds"]]

    # 4. 展开为多行
    expanded = df.explode("intervals", ignore_index=True)
    expanded[["starttime", "endtime"]] = pd.DataFrame(expanded["intervals"].tolist(), index=expanded.index)

    # 5. 计算停留时间（分钟）
    expanded["staytime"] = (expanded["endtime"] - expanded["starttime"]).dt.total_seconds() / 60

    # 6. 清理中间字段，导出
    expanded = expanded.drop(columns=["split_points", "bounds", "intervals"])
    expanded = expanded.sort_values(["Userid", "starttime"], kind="mergesort").reset_index(drop=True)
    expanded["hour"] = expanded["starttime"].dt.hour + 1

    return expanded


# 主处理流程：基于Merge_v1生成Merge_v1a（去除异常点）、Merge_v2a（按小时打断）、Merge_v2b（逐小时插值）
# 参数：接收一个元组 (input_file, out0, outA, outB)
def main_process(file_tuple):

    # 读取数据
    input_file, outfile_0, outfile_a, outfile_b = file_tuple
    data = pd.read_parquet(input_file, engine="pyarrow", dtype_backend="pyarrow")
    data = data.sort_values(["Userid", "starttime"], kind="mergesort").reset_index(drop=True)
    if "priority" not in data.columns:
        data["priority"] = data["ftype"].map(priority_map).fillna(0).astype("int16[pyarrow]")

    # 1. 去除异常点
    (data_cleaned, data) = clean_user_track(data, params=JumpFilterParams())
    try:
        data["speed"] = data["speed"].astype("float[pyarrow]")
        data["dist_center"] = data["dist_center"].astype("float[pyarrow]")
    except Exception as trans_e:
        # 如果转换失败，程序继续运行，但保持原类型
        logger.warning(f"类型转换警告: {trans_e}")
    data.to_parquet(outfile_0, engine="pyarrow", compression='zstd', index=False)

    # 2. 按小时打断记录
    data_expanded = hour_break(data_cleaned)
    try:
        data_expanded["staytime"] = data_expanded["staytime"].astype("float[pyarrow]")
        data_expanded["hour"] = data_expanded["hour"].astype("int16[pyarrow]")
    except Exception as trans_e:
        # 如果转换失败，程序继续运行，但保持原类型
        logger.warning(f"类型转换警告: {trans_e}")
    data_expanded.to_parquet(outfile_a, engine="pyarrow", compression='zstd', index=False)

    # 3. 时间插值
    # 步骤1: 每个用户在每个小时内，只保留即精度最高且时间最长的代表点
    data_sorted = data_expanded.sort_values(by=["Userid", "hour", "priority", "staytime"],
                                            ascending=[True, True, False, False])  # 场景 > wifi > 定位
    data = data_sorted.drop_duplicates(subset=["Userid", "hour"], keep="first").reset_index(drop=True)

    # 提前在小表上提取日期，建立哈希映射表，避免后续在大表上进行慢速分组
    data["date"] = data["starttime"].dt.date
    user_to_date = data.set_index("Userid")["date"]

    # 步骤2: 向量化创建完整小时骨架序列
    all_users = data["Userid"].unique()
    full_combinations = pd.MultiIndex.from_product([all_users, range(1, 25)], names=["Userid", "hour"]).to_frame(
        index=False
    ).astype({
        "Userid": "string[pyarrow]",
        "hour": "int16[pyarrow]"
    })

    # 步骤3: 合并完整骨架序列与实际数据
    merged = pd.merge(
        full_combinations,
        data[["Userid", "hour", "lng", "lat", "x", "y", "starttime"]],
        on=["Userid", "hour"],
        how="left",
    )
    # 在执行线性插值前，如果 starttime 缺失，说明该小时是新填充的骨架（即属于插值数据）
    merged["is_moving"] = merged["starttime"].isna().astype("bool[pyarrow]")

    # 步骤4: 向量化插值处理
    '''
    numeric_cols = ["lng", "lat", "x", "y"]
    merged[numeric_cols] = merged.groupby("Userid")[numeric_cols].transform(
        lambda g: g.interpolate(method="linear", limit_direction="both")
    )
    '''

    # 构造辅助列记录有观测的小时
    merged['observed_hour'] = np.where(merged['starttime'].notna(), merged['hour'], np.nan)  
    grp = merged.groupby("Userid")  # 全局分组对象

    last_hour = grp['observed_hour'].ffill()
    next_hour = grp['observed_hour'].bfill()
    denom = next_hour - last_hour
    denom_safe = np.where(denom == 0, 1.0, denom)  # 防除以0
    weight = (merged['hour'] - last_hour) / denom_safe
    numeric_cols = ["lng", "lat", "x", "y"]
    # 极速向量化计算 4 个数值列的线性插值
    for col in numeric_cols:
        last_val = grp[col].ffill()
        next_val = grp[col].bfill()
        merged[col] = last_val + weight * (next_val - last_val)

    # 步骤5: 将经纬度转换为H3网格索引
    coords = merged[["lat", "lng"]].drop_duplicates()
    coords["h3_grid"] = [
        h3.latlng_to_cell(lat, lng, H3_RESOLUTION) for lat, lng in zip(coords["lat"].values, coords["lng"].values)
    ]
    merged = merged.merge(coords, on=["lat", "lng"], how="left")

    # 步骤6: 添加日期列并选择输出字段
    merged["date"] = merged["Userid"].map(user_to_date)
    result = merged[["Userid", "date", "hour", "lng", "lat", "x", "y", "h3_grid", "is_moving"]]

    # 保存结果
    result.to_parquet(outfile_b, engine="pyarrow", compression='zstd', index=False)
    return


# --- 并行执行入口 ---
# input_dir: 输入根目录文件夹
# output_root: 输出根目录文件夹
# start_date: 字符串 '2024-01-01'，为 None 则不限制开始
# end_date: 字符串 '2024-01-10'，为 None 则不限制结束
# max_workers: 并行进程数，None 则使用 CPU 核心数
def run_parallel(input_dir, output_root, start_date=None, end_date=None, max_workers=None):

    # 1. 定义输出目录结构
    output_base_paths = {
        "raw_cleaned": Path(output_root) / "Merge_v1a",
        "hour_split": Path(output_root) / "Merge_v2a",
        "interpolated": Path(output_root) / "Merge_v2b",
    }

    input_root = Path(input_dir)
    # 获取并排序日期文件夹
    all_day_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])

    # 2. 日期范围筛选
    day_dirs = []
    for d in all_day_dirs:
        day_str = d.name
        # 字符串直接比较日期是安全的 (YYYY-MM-DD 格式)
        if start_date and day_str < start_date:
            continue
        if end_date and day_str > end_date:
            continue
        day_dirs.append(d)

    if not day_dirs:
        logger.info(f"未在指定范围 [{start_date} 至 {end_date}] 内找到日期文件夹。")
        return

    logger.info(f"待处理日期: {day_dirs[0].name} 至 {day_dirs[-1].name}，共 {len(day_dirs)} 天")

    # --- 天间串行 ---
    for day_dir in day_dirs:
        day_name = day_dir.name

        # 为当天创建输出目录
        current_day_outputs = {}
        for key, base_path in output_base_paths.items():
            target_dir = base_path / day_name
            target_dir.mkdir(parents=True, exist_ok=True)
            current_day_outputs[key] = target_dir

        # 3. 扫描文件并检查是否已处理过
        all_files = list(day_dir.glob("*.parquet"))
        tasks = []
        skipped_count = 0

        for f_path in all_files:
            file_name = f_path.name

            # 定义三个输出文件的具体路径
            out0 = current_day_outputs["raw_cleaned"] / file_name
            outA = current_day_outputs["hour_split"] / file_name
            outB = current_day_outputs["interpolated"] / file_name

            # 核心逻辑：检查最终产物 (outB) 是否已存在
            if outB.exists():
                skipped_count += 1
                continue

            tasks.append((str(f_path), str(out0), str(outA), str(outB)))

        if not tasks:
            logger.info(f"日期 {day_name}: 所有文件已存在，跳过。")
            continue

        if skipped_count > 0:
            logger.info(
                f"日期 {day_name}: 发现 {len(all_files)} 个文件，跳过已处理的 {skipped_count} 个，剩余 {len(tasks)} 个。"
            )
        else:
            logger.info(f"\n日期 {day_name}: 准备处理 {len(tasks)} 个文件")

        # --- 天内并行 ---
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(main_process, tasks), total=len(tasks), desc=f"  {day_name} 进度", leave=True))

    logger.info("\n[完成] 指定范围内的任务已全部处理完毕。")


if __name__ == "__main__":
    run_parallel(
        input_dir=INPUT_DIR,
        output_root=OUTPUT_ROOT,
        start_date=START_DATE,
        end_date=END_DATE,
        max_workers=NUM_WORKERS
    )
