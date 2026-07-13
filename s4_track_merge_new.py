# -*- coding: utf-8 -*-
"""
@File    :   s4_track_merge_new.py
@Time    :   2026/01/20 17:13
@Author  :   huangsh, FriedrichXR
@Version :   2.0
@Contact :   1126456109@qq.com，2249307370@qq.com
@Desc    :   将聚类、合并好的轨迹点去除异常点和跳跃点，并进行逐小时插值
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
START_DATE = "2022-10-01"  # 起始日期
END_DATE = "2022-12-31"  # 结束日期
# ========== 一般只需要修改以上内容 ==========

# 参数配置
H3_RESOLUTION = 9
MIN_VALID_HOURS = 5
MAX_CPU_USAGE = 0.8  # 最大cpu使用比例 (90%)
NUM_WORKERS = max(int(MAX_CPU_USAGE * cpu_count()), 12)  # 根据 CPU 核心数和内存设定进程数
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
    speed_thresh_ms: float = 100  # 异常速度阈值 100m/s = 360km/h
    dist_thresh_m: float = 50000  # 位移阈值 = 50km
    far_thresh_m: float = 100000  # 远离中心阈值 = 100km
    max_far_hours: float = 3  # 异常段最大持续时间 = 4h


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
def clean_user_track(df: pd.DataFrame, params: JumpFilterParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param df: 原始的data_v1数据DataFrame
    :param params: 判断异常点的参数类
    :return: cleaned_df: 去除异常点后的DataFrame, df: 含is_outlier标记的完整DataFrame
    """

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
    # 只要用户改变 或 远离状态改变，均强制切分并生成全局唯一的段 ID，阻断多用户粘连
    state_change = df["Userid"].ne(df["Userid"].shift()) | df["is_far"].ne(df["is_far"].shift())
    df["far_seg"] = state_change.astype("int64[pyarrow]").cumsum()

    far_mask = df["is_far"]  # 仅对 is_far 为 True 的段进行统计
    if not far_mask.any():
        # 情况 A: 如果全天都没有远离市中心的点
        df["far_outlier"] = False
    else:
        # 情况 B: 存在远离点，进行段聚合计算
        seg_stats = (
            df[far_mask]
            .groupby("far_seg")  # 此时各用户的段 ID 是全局唯一的，可以直接 groupby
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
    drop_cols_1 = ["dist", "speed", "dist_center", "is_far", "far_seg"]
    df.drop(columns=drop_cols_1, inplace=True)
    drop_cols_2 = ["point_outlier", "far_outlier", "is_outlier"]
    cleaned_df = df.loc[~df["is_outlier"]].drop(columns=drop_cols_2, errors="ignore")

    return cleaned_df, df


# 生成按小时打断后的明细记录
# 功能：仅在跨整点时拆分时间段，不跨整点的记录完整保留
def hour_break(df: pd.DataFrame, day_str: str) -> pd.DataFrame:

    # 1. 转换目标日期边界，利用原生时间戳进行极速范围判断
    target_start = pd.to_datetime(day_str)
    target_end = target_start + pd.Timedelta(days=1)

    # 2. 提前过滤，只保留与目标日期有重叠的记录，大幅减小后续计算规模
    df = df[(df["starttime"] < target_end) & (df["endtime"] >= target_start)].copy()
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    # 3. 计算每个区间的起点和终点所在的整点
    s_hour = df["starttime"].dt.floor("h")
    e_hour = df["endtime"].dt.floor("h")

    # 4. 计算每个区间跨越的整点数（即需要拆分出的额外段数）
    s_hour_np = s_hour.to_numpy(copy=False)
    e_hour_np = e_hour.to_numpy(copy=False)

    # 转换为秒再除以 3600 秒（1小时），得到跨越的整点小时差
    diff_seconds = (e_hour_np - s_hour_np).astype("timedelta64[s]").astype(np.int64)
    num_splits = (diff_seconds // 3600).astype(np.int32)

    # 5. 行复制：利用 repeat 快速复制行，彻底消灭 .explode()
    repeats = num_splits + 1
    expanded = df.loc[df.index.repeat(repeats)].reset_index(drop=True)

    # 6. 向量化生成组内序号 j (0, 1, 2...)，完全规避 GroupBy
    n = repeats.sum()
    r_cumsum = np.cumsum(repeats)
    out = np.ones(n, dtype=np.int32)
    out[0] = 0
    if len(repeats) > 1:
        out[r_cumsum[:-1]] = 1 - repeats[:-1]
    j = np.cumsum(out)

    # 7. 向量化计算拆分后的区间起止点（极速数学填充）
    orig_start = expanded["starttime"].to_numpy(copy=False)
    orig_end = expanded["endtime"].to_numpy(copy=False)
    repeated_s_hour = np.repeat(s_hour_np, repeats)

    j_timedelta = j.astype("timedelta64[h]")
    next_j_timedelta = (j + 1).astype("timedelta64[h]")

    # 用 max/min 算术直接算出拆分时间，速度达到硬件极限
    expanded["starttime"] = np.maximum(orig_start, repeated_s_hour + j_timedelta)
    expanded["endtime"] = np.minimum(orig_end, repeated_s_hour + next_j_timedelta)

    # 8. 过滤只属于目标日期的行 (使用原生时间戳快速比对，替代慢速的 .dt.date == target)
    expanded = expanded[(expanded["starttime"] >= target_start) & (expanded["starttime"] < target_end)].copy()
    if expanded.empty:
        return pd.DataFrame(columns=df.columns)

    # 9. 计算停留时间（分钟），使用最轻量的 NumPy 减法
    expanded["staytime"] = (
            (expanded["endtime"].to_numpy(copy=False) - expanded["starttime"].to_numpy(copy=False))
            .astype("timedelta64[s]")
            .astype(np.float64) / 60.0
    )

    # 10. 提取 hour，排序归整
    expanded["hour"] = expanded["starttime"].dt.hour + 1
    expanded = expanded.sort_values(["Userid", "starttime"], kind="mergesort").reset_index(drop=True)
    expanded = expanded[["Userid", "hour", "lng", "lat", "x", "y", "starttime", "endtime", "staytime", "priority"]]

    return expanded


# 对按小时打断的轨迹数据执行用户过滤、24小时时空骨架线性插值、以及H3编码还原。
def interpolate_data(df: pd.DataFrame, day_str: str) -> pd.DataFrame:
    """
    :param df: 膨胀后的轨迹 DataFrame，需包含 ['Userid', 'hour', 'priority', 'staytime', 'lng', 'lat', 'x', 'y']
    :param day_str: 当前 Parquet 文件对应的日期字符串，例如 '2024-01-01'
    :return: 经过24h插值和H3网格转换后的规范轨迹 DataFrame
    """

    # 步骤 1: 筛选与去重（每个用户在每个小时内，只保留精度最高且时间最长的代表点）
    data_sorted = df.sort_values(by=["Userid", "hour", "priority", "staytime"],
                                 ascending=[True, True, False, False])
    data = data_sorted.drop_duplicates(subset=["Userid", "hour"], keep="first").reset_index(drop=True)

    # 筛选出满足阈值的有效用户
    user_hours = data["Userid"].value_counts()
    valid_users = user_hours[user_hours >= MIN_VALID_HOURS].index
    data = data[data["Userid"].isin(valid_users)].reset_index(drop=True)

    # 步骤 2: 向量化创建 24 小时骨架
    all_users = data["Userid"].unique()
    full_combinations = pd.MultiIndex.from_product([all_users, range(1, 25)], names=["Userid", "hour"]).to_frame(
        index=False
    ).astype({
        "Userid": "string[pyarrow]",
        "hour": "int16[pyarrow]"
    })

    # 步骤 3: 骨架与观测数据对齐
    merged = pd.merge(
        full_combinations,
        data[["Userid", "hour", "lng", "lat", "x", "y"]],
        on=["Userid", "hour"],
        how="left",
    )
    # 如果经度缺失，说明该小时是新填充的骨架（属于插值/移动中数据）
    is_missing = merged["lng"].isna()

    # 步骤 4: 向量化时空约束线性插值
    # 构造内存级局部辅助 Series，记录原始有观测的小时
    observed_hour = pd.Series(
        np.where(~is_missing, merged["hour"], np.nan),
        index=merged.index
    )

    # 对局部 Series 执行极速分组前向/后向填充时间锚点
    grp = merged.groupby("Userid")  # 全局分组对象
    last_hour = observed_hour.groupby(merged["Userid"]).ffill()
    next_hour = observed_hour.groupby(merged["Userid"]).bfill()

    # 计算插值比例权重
    denom = next_hour - last_hour
    denom_safe = np.where(denom == 0, 1.0, denom)  # 防除以 0 警告
    weight = (merged['hour'] - last_hour) / denom_safe

    # 计算 4 个数值列的线性插值
    numeric_cols = ["lng", "lat", "x", "y"]
    for col in numeric_cols:
        last_val = grp[col].ffill()
        next_val = grp[col].bfill()
        merged[col] = last_val + weight * (next_val - last_val)

    # 补齐首尾两端边界空白
    merged[numeric_cols] = grp[numeric_cols].ffill()
    merged[numeric_cols] = merged.groupby("Userid")[numeric_cols].bfill()

    # 步骤 5: 向量化经纬度 H3 网格转换
    # 提取唯一坐标并过滤空值，安全防范 NAType 报错
    coords = merged[["lat", "lng"]].dropna(subset=["lat", "lng"]).drop_duplicates()

    # 仅在非重复坐标集合上运行 H3 转换，极大减少计算开销
    coords["h3_grid"] = [
        h3.latlng_to_cell(lat, lng, H3_RESOLUTION)
        for lat, lng in zip(coords["lat"].to_numpy(copy=False), coords["lng"].to_numpy(copy=False))
    ]
    # 构建高兼容性的哈希映射字典
    h3_dict = dict(
        zip(
            zip(coords["lat"].to_numpy(copy=False), coords["lng"].to_numpy(copy=False)),
            coords["h3_grid"].to_numpy(copy=False)
        )
    )
    # 将结果广播回大表，彻底规避缓慢的 Pandas 浮点 merge 关联
    merged["h3_grid"] = [
        h3_dict.get((lat, lng))
        for lat, lng in zip(merged["lat"].to_numpy(copy=False), merged["lng"].to_numpy(copy=False))
    ]

    # 步骤 6: 向量化计算最终的 is_moving
    # 提取原始观测的 H3 网格（插值点设为 None）
    original_h3 = merged["h3_grid"].where(~is_missing)

    # 获取每个插值网格的前向已知 H3 和后向已知 H3
    last_h3 = pd.Series(original_h3).groupby(merged["Userid"]).ffill()
    next_h3 = pd.Series(original_h3).groupby(merged["Userid"]).bfill()

    # 边界对齐：对于首尾两端的边界空缺，由于是平铺填充未发生位移，使其比对结果为一致（False）
    last_h3_filled = last_h3.fillna(next_h3)
    next_h3_filled = next_h3.fillna(last_h3)

    # 判定最终的 is_moving：只有该时段是插值生成的（is_missing），且前后 H3 网格不一致时才标记为 True
    merged["is_moving"] = (is_missing & (last_h3_filled != next_h3_filled)).astype("bool[pyarrow]")

    # 步骤 7: 日期广播与字段格式化
    merged["date"] = pd.to_datetime(day_str).date()
    result = merged[["Userid", "date", "hour", "lng", "lat", "x", "y", "h3_grid", "is_moving"]]

    return result


# 主处理流程：基于Merge_v1生成Merge_v1a（去除异常点）、Merge_v2a（按小时打断）、Merge_v2b（逐小时插值）
# 参数：接收一个元组 (input_file, out0, outA, outB)
def main_process(file_tuple):

    # 读取数据
    day_str, input_file, outfile_0, outfile_a, outfile_b = file_tuple
    data = pd.read_parquet(input_file, engine="pyarrow", dtype_backend="pyarrow")
    data = data.sort_values(["Userid", "starttime"], kind="mergesort").reset_index(drop=True)
    if "priority" not in data.columns:
        data["priority"] = data["ftype"].map(priority_map).fillna(0).astype("int16[pyarrow]")

    # 1. 去除异常点
    (data_cleaned, data) = clean_user_track(data, params=JumpFilterParams())
    data.to_parquet(outfile_0, engine="pyarrow", compression="zstd", index=False)

    # 2. 按小时打断记录
    data_expanded = hour_break(data_cleaned, day_str)
    try:
        data_expanded["staytime"] = data_expanded["staytime"].astype("float[pyarrow]")
        data_expanded["hour"] = data_expanded["hour"].astype("int16[pyarrow]")
    except Exception as trans_e:
        logger.warning(f"类型转换警告: {trans_e}")  # 如果转换失败，程序继续运行，但保持原类型
    data_expanded.to_parquet(outfile_a, engine="pyarrow", compression="zstd", index=False)

    # 3. 时间插值
    data_result = interpolate_data(data_expanded, day_str)
    try:
        numeric_cols = ["lng", "lat", "x", "y"]  # 将四列核心空间坐标统一转换为指定精度的 PyArrow 浮点数
        data_result = data_result.astype({col: "float[pyarrow]" for col in numeric_cols})
    except Exception as trans_e:
        logger.warning(f"类型转换警告: {trans_e}")  # 如果转换失败，程序继续运行，但保持原类型
    data_result.to_parquet(outfile_b, engine="pyarrow", compression='zstd', index=False)

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

            tasks.append((day_name, str(f_path), str(out0), str(outA), str(outB)))

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
