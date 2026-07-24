
import gc
import h3
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

SRC_DIR = Path(r"G:\data_v4\北京市\Merge_v2b")  # 路径配置
DST_DIR = Path(r"I:\data_v5.3\北京市\Merge_v0")
DST_MERGE_DIR = Path(r"I:\data_v5.3\北京市\Merge_v1")
DST_GRID_DIR = Path(r"I:\data_v5.3\北京市\Merge_v2")
SHAPEFILE_PATH = "data/shape/北京市.shp"
H3_SET_PATH = "data/shape/beijing_h3_set_9.pkl"

NIGHT_START = 22  # 夜间起始小时
NIGHT_END = 6  # 夜间结束小时 (不含)
DAY_START = 10  # 白天起始小时
DAY_END = 18  # 白天结束小时 (不含)
MAX_CPU_USAGE = 0.6  # 最大cpu使用比例 (90%)
NUM_WORKERS = min(int(MAX_CPU_USAGE * cpu_count()), 12)  # 根据 CPU 核心数和内存设定进程数

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# 生成用户画像文件
def generate_user_profiles(input_dir, output_file_path):

    root_path = Path(input_dir)
    # 确保输出目录存在
    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. 读取所有日期 ---
    date_folders = sorted([day_dir for day_dir in root_path.iterdir() if day_dir.is_dir()])  # 获取日期文件夹
    group_files = [f.name for f in date_folders[0].glob("group_*.parquet")]  # 确定所有的用户组文件名
    print(f"检测到 {len(date_folders)} 个日期，{len(group_files)} 个用户组")

    # 用于存放每个组聚合后的 DataFrame
    all_group_results = []

    # --- 2. 逐组循环处理 ---
    for g_file in tqdm(group_files, desc="Processing Groups"):
        group_data_list = []

        # 遍历日期文件夹，搜集该组在不同日期的表现
        for d_folder in date_folders:
            f_path = d_folder / g_file
            if f_path.exists():
                try:
                    # 使用 pyarrow 引擎读取，提高速度
                    df_tmp = pd.read_parquet(f_path, engine="pyarrow", dtype_backend="pyarrow")
                    df_tmp["home_work_ratio"] = (
                        (df_tmp["work_stay_hours"] * 2 - df_tmp["other_stay_hours"]) / 24.0
                    ).clip(-1, 1)
                    group_data_list.append(df_tmp)
                except Exception as e:
                    print(f"Error reading {f_path}: {e}")

        if not group_data_list:
            continue

        # 合并该组半年的所有记录（这里是该组几万人的半年数据）
        group_long_df = pd.concat(group_data_list, ignore_index=True)
        del group_data_list  # 及时释放内存

        # 基于是否有工作地网格，创建一个布尔标签
        group_long_df["user_type"] = np.where(group_long_df["work_h3_grid"].isna(), 0, 1)

        # --- 3. 聚合计算该组的用户画像 ---
        agg_logic = {
            "total_dist": ["mean", "std", "max"],
            "RG": ["mean", "std", "max"],
            "avg_dist_home": ["mean", "std"],  # 平均离家距离
            "dist_day": ["mean", "std", "max"],
            "RG_day": ["mean", "std", "max"],
            "dist_night": ["mean", "std", "max"],
            "RG_night": ["mean", "std", "max"],
            "home_stay_hours": ["mean", "std"],
            "work_stay_hours": ["mean", "std"],
            "other_stay_hours": ["mean", "std"],
            "home_work_ratio": "mean",
            "unique_grids": "mean",
            "commute_dist": "first",  # 静态属性
            "user_type": "first"  # 保留类型标签
        }

        # 执行聚合
        profiles = group_long_df.groupby("Userid").agg(agg_logic).reset_index()

        # 多层索引列名转换为字符串 (例如: total_dist_mean)
        profiles.columns = ["Userid"] + [f"{c[0]}_{c[1]}" for c in profiles.columns[1:]]

        # 转换数据类型
        for col in profiles.columns:
            if col not in ["Userid", "user_type_first"]:  # 跳过非数值列
                profiles[col] = profiles[col].astype("float[pyarrow]")

        # 将这一组的画像结果加入总列表
        all_group_results.append(profiles)

        # 显式清理内存，防止在循环中内存持续升高
        del group_long_df
        gc.collect()

    # --- 4. 最终合并所有组并保存 ---
    if all_group_results:
        final_profile_df = pd.concat(all_group_results, ignore_index=True)
        print(f"最终画像表构建完成，共 {len(final_profile_df)} 个用户")

        # 保存为单个 Parquet 文件
        final_profile_df.to_parquet(output_file_path, index=False, engine="pyarrow")
        print(f"结果已成功保存至: {output_file_path}")
    else:
        print("未生成任何有效画像数据。")


# 子进程任务：流式合并单个文件夹内的所有 Parquet 文件
def merge_single_day(args):
    day_dir, dst_root = args
    try:
        # 找到该文件夹下所有的 parquet 文件
        files = sorted(list(day_dir.glob("*.parquet")))
        if not files:
            return f"跳过 {day_dir.name}：未找到 parquet 文件"

        # 输出文件路径：例如 DST_DIR/2024-01-01.parquet
        out_file = dst_root / f"{day_dir.name}.parquet"
        if out_file.exists():
            return f"跳过 {day_dir.name}：合并后的文件已存在"

        # 1. 读取第一个文件的 Schema (表结构元数据)
        # 所有的 parquet 文件必须具有相同的列和数据类型
        schema = pq.read_schema(files[0])

        # 2. 打开 ParquetWriter 进行流式写入
        # compression='zstd' 压缩率高且解压极快，非常适合 Parquet
        with pq.ParquetWriter(out_file, schema=schema, compression="zstd") as writer:
            for f in files:
                # 逐个读取小文件
                table = pq.read_table(f)
                # 追加写入到大文件 (内存用完即刻释放)
                writer.write_table(table)

        return f"成功: {day_dir.name} (合并了 {len(files)} 个文件)"

    except Exception as e:
        return f"错误 {day_dir.name}: {e}"


# 主控函数：分配多进程任务
def merge_all_directories(src_root, dst_root):
    dst_root.mkdir(parents=True, exist_ok=True)

    # 找到所有的日期子文件夹
    day_dirs = sorted([d for d in src_root.iterdir() if d.is_dir()])
    logger.info(f"共发现 {len(day_dirs)} 个日期文件夹待合并")

    if not day_dirs:
        return

    # 构造任务参数列表
    tasks = [(d, dst_root) for d in day_dirs]

    # 启动多进程池
    logger.info(f"启动多进程合并，进程数: {NUM_WORKERS}")
    with Pool(processes=NUM_WORKERS) as pool:
        # imap_unordered 配合 tqdm 显示进度条
        results = list(tqdm(pool.imap_unordered(merge_single_day, tasks), total=len(tasks), desc="合并进度"))

    # 打印错误信息
    error_count = 0
    for res in results:
        if res and res.startswith("错误"):
            logger.error(res)
            error_count += 1

    logger.info(f"全部合并完成。异常数: {error_count}")


# 步骤3：统计每日出行移动指标
def user_stat():
    def q75(x):
        return x.quantile(0.75)

    files = sorted(list(DST_MERGE_DIR.glob("*.parquet")))
    metrics_list = []
    for f in files:
        try:
            # 读取单日个人数据，同时计算网格指标和全市指标
            # 1. 基于居住地的网格聚合 (Grid-Level)
            df = pd.read_parquet(f, engine="pyarrow", dtype_backend="pyarrow")
            df_home = df.dropna(subset=["home_h3_grid"])  # 按居住地网格聚合
            valid_df = df_home[(df_home["TTD"] < 500_000) & (df_home["RG"] < 200_000)]

            # 按常住地网格聚合
            grid_agg = (
                valid_df.groupby("home_h3_grid")
                .agg(
                    resident_count=("Userid", "nunique"),  # 常住人口数
                    TTD_median=("TTD", "median"),  # 出行距离中位数
                    RG_median=("RG", "median"),  # 活动范围中位数
                    RG_p75=("RG", q75),  # 活动范围75分位数 (头部活跃度)
                    hdis_max_median=("hdis_max", "median"),  # 最大离家距离中位数
                    hometime_mean=("hometime", "mean"),  # 平均在家时长/点数
                )
                .reset_index()
            )
            grid_agg = grid_agg[grid_agg["resident_count"] >= 10]  # 过滤低人数的噪音网格
            coords = grid_agg["home_h3_grid"].apply(lambda x: h3.cell_to_latlng(x))  # v4 API
            grid_agg["lat"] = [c[0] for c in coords]  # 对应经纬度
            grid_agg["lng"] = [c[1] for c in coords]

            # 整数列转为 int32, 浮点数列转为 float32
            grid_agg["resident_count"] = grid_agg["resident_count"].astype("int32")
            float_cols = ["TTD_median", "RG_median", "RG_p75", "hdis_max_median", "hometime_mean", "lat", "lng"]
            grid_agg[float_cols] = grid_agg[float_cols].astype("float32")

            # 保存网格聚合结果到同名文件
            out_grid_path = DST_GRID_DIR / f.name
            out_grid_path.parent.mkdir(parents=True, exist_ok=True)
            grid_agg.to_csv(out_grid_path.with_suffix(".csv"), index=False)

            # 2. 全市整体聚合 (City-Level)
            city_metrics = {
                "date": f.stem,
                "total_active_users": valid_df["Userid"].nunique(),
                # 中位数画像
                "TTD_median": valid_df["TTD"].median(),
                "RG_median": valid_df["RG"].median(),
                "hdis_max_median": valid_df["hdis_max"].median(),
                # 头部人群画像 (90分位数)
                "TTD_p90": valid_df["TTD"].quantile(0.90),
                "RG_p90": valid_df["RG"].quantile(0.90),
                # 昼夜差异 (只统计非0值，避免夜间不出门的人拉低整体水平)
                "RG_day_median": (
                    valid_df.loc[valid_df["RG1"] > 0, "RG1"].median() if "RG1" in valid_df.columns else pd.NA
                ),
                "RG_night_median": (
                    valid_df.loc[valid_df["RG2"] > 0, "RG2"].median() if "RG2" in valid_df.columns else pd.NA
                ),
            }
            metrics_list.append(city_metrics)
            logger.info(f"处理文件 {f} 成功")

        except Exception as e:
            logger.error(f"处理文件 {f} 失败: {e}")

    final_df = pd.DataFrame(metrics_list)
    final_df.to_csv(DST_GRID_DIR / "city_metrics.csv", index=False)


if __name__ == "__main__":
    generate_user_profiles(DST_DIR, DST_MERGE_DIR / "user_profiles.parquet")
    # merge_all_directories(DST_DIR, DST_MERGE_DIR)
    # user_stat()
