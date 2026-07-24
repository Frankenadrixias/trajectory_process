# -*- coding: utf-8 -*-
"""
@File    :   s5_home_identify.py
@Time    :   2026/04/26 23:56
@Author  :   FriedrichXR
@Version :   1.0
@Contact :   2249307370@qq.com
@Desc    :   将聚类、合并好的轨迹点去除异常点和跳跃点，并进行逐小时插值
"""
import os
import gc
import h3
import logging
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from pyproj import CRS, Transformer
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns

"""
# 数据结构
F:/data_v4/北京市/Merge_v2b/
    ├── 2024-01-01/
    │     ├── group_0.parquet
    │     └── group_1.parquet
    ├── 2024-01-02/
    │     ├── group_0.parquet
    │     └── ...
每个文件结构：Userid | lng | lat | hour | h3_grid
"""

# ================= 配置区域 =================
SRC_DIR = Path(r"K:\data_v4\北京市\Merge_v2b")  # 路径配置
DST_DIR = Path(r"G:\data_v5.4\北京市")
SHAPEFILE_PATH = "data/shape/北京市.shp"
H3_SET_PATH = "data/shape/beijing_h3_set_9.pkl"
# ========== 一般只需要修改以上内容 ==========

# 参数配置
NIGHT_START = 22  # 夜间起始小时
NIGHT_END = 6  # 夜间结束小时 (不含)
DAY_START = 10  # 白天起始小时
DAY_END = 18  # 白天结束小时 (不含)
NIGHT_DOMINANCE_RATIO = 0.6  # 夜间主位置占比阈值 (主位置次数 / 总夜间次数)
WORK_DOMINANCE_RATIO = 0.4  # 工作时间主位置占比阈值 (主位置次数 / 总日间次数)
TOP_NUM = 5  # 保留前几位比例
H3_RESOLUTION = 9  # H3 分辨率
MAX_CPU_USAGE = 0.6  # 最大cpu使用比例 (90%)
NUM_WORKERS = min(int(MAX_CPU_USAGE * cpu_count()), 12)  # 根据 CPU 核心数和内存设定进程数
albers_crs = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=105 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
trans = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(albers_crs), always_xy=True)

# 全局变量
shared_dataframe = None
shared_valid_users = None  # 用于在子进程中共享 valid_users
home_neighbors = pd.DataFrame()
work_neighbors = pd.DataFrame()

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ================= 功能函数 =================
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# ============================================
# PART 1: 基于 Shapefile 生成该区域内的所有 H3 index 集合
# ============================================
def build_h3_set(shape_path, resolution: int = H3_RESOLUTION):
    if Path(H3_SET_PATH).exists():
        logger.info(f"H3 集合文件已存在: {H3_SET_PATH}，跳过生成。")
        return

    logger.info(f"H3 版本: {h3.__version__}")
    logger.info(f"正在读取 Shapefile: {shape_path}")
    gdf = gpd.read_file(shape_path)
    h3_set = set()

    # 测试版本兼容问题，构造 H3 v4 认可的几何对象
    try:
        # 在 v4 中，Polygon 类通常被命名为 LatLngPoly，如果 h3.LatLngPoly 还报错说明安装可能不完整
        LatLngPoly = h3.LatLngPoly
    except AttributeError:
        from h3.api.basic_str import LatLngPoly   # 备选方案：尝试直接从底层模块调用

    # 执行填充：containment_mode=2 是“与边界相交”模式 (Intersects)
    # 如果版本不支持该参数，可删去，默认是中心点包含模式
    def polygon_to_cells(polygon):
        try:
            return h3.polygon_to_cells(polygon, resolution, containment_mode=2)
        except TypeError:
            # 如果不支持 containment_mode 参数，则使用默认模式
            return h3.polygon_to_cells(polygon, resolution)

    # 获取 geometry 对象
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # 统一将 Polygon 和 MultiPolygon 处理为列表
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            try:
                exterior = [(lat, lng) for lng, lat in poly.exterior.coords]  # 外环
                interiors = [[(lat, lng) for lng, lat in ring.coords] for ring in poly.interiors]  # 内环 (孔洞)

                # 构建 H3 几何对象，执行填充
                h3_poly = LatLngPoly(exterior, *interiors)
                cells = polygon_to_cells(h3_poly)
                h3_set.update(cells)

            except Exception as e:
                logger.error(f"处理多边形出错: {e}")

    logger.info(f"生成的 H3 网格总数: {len(h3_set)}")

    # 保存结果
    Path(H3_SET_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(H3_SET_PATH, "wb") as f:
        pickle.dump(h3_set, f)
    logger.info(f"H3 集合已保存至 {H3_SET_PATH}")


# ============================================
# PART 2: 基于筛选后的 H3 网格 id 统计用户的常住地和工作地
# ============================================
# 扫描所有日期目录，按 group 文件名组织文件路径
# 返回: {'group_0.parquet': [path_day1, path_day2...], 'group_1.parquet': [...] }
def organize_files_by_group(src_root):
    logger.info("正在扫描并按组索引文件...")
    files_map = defaultdict(list)

    # 递归查找所有 parquet
    all_files = sorted(list(src_root.glob("**/*.parquet")))

    for f in all_files:
        files_map[f.name].append(f)

    logger.info(f"共发现 {len(all_files)} 个文件，归类为 {len(files_map)} 个文件组")
    return files_map


# 处理单个组的所有历史数据，识别其中的常住用户
def process_single_group(group_name, file_list, beijing_h3_set):
    # 1. 统计该组所有日期的对应日间和夜间数据
    all_night_stats = []  # 保存夜间位置计数
    all_work_stats = []  # 保存日间位置计数

    for f in file_list:
        try:
            # 读取单日分组文件
            df = pd.read_parquet(f, columns=["Userid", "hour", "h3_grid"], engine="pyarrow", dtype_backend="pyarrow")

            # 筛选时间
            df_night = df[(df["hour"] < NIGHT_END) | (df["hour"] >= NIGHT_START)]
            df_work = df[(df["hour"] < DAY_END) & (df["hour"] >= DAY_START)]

            # 聚合: Userid, h3_grid -> count
            all_night_stats.append(df_night.groupby(["Userid", "h3_grid"]).size().reset_index(name="cnt"))
            all_work_stats.append(df_work.groupby(["Userid", "h3_grid"]).size().reset_index(name="cnt"))

        except Exception as e:
            logger.error(f"读取 {f} 失败: {e}")

    if not all_night_stats:
        logger.error(f"提取失败")
        return pd.DataFrame()

    # 2. 合并该组的同一用户在不同日期对同一网格的计数
    stats_night = pd.concat(all_night_stats).groupby(["Userid", "h3_grid"])["cnt"].sum().reset_index()
    stats_work = pd.concat(all_work_stats).groupby(["Userid", "h3_grid"])["cnt"].sum().reset_index()
    del all_night_stats, all_work_stats

    # 辅助函数：计算锚点（含邻域合并逻辑）
    def identify_anchor_logic(stats_df, ratio_threshold, mask_set=None):
        # a. 每个用户总计数
        user_totals = stats_df.groupby("Userid")["cnt"].sum().rename("total_cnt")

        # b. 计算每个用户出现前 top_num 的网格数量占比
        sorted_stats = stats_df.sort_values(by=["Userid", "cnt"], ascending=[True, False])
        top = sorted_stats.groupby("Userid").head(TOP_NUM).copy()
        top["total_cnt"] = top["Userid"].map(user_totals)
        top["grid_ratio"] = (top["cnt"] / top["total_cnt"]).astype(np.float32)

        # 转换为宽表
        top["rank"] = top.groupby("Userid").cumcount() + 1
        top_pivot = top.pivot(index="Userid", columns="rank", values="grid_ratio")  # 占比透视为宽表
        top_pivot.columns = [f"top{col}_ratio" for col in top_pivot.columns]

        # 确保 top1_ratio 到 top5_ratio 列全部存在（防止某些用户网格不足5个导致列缺失）
        for i in range(1, TOP_NUM + 1):
            col = f"top{i}_ratio"
            if col not in top_pivot.columns:
                top_pivot[col] = np.nan
        top_pivot = top_pivot[[f"top{i}_ratio" for i in range(1, TOP_NUM + 1)]].reset_index()

        # c. 寻找最常出现的单一网格
        dominant = (sorted_stats.drop_duplicates(subset=["Userid"], keep="first")
                    .rename(columns={"h3_grid": "center_grid", "cnt": "center_cnt"}))

        # d. 邻域合并 (grid_disk)
        def get_neighbors(gh):
            try:
                return list(h3.grid_disk(gh, 1)) if hasattr(h3, "grid_disk") else list(h3.k_ring(gh, 1))
            except:
                return [gh]

        dominant["neighborhood"] = dominant["center_grid"].apply(get_neighbors)
        nb_df = dominant[["Userid", "neighborhood"]].explode("neighborhood")
        nb_counts = nb_df.merge(stats_df, left_on=["Userid", "neighborhood"], right_on=["Userid", "h3_grid"])
        nb_sum = nb_counts.groupby("Userid")["cnt"].sum().rename("nb_total_cnt")

        # d. 汇总比例，并合并 top 比例
        res = (
            dominant.merge(nb_sum, on="Userid")
            .merge(user_totals, on="Userid")
            .merge(top_pivot, on="Userid", how="left")
        )
        res["ratio"] = (res["nb_total_cnt"] / res["total_cnt"]).astype(np.float32)

        # e. 筛选条件
        mask = res["ratio"] >= ratio_threshold
        if mask_set is not None:
            mask = mask & res["center_grid"].isin(mask_set)
        return res[mask].copy()

    # 3. 提取居住地 (必须满足阈值且在北京)
    home_users = identify_anchor_logic(stats_night, NIGHT_DOMINANCE_RATIO, beijing_h3_set)
    rename_dict_home = {"center_grid": "home_h3_grid", "ratio": "home_ratio"}
    for i in range(1, TOP_NUM + 1):
        rename_dict_home[f"top{i}_ratio"] = f"home_top{i}_ratio"
    home_users = home_users.rename(columns=rename_dict_home)

    # 4. 提取工作地 (仅针对已识别的常住用户)
    # 先过滤出常住用户白天的轨迹
    stats_work_filtered = stats_work[stats_work["Userid"].isin(home_users["Userid"])]
    work_users = identify_anchor_logic(stats_work_filtered, WORK_DOMINANCE_RATIO)
    rename_dict_work = {"center_grid": "work_h3_grid", "ratio": "work_ratio"}
    for i in range(1, TOP_NUM + 1):
        rename_dict_work[f"top{i}_ratio"] = f"work_top{i}_ratio"
    work_users = work_users.rename(columns=rename_dict_work)

    # 5. 合并结果 (Left Join 保证了不稳定的工作地显示为 NaN)
    home_cols = ["Userid", "home_h3_grid", "home_ratio"] + [f"home_top{i}_ratio" for i in range(1, TOP_NUM + 1)]
    work_cols = ["Userid", "work_h3_grid", "work_ratio"] + [f"work_top{i}_ratio" for i in range(1, TOP_NUM + 1)]
    final_df = home_users[home_cols].merge(work_users[work_cols], on="Userid", how="left")

    # 6. 批量计算坐标 (针对 Home 和 Work)
    def fill_coords(data, prefix):
        h3_col = f"{prefix}_h3_grid"
        # 排除 NaN 后的唯一网格进行转换，提高效率
        unique_grids = data[data[h3_col].notna()][h3_col].unique()
        grid_to_pt = {
            g: (h3.cell_to_latlng(g) if hasattr(h3, "cell_to_latlng") else h3.h3_to_geo(g)) for g in unique_grids
        }

        coords = data[h3_col].map(grid_to_pt)
        data[f"{prefix}_lat"] = [c[0] if isinstance(c, tuple) else np.nan for c in coords]
        data[f"{prefix}_lon"] = [c[1] if isinstance(c, tuple) else np.nan for c in coords]

        # 投影转换 (对非空值)
        mask = data[f"{prefix}_lon"].notna()
        if mask.any():
            x, y = trans.transform(data.loc[mask, f"{prefix}_lon"].values, data.loc[mask, f"{prefix}_lat"].values)
            data.loc[mask, f"{prefix}_x"] = x.astype(np.float32)
            data.loc[mask, f"{prefix}_y"] = y.astype(np.float32)
        return data

    final_df = fill_coords(final_df, "home")
    final_df = fill_coords(final_df, "work")

    return final_df


# 按组分批处理
def identify_residents_by_group(src_root, dst_root, h3_pkl_path):
    # 1. 准备基础数据：城市H3网格
    with open(h3_pkl_path, "rb") as f:
        beijing_h3_set = pickle.load(f)

    # 2. 对文件进行分组索引
    # map: {'group_0.parquet': [path_day1, path_day2...], ...}
    files_map = organize_files_by_group(src_root)
    all_residents_parts = []

    # 3. 按组遍历处理
    # 使用 tqdm 显示处理进度 (多少个组)
    group_names = sorted(list(files_map.keys()))
    for g_name in tqdm(group_names, desc="Processing Groups"):
        file_list = files_map[g_name]

        # 调用核心处理函数
        res_df = process_single_group(g_name, file_list, beijing_h3_set)

        if not res_df.empty:
            all_residents_parts.append(res_df)

        # 强制垃圾回收，确保内存释放
        gc.collect()

    logger.info("所有组处理完毕，正在合并结果...")

    if not all_residents_parts:
        logger.warning("未识别出任何常住用户！")
        return None

    # 4. 合并所有组的结果
    final_residents_df = pd.concat(all_residents_parts, ignore_index=True)
    print(final_residents_df.columns)

    # 5. 保存
    dst_root.mkdir(parents=True, exist_ok=True)
    out_file = dst_root / "residents1.parquet"
    final_residents_df.to_parquet(out_file, engine="pyarrow", compression="zstd", index=False)

    # 保存网格常住用户数量
    grid_counts_df = final_residents_df.groupby("home_h3_grid").size().reset_index(name="user_count")
    grid_counts_df["home_h3_grid"] = grid_counts_df["home_h3_grid"].astype(str)
    grid_counts_df["user_count"] = grid_counts_df["user_count"].astype(np.int32)
    out_file_show = dst_root / "residents_count1.parquet"
    grid_counts_df.to_parquet(out_file_show, engine="pyarrow", compression="zstd", index=False)

    logger.info(f"处理完成。总常住用户数: {len(final_residents_df)}")
    return final_residents_df


def plot_top5_ratios_boxplot(df, save_path=None):
    """绘制所有用户居住地和工作地前5网格占比的箱线图"""
    # 设置支持中文的字体
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 准备列名
    home_cols = [f"home_top{i}_ratio" for i in range(1, 6)]
    work_cols = [f"work_top{i}_ratio" for i in range(1, 6)]

    # 检查数据中是否存在对应的列
    has_home = all(col in df.columns for col in home_cols)
    has_work = all(col in df.columns for col in work_cols)

    if not has_home and not has_work:
        logger.warning("数据中未包含 Top 5 占比列，无法绘制箱线图。")
        return

    # 创建并排子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # 1. 绘制居住地 Top 5 占比
    if has_home:
        # 重命名列名以便在图表横轴显示
        home_plot_df = df[home_cols].rename(
            columns={col: f"Top {i}" for i, col in enumerate(home_cols, 1)}
        )
        sns.boxplot(ax=axes[0], data=home_plot_df, palette="Blues_r")
        axes[0].set_title("居住地 (Night) 前 5 网格占比分布")
        axes[0].set_xlabel("出现频次排名")
        axes[0].set_ylabel("占比")
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    else:
        axes[0].text(
            0.5,
            0.5,
            "无居住地数据",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )

    # 2. 绘制工作地 Top 5 占比
    if has_work:
        work_plot_df = df[work_cols].rename(
            columns={col: f"Top {i}" for i, col in enumerate(work_cols, 1)}
        )
        sns.boxplot(ax=axes[1], data=work_plot_df, palette="Oranges_r")
        axes[1].set_title("工作地 (Day) 前 5 网格占比分布")
        axes[1].set_xlabel("出现频次排名")
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    else:
        axes[1].text(
            0.5,
            0.5,
            "无工作地数据",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )

    plt.suptitle("常住人口出行锚点前 5 网格占比箱线图", fontsize=16, y=0.98)
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"箱线图已成功保存至: {save_path}")

    plt.show()


# ============================================
# PART 3: 基于 residents.parquet 从 Merge_v2b 数据中提取常住用户数据，计算移动指标
# ============================================
# 进程初始化函数：将主进程传来的 users_set 赋值给子进程的全局变量。
def init_worker(users_home_dataframe, home_path, work_path):
    global shared_valid_users
    global shared_dataframe
    global home_neighbors
    global work_neighbors

    shared_dataframe = users_home_dataframe
    shared_valid_users = set(users_home_dataframe["Userid"])
    home_neighbors = pd.read_feather(home_path, dtype_backend="pyarrow")
    work_neighbors = pd.read_feather(work_path, dtype_backend="pyarrow")


# 提取到主程序中执行的预处理函数：提取网格的邻域缓存
def prepare_neighbors_static(resident_df, dst_root):

    home_path = dst_root / "home_neighbors.feather"
    work_path = dst_root / "work_neighbors.feather"

    # 1. 检查邻域缓存是否存在
    if home_path.exists() and work_path.exists():
        logger.info("检测到邻域缓存文件已存在，跳过邻域计算，直接返回路径。")
        return home_path, work_path

    # 2. 如果不存在，执行计算逻辑
    logger.info("未检测到完整缓存，开始计算邻域集合（此过程较耗时）...")

    def get_disk(gh):
        try:
            return list(h3.grid_disk(gh, 1)) if hasattr(h3, "grid_disk") else list(h3.k_ring(gh, 1))
        except:
            return [gh]

    logger.info("预计算居住地邻域集合...")
    home_map = resident_df[["Userid", "home_h3_grid"]].dropna()
    home_map["home_valid_grids"] = home_map["home_h3_grid"].apply(get_disk)
    home_nb = home_map.explode("home_valid_grids")[["Userid", "home_valid_grids"]]
    home_nb["home_valid_grids"] = home_nb["home_valid_grids"].astype("string[pyarrow]")  # 提前转为 pyarrow 字符串
    home_nb["is_at_home_tmp"] = True
    del home_map

    logger.info("预计算工作地邻域集合...")
    work_map = resident_df[["Userid", "work_h3_grid"]].dropna()
    work_map["work_valid_grids"] = work_map["work_h3_grid"].apply(get_disk)
    work_nb = work_map.explode("work_valid_grids")[["Userid", "work_valid_grids"]]
    work_nb["work_valid_grids"] = work_nb["work_valid_grids"].astype("string[pyarrow]")
    work_nb["is_at_work_tmp"] = True
    del work_map

    # 3. 写入硬盘
    logger.info(f"正在将计算结果写入目录: {dst_root}")
    home_nb.to_feather(home_path, compression='zstd')
    work_nb.to_feather(work_path, compression='zstd')

    return home_path, work_path


#  单个文件的处理逻辑（子进程执行）：计算单个文件的每日指标
def filter_single_file(args):
    # 参数传递
    f_path, src_root, dst_root = args
    try:
        # 计算相对路径和输出路径
        rel_path = f_path.relative_to(src_root)
        out_path = dst_root / "Merge_v0" / rel_path

        # 如果文件已存在，跳过（可选，根据需求保留或删除）
        if out_path.exists():
            return None
        # 读取数据
        df = pd.read_parquet(f_path, engine="pyarrow", dtype_backend="pyarrow",
                             columns=["Userid", "hour", "x", "y", "h3_grid"])

        # 使用全局变量进行过滤
        df_sel = df[df["Userid"].isin(shared_valid_users)]
        if df_sel.empty:
            return f"Skip: {f_path.name} (Empty)"
        del df

        # 关联居住地信息
        # left join: 如果某用户没有推断出家（比如从没在夜间出现），相关字段为 NaN
        df_sel = pd.merge(df_sel, shared_dataframe, on="Userid", how="left")
        df_sel = df_sel.sort_values(by=["Userid", "hour"], kind="mergesort")

        # 1. 基础位移计算 (Vectorized)
        # 计算前后两点的距离 (Step Distance)
        dist_sq = (df_sel["x"] - df_sel["x"].shift(1)) ** 2 + (df_sel["y"] - df_sel["y"].shift(1)) ** 2  # 距离公式
        df_sel["step_dist"] = np.sqrt(dist_sq).fillna(0).astype("float[pyarrow]")
        # 处理用户边界：如果当前行和上一行不是同一个用户，距离设为 0
        df_sel.loc[df_sel["Userid"] != df_sel["Userid"].shift(1), "step_dist"] = 0

        # 2. 居住地相关计算
        has_home_mask = df_sel["home_h3_grid"].notna()
        df_sel["dist_to_home"] = pd.Series(dtype="float[pyarrow]", index=df_sel.index)
        # 离家距离计算
        df_sel.loc[has_home_mask, "dist_to_home"] = np.hypot(
            df_sel.loc[has_home_mask, "x"] - df_sel.loc[has_home_mask, "home_x"],
            df_sel.loc[has_home_mask, "y"] - df_sel.loc[has_home_mask, "home_y"],
        )
        # 清理中间列
        df_sel.drop(columns=["home_x", "home_y", "work_x", "work_y"], inplace=True, errors="ignore")

        # 3. 判断是否在家 (通过 Left Join 匹配邻域)
        # 原理：如果当前 (Userid, h3_grid) 能在预生成的邻域表里找到，说明他在家
        df_sel = pd.merge(df_sel, home_neighbors, how="left",
                          left_on=["Userid", "h3_grid"], right_on=["Userid", "home_valid_grids"])
        df_sel["is_at_home"] = df_sel["is_at_home_tmp"].eq(True).fillna(False)

        # 4. 判断是否在岗 (同理)
        df_sel = pd.merge(df_sel, work_neighbors, how="left",
                          left_on=["Userid", "h3_grid"], right_on=["Userid", "work_valid_grids"])
        df_sel["is_at_work"] = df_sel["is_at_work_tmp"].eq(True).fillna(False)
        df_sel["is_at_work"] = np.where(df_sel["is_at_home"], False, df_sel["is_at_work"])
        # 清理中间列
        df_sel.drop(columns=["is_at_home_tmp", "is_at_work_tmp", "home_valid_grids", "work_valid_grids"],
                    inplace=True, errors="ignore")

        # 离家且离岗 (用于识别在其他场所的活动)
        df_sel["is_searching"] = (~df_sel["is_at_home"]) & (~df_sel["is_at_work"])

        # 4. 聚合统计
        grp = df_sel.groupby("Userid")  # 分组

        # 定义聚合函数：Rg计算辅助函数 (利用 var)
        def calc_rg(g):
            if len(g) < 2:
                return 0.0
            return np.sqrt(g["x"].var(ddof=0) + g["y"].var(ddof=0))  # ddof=0 对应总体方差，符合Rg物理定义

        # --- A. 全天基础行为特征 ---
        res = grp.agg(
            total_dist=("step_dist", "sum"),  # 总位移：反映活跃程度
            max_dist=("step_dist", "max"),  # 最大单次位移
            home_stay_hours=("is_at_home", "sum"),  # 在家时长：统计 is_at_home 为 True 的行数
            work_stay_hours=("is_at_work", "sum"),  # 在岗时长：统计 is_at_work 为 True 的行数
            other_stay_hours=("is_searching", "sum"),  # 在其他地方时长：反映“探索/社交”积极性
            unique_grids=("h3_grid", "nunique"),  # 到访网格数：空间多样性
            avg_dist_home=("dist_to_home", "mean"),  # 平均离家距离
        )
        res["RG"] = grp.apply(calc_rg, include_groups=False)  # 计算全天 RG

        # --- B. 昼夜节奏特征 ---
        # 白天活跃度
        df_sel_day = df_sel[(df_sel["hour"] >= DAY_START) & (df_sel["hour"] <= DAY_END)]  # 标记时段
        if not df_sel_day.empty:
            grp_day = df_sel_day.groupby("Userid")
            res_day = grp_day.agg(dist_day=("step_dist", "sum"))  # 计算白天移动距离
            res_day["RG_day"] = grp_day.apply(calc_rg, include_groups=False)  # 计算白天移动距离
            res = res.join(res_day)

        # 夜间活跃度
        df_sel_night = df_sel[(df_sel["hour"] < NIGHT_END) | (df_sel["hour"] >= NIGHT_START)]  # 标记时段
        if not df_sel_night.empty:
            grp_night = df_sel_night.groupby("Userid")
            res_night = grp_night.agg(dist_night=("step_dist", "sum"))  # 计算夜间移动距离
            res_night["RG_night"] = grp_night.apply(calc_rg, include_groups=False)  # 计算夜间移动距离
            res = res.join(res_night)

        type_map = {
            "home_stay_hours": "int16[pyarrow]",
            "work_stay_hours": "int16[pyarrow]",
            "other_stay_hours": "int16[pyarrow]",
            "unique_grids": "int16[pyarrow]",
            "RG": "float[pyarrow]",
            "RG_day": "float[pyarrow]",
            "RG_night": "float[pyarrow]",
        }
        res = res.astype(type_map)
        del df_sel

        # 5. 计算衍生分类指标 (用于后续聚类)
        res = res.reset_index()
        # 关联静态信息（如推断出的居住地类型等）
        final = pd.merge(res, shared_dataframe, on="Userid", how="left")

        # 职住距离 (Commute Distance)
        final["commute_dist"] = (
            np.sqrt((final["home_x"] - final["work_x"]) ** 2 + (final["home_y"] - final["work_y"]) ** 2)
            .fillna(0)
            .astype("float[pyarrow]")
        )
        # 清理中间列
        final.drop(columns=["home_x", "home_y", "work_x", "work_y"], inplace=True, errors="ignore")

        # 锚点依赖度：时间花在职住地的比例
        final["anchor_dependency"] = ((final["home_stay_hours"] + final["work_stay_hours"]) / 24.0).astype(
            "float[pyarrow]"
        )

        # 职住时间比 (Home stay vs Work stay)
        final["home_work_ratio"] = (final["work_stay_hours"] / final["home_stay_hours"]).astype("float[pyarrow]")

        # 6. 保存结果
        out_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在
        final.to_parquet(out_path, index=False)
        return f"Success: {f_path.name}"

    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"Failed: {f_path.name} - {str(e)}"


# 根据生成的 residents.parquet 并行过滤原始数据
def filter_groups_by_residents_parallel(src_root, dst_root):
    resident_file = dst_root / "residents.parquet"
    if not resident_file.exists():
        logger.error("常住用户文件不存在")
        return

    # 1. 读取常住用户 (主进程执行)
    logger.info("正在读取常住用户名单...")
    resident_df = pd.read_parquet(
        resident_file, engine="pyarrow", dtype_backend="pyarrow",
        columns=["Userid", "home_h3_grid", "work_h3_grid", "home_x", "home_y", "work_x", "work_y"],
    )
    logger.info(f"常住用户加载完毕，共 {len(resident_df)} 人")

    if resident_df.empty:
        logger.warning("常住用户列表为空，停止处理。")
        return

    # 2. 在启动并行池之前，先在主进程算好邻域
    logger.info("开始生成邻域缓存表...")
    home_p, work_p = prepare_neighbors_static(resident_df, dst_root)

    # 3. 扫描文件，准备任务参数：(文件路径, 源根目录, 目标根目录)
    files = sorted(list(src_root.glob("**/*.parquet")))
    tasks = [(f, src_root, dst_root) for f in files]
    logger.info(f"待处理文件总数: {len(files)}")

    # 4. 启动并行处理
    logger.info(f"启动并行池，进程数: {NUM_WORKERS}")
    # 使用 initializer 将 valid_users 共享给子进程
    with Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(resident_df, home_p, work_p)) as pool:
        # imap_unordered 配合 tqdm 显示进度
        results = list(
            tqdm(pool.imap_unordered(filter_single_file, tasks, chunksize=5), total=len(tasks), desc="并行过滤中")
        )

    # (可选) 检查错误日志
    error_count = 0
    for res in results:
        if res and res.startswith("Error"):
            logger.error(res)
            error_count += 1

    logger.info(f"处理完成。错误文件数: {error_count}")


if __name__ == "__main__":
    # build_h3_set(shape_path=SHAPEFILE_PATH, resolution=8)  # 生成 H3 集合
    # identify_residents_by_group(SRC_DIR, DST_DIR, H3_SET_PATH)  # 识别常住用户
    filter_groups_by_residents_parallel(SRC_DIR, DST_DIR)  # 过滤出最终文件
