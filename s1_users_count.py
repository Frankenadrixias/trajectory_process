# -*-coding:utf-8 -*-
"""
@File    :   s1_users_count.py
@Time    :   2026/01/13 16:37
@Author  :   huangsh, FriedrichXR
@Version :   2.0
@Contact :   1126456109@qq.com，2249307370@qq.com
@Desc    :   基于排序好的parquet文件提取用户信息表
"""
import gc
import logging
import warnings
import pandas as pd
from pathlib import Path
from collections import defaultdict

"""
基于排序好的parquet文件（data_v3）统计输出用户信息
共同用户表 CUsers.feather (长期观测用户id文件) 和全部用户表 Users.feather (全部用户id文件)
输入：data_v3
输出：data_v3

输入目录结构要求：
F:/data_v3/北京24/
    ├── SceneReco/
    │   ├── 2024-01-01.parquet
    │   └── 2024-01-02.parquet
    ├── Timing/
    ├── WifiConnect/
    └── WifiStable/

输出目录结构：
F:/data_v3/北京24/
    ├── daily_users/
    │   ├── 2024-01-01.feather
    │   └── 2024-01-02.feather
    ├── SceneReco/
    │   ├── 2024-01-01.parquet
    │   └── 2024-01-02.parquet
    ├── Timing/
    ├── WifiConnect/
    ├── WifiStable/
    ├── CUsers.feather (长期观测用户id文件)
    ├── Users.feather (全部用户id文件)
    └── user_active_days.feather (用户活跃天数文件)
"""

# ================= 配置数据 =================
root_path = Path(r"H:\data_v3\厦门市")
categories = ["Timing", "SceneReco", "WifiConnect", "WifiStable"]
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
# ==========================================


# 保存用户统计结果
# 对每天：读取 4 种 datatype parquet，求并集 & 交集
def count_daily_users(root_dir: Path, user_col: str = "Userid"):

    # 假设所有类型文件名一致（按天）
    sample_dir = root_dir / categories[0]
    days = sorted(f.stem for f in sample_dir.glob("*.parquet"))
    logger.info(f"共 {len(days)} 天")

    # 初始化变量
    all_users_set = set()
    common_users_set = None
    daily_counts = []
    daily_category_counts = defaultdict(dict)  # 日期 -> 类别 -> 用户数
    user_active_days = defaultdict(int)  # 用户id -> 活跃天数

    # 创建输出目录
    users_dir = root_dir / "daily_users"
    users_dir.mkdir(parents=True, exist_ok=True)

    # 当日数据
    for i, day in enumerate(days, 1):

        logger.info(f"[{i}/{len(days)}] 处理日期 {day}")
        union_users_day = set()

        for cat in categories:
            f = root_dir / cat / f"{day}.parquet"
            if not f.exists():
                logger.warning(f"缺失文件: {f}")
                continue

            # 计算当天的所有用户（四类数据的并集）
            df = pd.read_parquet(f, columns=[user_col], engine="pyarrow", dtype_backend="pyarrow")
            users = df[user_col].unique()
            union_users_day.update(users)
            daily_category_counts[day][cat] = len(users)
            del users

        # 统计用户活跃天数：每个用户当天只 +1 次
        for uid in union_users_day:
            user_active_days[uid] += 1

        # 更新累计用户
        all_users_set.update(union_users_day)

        # 更新累计公共用户
        if common_users_set is None:
            common_users_set = set(union_users_day)
        else:
            common_users_set.intersection_update(union_users_day)

        # 构建行数据
        row = {
            "date": day,
            "user_count": len(union_users_day),
            "cumulative_total_users": len(all_users_set),
            "cumulative_common_users": len(common_users_set),
            "Timing_count": daily_category_counts[day].get("Timing", 0),
            "SceneReco_count": daily_category_counts[day].get("SceneReco", 0),
            "WifiConnect_count": daily_category_counts[day].get("WifiConnect", 0),
            "WifiStable_count": daily_category_counts[day].get("WifiStable", 0),
        }
        daily_counts.append(row)

        # 保存每日用户数据
        daily_user_df = pd.DataFrame({user_col: list(union_users_day)})
        daily_user_df.to_feather(users_dir / f"{day}.feather")

        logger.info(
            f"  当日活跃用户数: {len(union_users_day)}，"
            f"累计公共用户数: {len(common_users_set)}，"
            f"累计总用户数: {len(all_users_set)}"
        )

        del union_users_day
        gc.collect()

    # 保存每日统计
    if daily_counts:
        daily_counts_df = pd.DataFrame(daily_counts)
        daily_counts_df.to_csv(root_dir / "daily_user_counts.csv", index=False)
        logger.info(f"保存每日用户统计: {len(daily_counts_df)} 天")

    # 保存公共用户
    if common_users_set is not None:
        common_df = pd.DataFrame({"Userid": list(common_users_set)})
        common_df.to_feather(root_dir / "CUsers.feather")
        logger.info(f"保存公共用户: {len(common_users_set)} 个用户")
    else:
        logger.warning("未找到公共用户")
        pd.DataFrame(columns=["Userid"]).to_feather(root_dir / "CUsers.feather")

    # 保存所有用户
    all_df = pd.DataFrame({user_col: list(all_users_set)})
    all_df.to_feather(root_dir / "Users.feather")
    logger.info(f"保存所有用户: {len(all_users_set)} 个用户")

    df = pd.DataFrame(user_active_days.items(), columns=[user_col, "active_days"])
    df.to_feather(root_dir / "user_active_days.feather")
    logger.info(f"用户活跃天数表已保存")

    s = pd.Series(user_active_days.values())
    dist = s.value_counts().sort_index().reset_index()
    dist.columns = ["active_days", "user_count"]
    dist.to_csv(root_dir / "user_active_distribution.csv", index=False)
    logger.info(f"活跃天数分布已保存")

    return


def compute_union_and_intersection(input_dir_common: Path, input_dir_union: Path, user_col: str = "Userid"):
    common_feather_files = sorted(input_dir_common.glob("*.feather"))
    union_feather_files = sorted(input_dir_union.glob("*.feather"))

    if not common_feather_files or not union_feather_files:
        raise ValueError(f"目录中没有要求的 feather 文件")

    union_users = set()
    intersection_users = None

    for i, f in enumerate(common_feather_files, 1):
        logger.info(f"[{i}/{len(common_feather_files)}] 读取 {f.name}")
        df = pd.read_feather(f, columns=[user_col])
        users = set(df[user_col].astype(str))
        if intersection_users is None:  # 交集
            intersection_users = users
        else:
            intersection_users.intersection_update(users)
        del df, users  # 释放
        gc.collect()
        logger.info(f"当前交集数: {len(intersection_users)}")

    # 保存交集
    pd.DataFrame({user_col: list(intersection_users)}).to_feather(input_dir_common / "CUsers.feather")
    logger.info(f"交集已保存，总人数：{len(intersection_users)}")

    for i, f in enumerate(union_feather_files, 1):
        logger.info(f"[{i}/{len(union_feather_files)}] 读取 {f.name}")
        df = pd.read_feather(f, columns=[user_col])
        users = set(df[user_col].astype(str))
        union_users.update(users)  # 并集
        del df, users  # 释放
        gc.collect()
        logger.info(f"当前并集数: {len(union_users)}")

    # 保存并集
    pd.DataFrame({user_col: list(union_users)}).to_feather(input_dir_union / "Users.feather")
    logger.info(f"并集已保存，总人数：{len(union_users)}")


def select_common_users(root_dir: Path, user_col: str, thresh_days: int):
    user_path = root_dir / "user_active_days.feather"
    df = pd.read_feather(user_path, columns=[user_col, "active_days"])
    common_users = df.loc[df["active_days"] >= thresh_days, user_col].drop_duplicates()
    common_users.to_frame(name=user_col).to_feather(root_dir / "select_common_users.feather")


if __name__ == "__main__":
    count_daily_users(root_path, "Userid")
    # compute_union_and_intersection(Path(r"G:\data_v3\CUsers"), Path(r"G:\data_v3\Users"), "Userid")
    # select_common_users(root_path, "Userid", 184)
