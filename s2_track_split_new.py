# -*-coding:utf-8 -*-
"""
@File    :   s2_track_split_new.py
@Time    :   2026/01/14 16:39
@Author  :   huangsh, FriedrichXR
@Version :   2.0
@Contact :   1126456109@qq.com，2249307370@qq.com
@Desc    :   根据拆分后的用户将同一天改用户组的轨迹放在一个文件内
"""
import os
import gc
import logging
import pandas as pd
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

"""
输入：data_v3
输出：data_v4/Merge_v0

输入目录结构要求：
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
    ├── CUsers.feather
    ├── Users.feather
    └── user_active_days.feather
    
输出目录结构：
F:/data_v4/北京24/Merge_v0/
    ├── 2024-01-01/
    │   ├── group_0.parquet
    │   ├── group_1.parquet
    │   └── rest.parquet
    ├── 2024-01-02/
    │   └── ...
    └── ...
"""

# ================= 配置数据 =================
INPUT_DIR = r"H:\data_v3\厦门市"
OUTPUT_DIR = r"H:\data_v4\厦门市\Merge_v0"
USER_FILE = r"H:\data_v3\厦门市\Users.feather"
START_DATE = "2023-01-01"  # 起始日期
END_DATE = "2023-06-30"  # 结束日期

# 内存控制参数
GROUP_SIZE = 200000  # 用户处理批次大小，根据公共用户数确定
PROCESSES = 1  # 每批处理的日期数，一般根据电脑内存设定，保证每天的完整4套数据可以读进内存
THREADS_PER_DAY = 12  # 每个日期进程内的线程数
# =========== 一般仅需要修改以上内容 ===========

# 初始化存储变量
COMMON_USERS: set[str]  # 用于 isin / in 判断从属
COMMON_USER_LIST: list[str]  # 用于分组 / chunk
N_USERS: int

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# 创建目录（如果不存在）
def create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"创建目录: {path}")
    return path


# 定义子进程初始化函数
def init_worker_common_users(user_file: Path, user_col: str):
    global COMMON_USERS, COMMON_USER_LIST, N_USERS

    df = pd.read_feather(user_file, columns=[user_col])
    COMMON_USERS = set(df[user_col].astype(str))  # 转换为集合提高查找效率
    COMMON_USER_LIST = sorted(COMMON_USERS)
    N_USERS = len(COMMON_USER_LIST)

    del df
    gc.collect()
    logger.info(f"[PID {os.getpid()}] 进程加载公共用户: {N_USERS}")


# 用户 group 生成器
def iter_user_groups(user_list: list[str], group_size: int):
    for start in range(0, len(user_list), group_size):
        end = min(start + group_size, len(user_list))
        idx = start // group_size
        yield idx, user_list[start:end]


# 处理单日数据：加载数据、处理用户组、处理未注册用户
# 输入参数：当日日期字符串
def process_single_day(day: str):
    logger.info(f"处理日期: {day}")
    day_start = time.time()
    global COMMON_USERS, COMMON_USER_LIST, N_USERS

    # 1. 加载当天数据
    try:
        paths = [os.path.join(INPUT_DIR, d, f"{day}.parquet") for d in
                 ["Timing", "SceneReco", "WifiConnect", "WifiStable"]]
        # 读取4类数据，只选择必要的列
        df_timing, df_scene, df_wifi_c, df_wifi_s =\
            [pd.read_parquet(p, engine="pyarrow", dtype_backend="pyarrow",
                             columns=['Userid', 'lng', 'lat', 'starttime', 'endtime']) for p in paths]
        logger.info(f"[{day}] 加载完成: Timing({len(df_timing)}行), Scene({len(df_scene)}行),"
                    f"WifiConnect({len(df_wifi_c)}行), WifiStable({len(df_wifi_s)}行)")
    except Exception as e:
        logger.error(f"[{day}] 加载数据失败: {str(e)}")
        return False

    # 2. group 多线程处理用户组（并行）
    day_output_dir = create_dir(os.path.join(OUTPUT_DIR, day))
    logger.info(f"[{day}] 启动 {THREADS_PER_DAY} 线程处理用户组...")
    with ThreadPoolExecutor(max_workers=THREADS_PER_DAY) as executor:
        tasks = []
        for idx, user_group in iter_user_groups(COMMON_USER_LIST, GROUP_SIZE):
            task = (day, idx, user_group, df_timing, df_scene, df_wifi_c, df_wifi_s, day_output_dir)
            tasks.append(executor.submit(process_user_group, task))
        for t in as_completed(tasks):
            t.result()

    # 3. 处理 rest 用户（串行一次）
    try:
        rest_path = os.path.join(day_output_dir, "rest.parquet")
        if not os.path.exists(rest_path):
            logger.info(f"[{day}] 处理其他用户...")
            rest_parts = []
            for src, df in (("timing", df_timing), ("scene", df_scene), ("wifi", df_wifi_c), ("wifi", df_wifi_s)):
                mask = ~df["Userid"].isin(COMMON_USERS)  # 筛选未注册用户数据
                if mask.any():
                    part = df.loc[mask].assign(ftype=src)  # 添加类型标识
                    rest_parts.append(part)
                del mask
            # 检查是否有数据
            if rest_parts:
                # 合并数据、排序并保存
                merged = pd.concat(rest_parts).sort_values(["Userid", "starttime", "endtime"])
                merged.to_parquet(rest_path, engine="pyarrow", compression='zstd', index=False)
                logger.info(f"[{day}] 保存其他用户数据: {len(merged)} 行")
            else:
                pd.DataFrame().to_parquet(rest_path)
                logger.info(f"[{day}] 无其他用户数据，创建空文件")
        else:
            logger.info(f"[{day}] 其他用户文件已存在，跳过")
    except Exception as e:
        logger.error(f"[{day}] 处理未注册用户失败: {str(e)}", exc_info=True)

    # 4. 清理释放内存
    del df_timing, df_scene, df_wifi_c, df_wifi_s
    gc.collect()
    logger.info(f"[{day}] 处理完成，耗时{time.time() - day_start:.1f}s")
    return True


# 线程函数：处理单个用户组
# 输入参数：日期，用户组id和对应df，四类数据df，输出路径
def process_user_group(args: tuple):
    day, idx, user_group, df_timing, df_scene, df_wifi_c, df_wifi_s, day_output_dir = args
    start_time = time.time()
    group_name = f"group_{idx}"

    try:
        out_path = os.path.join(day_output_dir, f"{group_name}.parquet")
        # 跳过已处理文件
        if os.path.exists(out_path):
            logger.info(f"[{day}][{group_name}] 文件已存在，跳过")
            return
        # 筛选并合并数据
        merged_parts = []
        for src, df in (("timing", df_timing), ("scene", df_scene), ("wifi", df_wifi_c), ("wifi", df_wifi_s)):
            mask = df["Userid"].isin(user_group)
            if mask.any():
                part = df.loc[mask]
                part = part.assign(ftype=src)  # 添加类型标识
                merged_parts.append(part)
            del mask, part
        # 合并 + 排序，保存结果
        if merged_parts:
            merged = pd.concat(merged_parts).sort_values(["Userid", "starttime", "endtime"])
            merged.to_parquet(out_path, engine="pyarrow", compression='zstd', index=False)
            logger.info(f"[{day}][{group_name}] 完成 ({len(merged)} 行, {time.time() - start_time:.1f}s)")
            del merged
        # 空数据处理
        else:
            pd.DataFrame().to_parquet(out_path)
            logger.info(f"[{day}][{group_name}] 为空 (0 rows)")

    except Exception as e:
        logger.error(f"[{day}][{group_name}] 错误: {str(e)}", exc_info=True)

    del merged_parts


# 主进程
def main():
    logger.info("===== Program started =====")
    create_dir(OUTPUT_DIR)

    # 生成日期列表
    all_dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D").strftime("%Y-%m-%d").tolist()
    logger.info(f"处理日期范围: {all_dates}")
    logger.info(f"使用 {PROCESSES} 进程开始处理...")

    # 分批次处理
    with Pool(
        processes=PROCESSES,
        initializer=init_worker_common_users,
        initargs=(Path(USER_FILE), "Userid"),
    ) as pool:
        pool.map(process_single_day, all_dates)

    logger.info("===== Program completed successfully =====")


if __name__ == "__main__":
    main()
