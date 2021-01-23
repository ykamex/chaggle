import os
import random
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
import torch


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2 
    dfs = []
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dfs.append(df[col].astype(np.int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dfs.append(df[col].astype(np.int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dfs.append(df[col].astype(np.int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dfs.append(df[col].astype(np.int64) ) 
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dfs.append(df[col].astype(np.float16))
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dfs.append(df[col].astype(np.float32))
                else:
                    dfs.append(df[col].astype(np.float64))
        else:
            dfs.append(df[col])
    
    df_out = pd.concat(dfs, axis=1)
    if verbose:
        end_mem = df_out.memory_usage().sum() / 1024**2
        num_reduction = str(100 * (start_mem - end_mem) / start_mem)
        print(f'Mem. usage decreased to {str(end_mem)[:3]}Mb:  {num_reduction[:2]}% reduction')
    return df_out


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_delete_cols(input_df: pd.DataFrame,
                    threshold: float) -> Tuple[List, List, List]:
    """Detect unnecessary columns for deleting
    Args:
        input_df (pd.DataFrame): input_df
        threshold (float): deleting threshold for correlations of columns
    Returns:
        Tuple[List, List, List]: unique_cols, duplicated_cols, high_corr_cols
    """
    unique_cols = list(input_df.columns[input_df.nunique() == 1])
    duplicated_cols = list(input_df.columns[input_df.T.duplicated()])

    buf = input_df.corr()
    counter = 0
    high_corr_cols = []
    try:
        for feat_a in [x for x in input_df.columns]:
            for feat_b in [x for x in input_df.columns]:
                if (
                    feat_a != feat_b
                    and feat_a not in high_corr_cols
                    and feat_b not in high_corr_cols
                ):
                    c = buf.loc[feat_a, feat_b]
                    if c > threshold:
                        counter += 1
                        high_corr_cols.append(feat_b)
                        print(
                            "{}: FEAT_A: {} FEAT_B: {} - Correlation: {}".format(
                                counter, feat_a, feat_b, c
                            )
                        )
    except:
        pass
    return unique_cols, duplicated_cols, high_corr_cols


def get_classified_cols(input_df: pd.DataFrame,
                        t1: int = 10,
                        t2: int = 50,
                        show: bool = False) -> Tuple[List, List, List, List]:
    """DataFrameの列名を以下の通りに分割してprint&returnする
    ※カテゴリ変数にはdatetime型を含む
    1. {t1}未満の要素数のカテゴリ変数
    2. {t1}以上{t2}未満の要素数のカテゴリ変数
    3. {t2}以上の要素数のカテゴリ変数
    4. 数値変数
    
    Args:
        input_df:
            pd.DataFrame
        t1:
            閾値1
        t2:
            閾値2
        show:
            Trueにすると各カテゴリ変数の要素数をSeriesで表示
    
    Returns:
        descの通り。
    
    TODO:
        t1とt2に割合を渡せる様にしても良いかも
        
    """
    _df = input_df[input_df.select_dtypes(['object','category','datetime']).columns].nunique().sort_values()
    
    if show:
        display(_df)
    
    cat_cols_s = list(_df[_df < t1].index)
    cat_cols_m = list(_df[(_df >= t1) & (_df < t2)].index)
    cat_cols_l = list(_df[_df > t2].index)
    num_cols = list(input_df.select_dtypes(['number', 'bool']))
    
    print(f'■ object etc. <{t1} nuique：')
    print(cat_cols_s)
    print()
    
    print(f'■ object etc. {t1}-{t2} nuique：')
    print(cat_cols_m)
    print()
    
    print(f'■ object etc. >{t2} nuique：')
    print(cat_cols_l)
    print()
    
    print('■ number / bool：')
    print(num_cols)
    
    return cat_cols_s, cat_cols_m, cat_cols_l, num_cols


def check_manynull_cols(input_df: pd.DataFrame,
                        null_rate: float = 0.3):
    """欠損の割合が一定以上の列をprintするだけ
    Args:
        input_df(pd.DataFrame): input_df
        null_rate(float): 欠損が全体の何割以上の列を表示するか
    Return:
        nothing
    """
    n = len(input_df) * null_rate
    print(input_df.isnull().sum()[input_df.isnull().sum() > n])