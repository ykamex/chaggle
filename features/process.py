from contextlib import contextmanager
from time import time
from tqdm import tqdm
import pandas as pd


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


def get_function(block, is_train):
    s = mapping = {
        True:'fit',
        False:'transform'
    }.get(is_train)
    return getattr(block, s)


def add_feature(input_df, blocks, is_train=False):
    """
    DataFrameにblocksの処理により新しい列を追加する（input_dfに追加してReturn）

    """
    output_df = input_df.copy()
    
    for block in tqdm(blocks, total=len(blocks)):
        func = get_function(block, is_train)
        
        with timer(prefix='create ' + str(block) + ' '):
            _df = func(input_df)
        
        assert len(_df) == len(input_df), func.__name__
        output_df = pd.concat([output_df, _df], axis=1)
        
    return output_df


def to_feature(input_df, blocks, is_train=False):
    """
    DataFrameからblocksの処理により新しい特徴量を作る（input_dfとは別物をReturn）
	
    """
    output_df = pd.DataFrame()
    
    for block in tqdm(blocks, total=len(blocks)):
        func = get_function(block, is_train)
        
        with timer(prefix='create ' + str(block) + ' '):
            _df = func(input_df)
            
        assert len(_df) == len(input_df), func.__name__
        output_df = pd.concat([output_df, _df], axis=1)
        
    return output_df