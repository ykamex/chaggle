import itertools
from typing import List, Union, Optional, Tuple
import pandas as pd


def combine_cols(input_df: pd.DataFrame,
                 column: List[str],
                 r: int = 2,
                 fillna: bool = False) -> pd.DataFrame:
    """複数列を文字列として結合した列をDaraFrameに追加する
    Args:
        input_df(DaraFrame): input_df 
        column(list): リストで渡す.r個の組み合わせで新規列を作る
        r(int): [-1,0,2以上の整数r]を入力する
                2以上の整数の場合はｒ個の組み合わせを作る
                0の場合はリストの先頭を固定して、他の要素との2個の組み合わせをlen(column)-1個作る
                -1の場合はリストの順序を変えずに上から順次加えていった順列をlen(column)-1個作る
        fillna(bool):
            Trueにすると、組み合わせ列の中にひとつでも欠損があると生成列でも欠損になる
            Falseの場合は'unk'で欠損が補完される
    Returns:
        input_dfに新しくできた列を結合したDataFrame
    """
    if fillna:
        _df = input_df[column].copy()
    else:
        _df = input_df[column].copy().fillna('unk')
    
    if (r > 1)&(isinstance(r, int)):
        cols_comb = [list(i) for i in itertools.combinations(column, r)]
    elif r == 0:
        cols_comb = []
        for i in range(1,len(column)):
            cols_comb.append([column[0], column[i]])
    elif r == -1:
        cols_comb = []
        for i in range(2,len(column)+1):
            cols_comb.append(column[:i])
    else:
        raise ValueError(f'`r` must be an integer greater than or equal to 2 or -1. input:{r}')
            
    for col in cols_comb:
        _df['_'.join(col)] = _df[col[0]].astype(str)
        for c in col[1:]:
            _df['_'.join(col)] += '_' + _df[c].astype(str)
    
    _df = _df.drop(column, axis=1)
            
    return pd.concat([input_df, _df], axis=1)


def fillna(train_df: pd.DataFrame,
           test_df: pd.DataFrame,
           column: Union[str, List],
           how: str = 'mean',
           whole:bool = False,
           key: Optional[str] = None)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """ちょっと便利に欠損値補完する
    Args:
        train_df(DataFrame): train_df
        test_df(DataFrame): test_df
        column(str/list): 欠損を補完したい列
        how(str): ['mean', 'median']から選択
        whole(bool): Trueにすると、train_dfとtest_dfを結合した上で平均/中央値を計算する
        key(str): 設定すると、keyでgroupbyしたgroupでの平均/中央値を計算して埋める
                  keyに欠損があるとエラー.key2つ以上の設定をするとエラー.
    Returns:
        欠損を埋めたtrain_dfとtest_df
    
    """   
    column = [column] if isinstance(column, str) else column
    
    # 関数内関数: 一つのDataFrameの欠損を埋める
    def fill_one_df(input_df):
        output_df = input_df.copy()
        if key is None:
            for c in column:
                if how == "median":
                    output_df[c].fillna(output_df[c].median(), inplace=True)
                elif how == "mean":
                    output_df[c].fillna(output_df[c].mean(), inplace=True)
        else:
             for c in column:
                if how == "median":
                    f = lambda x: x.fillna(x.median())
                    output_df[c] = output_df[[c] + [key]].groupby(key).transform(f)
                elif how == "mean":
                    f = lambda x: x.fillna(x.mean())
                    output_df[c] = output_df[[c] + [key]].groupby(key).transform(f)

        return output_df
    
    if whole:
        meta_df = pd.concat([train_df, test_df])
        meta_df = fill_one_df(input_df=meta_df)
        output_train = meta_df[:len(train_df)]
        output_test = meta_df[len(train_df):]
    else:
        output_train = fill_one_df(input_df=train_df)
        output_test = fill_one_df(input_df=test_df)

    return output_train, output_test


def datetime_parser(input_df: pd.DataFrame,
                    column: Union[str, List]) -> pd.DataFrame:
    """時系列情報を分割して新たな列に加える
    Args:
        input_df(DataFrame): input_df
        column(list/str): column
    Returns:
        input_dfにdatetimeを分割した列を追加したDataFrame
    """
    output_df = input_df.copy()
    column = [column] if isinstance(column, str) else column

    for c in column:
        output_df[c + "_year"] = pd.to_datetime(output_df[c]).dt.year
        output_df[c + "_quater"] = pd.to_datetime(output_df[c]).dt.quarter
        output_df[c + "_month"] = pd.to_datetime(output_df[c]).dt.month
        output_df[c + "_weekofyear"] = pd.to_datetime(output_df[c]).dt.isocalendar().week
        output_df[c + "_weekofmonth"] = pd.to_datetime(output_df[c]).dt.isocalendar().week\
                                        - output_df[c].astype('datetime64[M]').dt.isocalendar().week + 1
        output_df[c + "_day"] = pd.to_datetime(output_df[c]).dt.day
        output_df[c + "_dow"] = pd.to_datetime(output_df[c]).dt.dayofweek
        output_df[c + "_hour"] = pd.to_datetime(output_df[c]).dt.hour
        output_df[c + "_minute"] = pd.to_datetime(output_df[c]).dt.minute
    
    return output_df