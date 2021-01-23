import sys
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, KFold, GroupKFold
from matplotlib import pyplot as plt
import seaborn as sns

class MovingWindowKFold(TimeSeriesSplit):
    """時系列情報が含まれるカラムでソートした iloc を返す KFold
    Example:
    >>> folds = MovingWindowKFold(ts_column='year-month', n_splits=5)
    >>> for i, (train_index, test_index) in enumerate(folds.split(df)):
    >>> ...

    """
    def __init__(self, ts_column, clipping=False, n_splits=5, *args, **kwargs):
        """
        Args:
            ts_column(str): 時系列データのカラムの名前
            clipping(bool): 得られる添字のリストの長さを過去最小の Fold に揃えるフラグ
            */** : sklearnのTimeSeriesSplitに渡したいもの何でも
            
        """
        super().__init__(*args, **kwargs)
        self.ts_column = ts_column
        self.clipping = clipping
        self.n_splits = n_splits

    def split(self, X, *args, **kwargs):
        # 渡されるデータは DataFrame を仮定する
        assert isinstance(X, pd.DataFrame)

        # clipping が有効なときの長さの初期値
        train_fold_min_len, test_fold_min_len = sys.maxsize, sys.maxsize

        # 時系列のカラムを取り出す
        ts = X[self.ts_column]
        # 元々のインデックスを振り直して iloc として使える値 (0, 1, 2...) にする
        ts_df = ts.reset_index()
        # 時系列でソートする
        sorted_ts_df = ts_df.sort_values(by=self.ts_column)
        # スーパークラスのメソッドで添字を計算する
        for train_index, test_index in super().split(sorted_ts_df, *args, **kwargs):
            # 添字を元々の DataFrame の iloc として使える値に変換する
            train_iloc_index = sorted_ts_df.iloc[train_index].index
            test_iloc_index = sorted_ts_df.iloc[test_index].index

            if self.clipping:
                # TimeSeriesSplit.split() で返される Fold の大きさが徐々に大きくなることを仮定している
                train_fold_min_len = min(train_fold_min_len, len(train_iloc_index))
                test_fold_min_len = min(test_fold_min_len, len(test_iloc_index))

            yield list(train_iloc_index[-train_fold_min_len:]), list(test_iloc_index[-test_fold_min_len:])
            
    def plot_line(self, input_df, target):
        fig, axes = plt.subplots(self.n_splits, 1, figsize=(12, 12))
        for i, (train_index, test_index) in enumerate(self.split(input_df)):
            sns.lineplot(data=input_df, x=self.ts_column, y=target, ax=axes[i], label='original')
            sns.lineplot(data=input_df.iloc[train_index], x=self.ts_column, y=target, ax=axes[i], label='train')
            sns.lineplot(data=input_df.iloc[test_index], x=self.ts_column, y=target, ax=axes[i], label='test')

        plt.legend()
        plt.show()
        

def create_folds(input_df, validation="kf", fold=5, random_state=0, column:str=None):
    """
    Args:
        validation : 'kf'  for KFold
                     'gkf' for GroupKFold
                     'skf' for StratifiedKFold
        column  : col for groups if 'gkf', target_col if 'skf'

    """
    if validation == "skf":
        # 今の実装はbinningが前提
        # binningしないときはsplitのyにtargetを与える様に修正
        num_bins = np.int(1 + np.log2(len(input_df)))
        bins = pd.cut(input_df[column],
                     bins = num_bins,
                     labels=False
                     )
        skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
        fold_ids = list(skf.split(X=data, y=bins.values))
    
    if validation == "kf":
        kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
        fold_ids = list(kf.split(input_df))
        
    if validation == "gkf":
        ### Shuffleできるver
        # https://github.com/ghmagazine/kagglebook/blob/master/ch05/ch05-01-validation.py
        ids = input_df[column]
        unique_ids = ids.unique()
        kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
        fold_ids = []
        for tr_group_idx, va_group_idx in kf.split(unique_ids):
            tr_groups = unique_ids[tr_group_idx]
            va_groups = unique_ids[va_group_idx]
            fold_ids.append((np.array(ids[ids.isin(tr_groups)].index),
                             np.array(ids[ids.isin(va_groups)].index)))

        ### Shuffleできなくて使いづらいver(しかもgroupにnanがあるとErrorになる)
        # gkf = GroupKFold(n_splits=fold)
        # fold_ids = list(gkf.split(X=input_df, groups=input_df[column]))
    
    return fold_ids