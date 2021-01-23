from chaggle.features.base import BaseBlock
from typing import Union, List, Optional
import pandas as pd
import numpy as np
import category_encoders as ce


#==============================================================================
# Blocks for General Feature Engineering
#==============================================================================
class RawBlock(BaseBlock):
    """input_dfの指定列をそのまま返す"""
    def __init__(self,
                 column: Union[str, List],
                 ):
        """
        Args:
            column(str/list): 返したい列.
        """
        self.column = [column] if isinstance(column, str) else column
  
    def transform(self, input_df):
        return input_df[self.column].copy()


class PortfolioBlock(BaseBlock):
    """レコード or 特徴量間で集計した数値の比率を特徴量化する"""
    def __init__(self,
                 column: Union[str, List],
                 key: Union[str, List],
                 axis: int=1):
        """
        Args:
            column(str/list): 割合を求めたい列
            key(str/list): 集計するための列
            axis(int): [0, 1]を入力
                1:横方向に集計（key毎に正規化）
                    複数の数値列に対して、keyで集計した時の各列の割合.
                    columnが1つしかない場合はすべて1.0になってしまうのでError.
                0:縦方向に集計（column毎に正規化）
                    1つの数値列に対して、keyで集計した時のkeyの値別の割合.
        """
        self.column = [column] if isinstance(column, str) else column
        self.key = [key] if isinstance(key, str) else key
        self.axis = axis
        self.meta_df = None
        
        if (len(self.column)==1)&(self.axis==1):
            raise ValueError(
                'When `axis` is 1, the number of elements in `column` must be greater than 1.'
            )
    
    def fit(self, input_df):
        _df = input_df.groupby(self.key)[self.column].sum()
        
        if self.axis:
            _df = (_df.T / _df.sum(axis=1)).T
            
        else:
            _df = (_df / _df.sum(axis=0))
            
        self.meta_df = _df.reset_index()
        return self.transform(input_df)
    
    def transform(self, input_df):
        output_df = pd.merge(input_df[self.key], self.meta_df,
                          on=self.key, how='left').drop(columns=self.key)
        output_df = output_df.add_prefix(f'PF@{"+".join(self.key)}_')
        return output_df


class SerialBlock(BaseBlock):
    """cumcountでSerial No.をふる.
    基本はkeyの列が全て同じ場合に+1ずつしていくが、
    オプションで、例えば「keyが3個中2個同じの時はカウントを止める」が可能.

    """
    def __init__(self,
                 key: Union[List, str],
                 whole_df: Optional[pd.DataFrame] = None,
                 duplicate: bool = False,
                 primary_idx:int = 1):
        """
        Args:
            key(str/list): Groupbyで使用する集計キー
            whole_df(DataFrame): 任意。設定するとtrain_df/test_dfそれぞれでのシリアルでなくwhole_dfで採番する.
                                 ※duplicate==Trueの場合は入力した方が良い気がする.
            duplicate(bool): 
                Falseだとkeyのすべてが一致するレコードグループ内で+1ずつcumcount.
                Trueだと同じレコードグループの中でlist(key)のprimary_idxにあたる列が
                一致している間はカウントを止める.
                keyの要素が2以下の時はTrueにしても強制的にFalseになる.
                e.g. 県、市、年度までが同じレコードが複数あるデータフレームがある.
                     これらの3つを順にlistにしてkeyに設定した際、Trueの場合には
                     県、市、年度が同じ場合は同じシリアル、年度が変わったときだけシリアルを+1する.
            primary_idx(int):
                duplicate==Trueの時に使う。使われ方は上述の通り.
        
        """
        self.whole_df = whole_df
        self.key = [key] if isinstance(key, str) else key
        self.duplicate = duplicate if len(self.key) > 2 else False
        self.primary_idx = primary_idx
        self.column_name = None
        
    def transform(self, input_df):
        """
        fitを作ってしまうとtest_dfにtrain_dfの結果をmergeすることになってしまう.
        別々のSerialをふりたい状況の方が多いと思われるため、whole_dfの処理もtransformで行う.
        """
        if self.whole_df is None:
            meta_df = self._get_meta_df(input_df)
        else:
            meta_df = self._get_meta_df(self.whole_df)
        
        if self.duplicate:
            output_df = pd.merge(input_df, meta_df, on=self.key,
                                 how='left', sort=False)[self.column_name]
        else:
            output_df = pd.DataFrame()
            _rank = input_df.sort_values(self.key).groupby(self.key).cumcount() + 1
            output_df[self.column_name] = _rank.sort_index()
        
        return output_df
    
    def _get_meta_df(self, input_df):
        if self.duplicate:
            self.column_name = f'dNo@{"+".join(self.key)}'
            _df = input_df.sort_values(self.key)
            ids = _df.groupby(self.key[self.primary_idx]).grouper.group_info[0]
            comb = _df.groupby(self.key).grouper.group_info[0]
            
            count = []
            nth = 0
            
            for i, c in enumerate(comb):
                if i == 0 or ids[i] != ids[i-1]:
                    nth = 1
                elif c != comb[i-1]:
                    nth += 1
                count += [nth]
                
            _df[self.column_name] = count
            output_df = _df[self.key + [self.column_name]].sort_index().drop_duplicates()
            
            return output_df
        
        else:
            self.column_name = f'No@{"+".join(self.key)}'
            return None


class IsxBlock(BaseBlock):
    """真偽値を特徴量にする"""
    def __init__(self,
                 column: Union[str, List],
                 operator: str = '=',
                 comparison: Union[str, int, float, List] = 'null'
                 ):
        """
        Args:
            column: 真偽値判定の対象となる列
            comparison & operator:
                'null'(str) & any input:
                    operatorに関わらず`df.isnull()`の真偽値を返します.
                threshold(int|float) & ['==', '!=', '>', '>=', '<', '<=']:
                    columnの値が演算子の左側にいるものとして真偽判定されます.
                    各演算子はpythonで標準入力した時と同じ意味です.
                    他のすべての入力は'=='として処理されます.
                wordlist(list(str)) & ['isin', 'contain']:
                    それぞれ`df.isin(wordlist)`か`df.astype(str).str.contains(w).astype(int)`を返します.
                    他のすべてのoperatorの入力は'isin'として処理されます.
                NOTE:
                    'null'以外のstrが入力された場合はlistに変換されwordlistとして扱われます.
                    その他想定外の入力があった場合はError（もしくは空のDataFrameが返されます）.
                    真偽値の結果がすべて0もしくは1となる場合は出力されません.

        """
        self.column = [column] if isinstance(column, str) else column
        self.operator = operator
        if (comparison != 'null')&(isinstance(comparison, str)):
            comparison = [comparison]
        self.comparison = comparison
        
    def transform(self, input_df):
        output_df = pd.DataFrame()
        for c in self.column:
            if self.comparison == 'null':
                output_df[f'{c}_isnull'] = input_df[c].isnull().astype(int)

            elif isinstance(self.comparison, (int, float)):
                if self.operator == '!=':
                    output_df[f'{c}!={self.comparison}'] = (input_df[c] != self.comparison).astype(int)
                elif self.operator == '>':
                    output_df[f'{c}>{self.comparison}'] = (input_df[c] > self.comparison).astype(int)
                elif self.operator == '>=':
                    output_df[f'{c}>={self.comparison}'] = (input_df[c] >= self.comparison).astype(int)
                elif self.operator == '<':
                    output_df[f'{c}<{self.comparison}'] = (input_df[c] < self.comparison).astype(int)
                elif self.operator == '<=':
                    output_df[f'{c}<={self.comparison}'] = (input_df[c] <= self.comparison).astype(int)
                else:
                    output_df[f'{c}=={self.comparison}'] = (input_df[c] == self.comparison).astype(int)
            
            elif isinstance(self.comparison, list):
                if self.operator == 'contains':
                    for w in self.comparison:
                        output_df[f'{c}_has_{w}'] = input_df[c].astype(str).str.contains(w).astype(int)
                else:
                    output_df[f'{c}_isin_{"_".join(self.comparison)}'] = input_df[c].isin(self.comparison).astype(int)
        
        return self._delete_no_information_cols(output_df)
    
    def _delete_no_information_cols(self, input_df):
        """全て0/1の列は生成されても削除する"""
        output_df = input_df.copy()
        for c in output_df:
            if (output_df[c].sum() == 0)|(output_df[c].sum() == len(output_df[c])):
                output_df = output_df.drop(c, axis=1)
        return output_df


class GroupbyBlock(BaseBlock):
    """いわゆるaggregate処理をするクラスの親クラス
    …と思ったけど結局全機能を詰め込んでしまっていて子がいない
    TODO:
        - インスタンス化が大変だから分けたいけど、列指定の手間と計算量的にはこれが一番楽な気がする
        - freqをいちいち指定しないでも良い感じの値を選ぶようにしたい
        - 出力の列の順番を計算量できるだけ増やさずに分かり易くしたい

    """   
    def __init__(self,
                 column: Union[str, List],
                 key: Union[str, List],
                 agg: Union[str, List],
                 whole_df: Optional[pd.DataFrame] = None,
                 diff: bool = False,
                 ratio: bool = False,
                 lag: int = 0,
                 lead: int = 0,
                 fmt: Optional[str] = None,
                 freq: Optional[str] = None,
                 handle_missing: Optional[Union[str, int, float]] = None,
                 ts_diff: bool = False,
                 pct_change: bool = False,
                 delete_present_data:bool = False):
        """
        Args:
            column(str/list): 集計する値.columnというかvalue.
            key(str/list) : groupbyの集計key
                NOTE: lag/leadを計算するときは最後のインデックスはdatetime型か数値型を指定すること
            agg(str/list) : 集計関数
            whole_df(DataFrame) : 任意.入力するとwhole_dfでfitする.Noneであればfitへの入力にfitする
            diff(bool): 各レコードの値と集計値との差をとるかどうか.時系列のdiffじゃないので注意
                        columnのdtypeがobjectの場合は強制的にFalseの扱いになる
            ratio(bool): 各レコードの値と集計値との比をとるかどうか
                         columnのdtypeがobjectの場合は強制的にFalseの扱いになる
            lag(int): 何datetime単位分前からのlagをとるか.diffやpct_changeにも適用される.
                      shift()の引数なので大きいほど昔のlagをとる点に注意.
            lead(int): 何datetime単位分後までのleadをとるか.diffやpct_changeにも適用される.
                       shift()の引数なので小さいほど将来のleadをとる点に注意.
            fmt(str): `pd.to_datetime()`に渡す引数。どういう形式の数字をdatetime型に修正するかの指定
                      None(default)の場合は関数が自動的に判別してくれるが、年/月/日だけとかの判別は厳しい
                      reference: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
            freq(str): `pd.date_range()`に渡す引数。存在しない期間をどの間隔で埋めるかの指定
                       reference: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
            handle_missing(int/float/str): レコードに存在しない期間のデータをどう扱うか.
                                           数値型はfillnaのvalueに、str型はmethodに渡される.
            ts_diff(bool): 時系列のdiffをとるかどうか.lagとleadが両方0だとTrueでも無意味
            pct_change(bool): 時系列のpct_changeをとるかどうか.lagとleadが両方0だとTrueでも無意味
            delete_present_data(bool): lag系特徴量を作るが現時点の時系列の情報は消したいときにTrueにする
        NOTE:
            shift関係の特徴量は、「他のGroupにはあるけどそのGroupではレコードが無い時系列」を補完した上で
            ラグを作成しています.各Groupの時系列の連続性が飛び飛びの場合はNaNだらけになる可能性があります.
            reference: https://analytics-note.xyz/programming/dataframe-complement/
        """
        self.column = [column] if isinstance(column, str) else column
        self.key = [key] if isinstance(key, str) else key 
        self.agg = [agg] if isinstance(agg, str) else agg
        self.whole_df = whole_df
        self.features = None
        # Variables for extra fe
        self.diff = diff
        self.ratio = ratio
        self.lag = lag
        self.lead = lead
        self.fmt = fmt
        self.freq = freq
        self.handle_missing = handle_missing
        self.ts_diff = ts_diff
        self.pct_change = pct_change
        self.delete_present_data = delete_present_data
    
    def _aggregate(self, input_df):
        all_features = list(set(self.key + self.column))
        new_features = self._get_feature_names()
        if self.whole_df is None:
            features = input_df[all_features].groupby(self.key)[self.column].agg(self.agg).reset_index()
            features.columns = self.key + new_features
        else:
            features = self.whole_df[all_features].groupby(self.key)[self.column].agg(self.agg).reset_index()
            features.columns = self.key + new_features
        if (self.lag != 0)|(self.lead != 0):
            features = self._fill_features(features)
            self.features = self._get_shift_features(features)
        else:
            self.features = features
    
    def fit(self, input_df):
        self._aggregate(input_df)
        return self.transform(input_df)
    
    def transform(self, input_df):
        output_df = input_df.copy()
        if (self.lag != 0)|(self.lead != 0):
            output_df[self.key[-1]] = pd.to_datetime(output_df[self.key[-1]], format=self.fmt)
        output_df = output_df.merge(self.features, how="left", on=self.key)

        if (self.diff)|(self.ratio):
            try:
                output_df = self._get_diff_ratio_features(output_df)
            except:
                raise ValueError('diff/ratioがとれないよ.columnに数値じゃなくてカテゴリが混ざってない？')
        
        output_df = output_df.drop(input_df.columns, axis=1)
        if self.delete_present_data:
            output_df = output_df.drop(self._get_feature_names(), axis=1)
        return output_df
    
    def _get_feature_names(self):
        _agg = []
        for a in self.agg:
            if not isinstance(a, str):
                _agg.append(a.__name__)
            else:
                _agg.append(a)
        return [f'{"_".join([a, c])}@{"+".join(self.key)}' for c in self.column for a in _agg]
    
    # extra functions
    def _get_diff_ratio_features(self, input_df):
        cnt = 0
        for c in self.column:
            for a in self.agg:
                agg_name = a.__name__ if not isinstance(a, str) else a
                if self.diff:
                    input_df[f'diff_{agg_name}_{c}@{"+".join(self.key)}'] = input_df[c] - input_df[self._get_feature_names()[cnt]]
                if self.ratio:
                    input_df[f'ratio_{agg_name}_{c}@{"+".join(self.key)}'] = input_df[c] / input_df[self._get_feature_names()[cnt]]
                cnt += 1
        return input_df

    def _fill_features(self, features):
        features[self.key[-1]] = pd.to_datetime(features[self.key[-1]], format=self.fmt)
        
        start_dt = features[self.key[-1]].min()
        end_dt = features[self.key[-1]].max()
        datetime_range = pd.date_range(start_dt, end_dt, freq=self.freq)
        some_of_uniques = [features[c].unique() for c in features[self.key[:-1]]]
        
        mesh = np.meshgrid(*some_of_uniques, datetime_range)
        _df = pd.DataFrame([c.ravel() for c in mesh], index=self.key).T
        features = _df.merge(features, how='left', on=self.key)
        return features
    
    def _get_shift_features(self, features):
        _df = features.drop(self.key[-1], axis=1).copy()
        fill_params = {'value': self.handle_missing if isinstance(self.handle_missing,(int, float))
                                                    else np.nan if self.handle_missing is None else None,
                       'method': self.handle_missing if isinstance(self.handle_missing, str) else None
                      }
        for i in range(self.lead, self.lag+1):
            if i != 0:
                _shift = _df.groupby(self.key[:-1]).transform(lambda x: x.fillna(**fill_params).shift(i)).add_prefix(f'Shift{i}_')
                features = pd.concat([features, _shift], axis=1)
                if self.ts_diff:
                    _ts_diff = _df.groupby(self.key[:-1]).transform(lambda x: x.fillna(**fill_params).diff(i)).add_prefix(f'ts_diff{i}_')
                    features = pd.concat([features, _ts_diff], axis=1)
                if self.pct_change:
                    _pct_change = _df.groupby(self.key[:-1]).transform(lambda x: x.fillna(**fill_params).pct_change(i)).add_prefix(f'pct_change{i}_')
                    features = pd.concat([features, _pct_change], axis=1)
        return features


class RollingBlock(GroupbyBlock):
    """aggregateしたうえで移動平均を求める
    NOTE:
        すべてrolling(window).mean()を計算しており、agg()を使っていないことに注意
    TODO:
        - 平均変化率求められる様にしたい（別のBlockにした方が良いかも）
        - transformを使うことでfillnaが上手く出来る様になったけど、rollingしたあとのaggが使えなくなったからどうにかしたい
    """
    def __init__(self,
                 column: Union[str, List],
                 key: Union[str, List],
                 agg: Union[str, List],
                 whole_df: Optional[pd.DataFrame] = None,
                 window: int = 5,
                 shift: int = 1,
                 fmt: Optional[str] = None,
                 freq: Optional[str] = None,
                 handle_missing: Optional[Union[str, int, float]] = None):
        """
        Args:
            column(str/list): 集計する値.columnというかvalue.
            key(str/list) : groupbyの集計key
                NOTE: 最後のインデックスはdatetime型か数値型を指定すること
            agg(str/list) : 集計関数（集計時使用）
            whole_df(DataFrame) : 任意.入力するとwhole_dfでfitする.Noneであればfitへの入力にfitする
            window(int): rollingの引数windowの値.rolling前にshift()がかかってるのでself.shiftからのrollingなことに注意.
            shift(int): rollingまでにどれだけshiftするかの値.
            fmt(str): `pd.to_datetime()`に渡す引数。どういう形式の数字をdatetime型に修正するかの指定
                      None(default)の場合は関数が自動的に判別してくれるが、年/月/日だけとかの判別は厳しい
                      reference: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
            freq(str): `pd.date_range()`に渡す引数。存在しない期間をどの間隔で埋めるかの指定
                       reference: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
            handle_missing(int/float/str): レコードに存在しない期間のデータをどう扱うか.
                                           数値型はfillnaのvalueに、str型はmethodに渡される.
        NOTE:
            rolling時、「他のGroupにはあるけどそのGroupではレコードが無い時系列」を補完しています.
            各Groupの時系列の連続性が飛び飛びの場合はNaNだらけになる可能性があります.

        """
        self.column = [column] if isinstance(column, str) else column
        self.key = [key] if isinstance(key, str) else key 
        self.agg = [agg] if isinstance(agg, str) else agg
        self.whole_df = whole_df
        self.features = None
        self.window = window
        self.shift = shift
        self.fmt = fmt
        self.freq = freq
        self.handle_missing = handle_missing
    
    def _aggregate(self, input_df):
        if self.whole_df is None:
            features = self._get_features(input_df)
        else:
            features = self._get_features(self.whole_df)
        self.features = features
        
    def _get_features(self, input_df):
        all_features = list(set(self.key + self.column))
        new_features = self._get_feature_names()
        fill_params = {'value': self.handle_missing if isinstance(self.handle_missing,(int, float))
                                                    else np.nan if self.handle_missing is None else None,
                       'method': self.handle_missing if isinstance(self.handle_missing, str) else None
                      }        
        features = input_df[all_features].groupby(self.key)[self.column].agg(self.agg).reset_index()
        features.columns = self.key + new_features
        features = self._fill_features(features)
        
        group = features.groupby(self.key[:-1])[new_features]
        meta_df = group.transform(lambda x: x.fillna(**fill_params).shift(self.shift).rolling(self.window).mean()).reset_index(drop=True)
        meta_df.columns = [f'Rolling{self.window}_shift{self.shift}_mean_of_{c}' for c in self._get_feature_names()]
        features = pd.concat([features[self.key], meta_df], axis=1)
        return features
    
    def transform(self, input_df):
        output_df = input_df.copy()
        output_df[self.key[-1]] = pd.to_datetime(output_df[self.key[-1]], format=self.fmt)
        output_df = output_df.merge(self.features, how="left", on=self.key)
        return output_df.drop(input_df.columns, axis=1)
    
    def _get_diff_ratio_features(self, input_df):
        raise NotImplementedError()
        
    def _get_shift_features(self, input_df):
        raise NotImplementedError()


#==============================================================================
# Blocks for Category Encoding
#==============================================================================


class LabelEncodingBlock(BaseBlock):
    """For LabelEncoding"""
    def __init__(self,
                 column: Union[str, List],
                 whole_df: pd.DataFrame = None,
                 handle_unknown: str = 'value',
                 handle_missing: str = 'value'):
        """
        Args:
            column(str,list): encode column(s)
            whole_df(DataFrame): 入力するとwhole_dfにfit.Noneのままならtrainにfit.
            handle_unknown(str): fit時に無かった値がtransformで出てきたらどうするか
                                'value': default.-1に変換する
                                'error': raise ValueError
                                'return_nan': nanに変換する
            handle_missing(str): nanをどう変換するか
                                'value': default.一つのカテゴリとして変換する
                                        fit時に無くtransformで出てきた場合は-2に変換する
                                'error': raise ValueError
                                'return_nan': nanに変換する
        reference:
            https://contrib.scikit-learn.org/category_encoders/ordinal.html
                
        """
        self.column = column
        self.whole_df = whole_df
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.encoder = None
        
    def fit(self, input_df):
        self.encoder = ce.OrdinalEncoder(cols=self.column,
                                         handle_unknown=self.handle_unknown,
                                         handle_missing=self.handle_missing)
        if self.whole_df is None:
            self.encoder.fit(input_df[self.column])
        else:
            self.encoder.fit(self.whole_df[self.column])
        return self.transform(input_df)
    
    def transform(self, input_df):
        output_df = self.encoder.transform(input_df[self.column]).add_prefix('LE_')
        return output_df


class CountEncodingBlock(BaseBlock):
    """For CountEncoding"""
    def __init__(self,
                 column: Union[str, List],
                 whole_df: pd.DataFrame = None,
                 handle_unknown: Optional[Union[int, str]] = None,
                 handle_missing: str = 'count',
                 min_group_size: Optional[int] = None,):
        """
        Args:
            column(str,list): encode column(s)
            whole_df(DataFrame): 入力するとwhole_dfにfit.Noneのままならtrainにfit.
            handle_unknown(str,int): fit時に無かった値がtransformで出てきたらどうするか
                                    None: default.nanに変換する
                                    'error': raise ValueError
                                    'return_nan': nanに変換する
                                    int: 入力値に変換する
            handle_missing(str): nanをどう変換するか
                                'count': default.一つのカテゴリとして変換する
                                'error': raise ValueError
                                'return_nan': nanに変換する
            min_group_size(int): 入力数以下のカウントとなるカテゴリはまとめてカウントする
        reference:
            https://contrib.scikit-learn.org/category_encoders/count.html

        """
        self.column = column
        self.whole_df = whole_df
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing
        self.min_group_size = min_group_size
        self.encoder = None
        
    def fit(self, input_df):
        self.encoder = ce.CountEncoder(cols=self.column,
                                       handle_unknown=self.handle_unknown,
                                       handle_missing=self.handle_missing,
                                       min_group_size=self.min_group_size
                                       )
        if self.whole_df is None:
            self.encoder.fit(input_df[self.column])
        else:
            self.encoder.fit(self.whole_df[self.column])
        return self.transform(input_df)
    
    def transform(self, input_df):
        output_df = self.encoder.transform(input_df[self.column]).add_prefix('CE_')
        return output_df


class OneHotEncodingBlock(BaseBlock):
    """For OneHotEncoding"""
    def __init__(self,
                 column: Union[str, List],
                 whole_df: pd.DataFrame = None,
                 handle_missing: str = 'value',
                 min_freq: Union[int, float] = 0,
                 max_columns: Optional[int] =None):
        """
        Args:
            column(str,list): encode column(s)
            whole_df(DataFrame): 入力するとwhole_dfにfit.Noneのままならtrainにfit.
            handle_missing(str): nanをどう変換するか
                                'value': default.エンコード対象に加える
                                'error': raise ValueError.不正な値はすべて'error'入力と見なす
                                         nanが無かった場合はvalueと同様の処理
                                'return_nan': エンコード対象から除外する
            min_freq(int/float): エンコードの対象になるのに必要な最低出現数/率
                                 1以上の数をを渡すとint(freq)以上の出現数のもの、
                                 0-1の小数で渡すとfreq率以上の出現数のものをエンコードする.
            max_columns(int): エンコードの対象とする最大要素数（＝最大生成列数/column）。
        TODO:
            handle_unknownの実装
        """
        self.column = [column] if isinstance(column, str) else column
        self.whole_df = whole_df
        self.handle_missing = handle_missing
        self.min_freq = min_freq
        self.max_columns = max_columns
        self.cats_ = None
        
    def fit(self, input_df):
        if self.whole_df is None:
            self.cats_ = self._get_cats(input_df)
        else:
            self.cats_ = self._get_cats(self.whole_df)
        
        return self.transform(input_df)
    
    def transform(self, input_df):
        output_df = pd.DataFrame()
        for c, idx in self.cats_.items():
            x = pd.Categorical(train[c].fillna('NaN'), categories=idx)
            _df = pd.get_dummies(x, dummy_na=False)
            _df = _df.add_prefix(f'OH_{c}=')
            output_df = pd.concat([output_df, _df], axis=1)
        return output_df

    def _get_cats(self, input_df):
        cats = {}
        for c in self.column:
            # handle_missingの値次第で種類の違うvalue_counts
            if self.handle_missing == 'value':
                vc = input_df[c].fillna('NaN').value_counts()
            elif self.handle_missing == 'return_nan':
                vc = input_df[c].value_counts()
            else:
                if input_df[self.column].isnull().any().any():
                    raise ValueError('Columns to be encoded can not contain null')
                else:
                    vc = input_df[c].value_counts()
            
            # min_freq以上の出現数のものに絞る
            if 0 < self.min_freq < 1.:
                min_count = len(input_df) * self.min_freq
            elif self.min_freq <= 0:
                min_count = 0
            else:
                min_count = int(self.min_freq)

            cat = vc[vc >= min_count].index
            
            # 列数がmax_columns以下になる様に絞る
            if self.max_columns is not None and len(cat) > self.max_columns:
                n = max(0, self.max_columns)
                n = np.floor(n)
                cat = cat[:int(n)]
            
            cats[c] = cat
        return cats


class TargetEncodingBlock(BaseBlock):
    """For TargetMeanEncoding with CV"""
    def __init__(self,
                 column: Union[str, List],
                 target: str,
                 cv: List,
                 handle_missing: str = 'return_nan',
                 smoothing: float = 1.0,
                 min_samples_leaf:int = 1):
        """
        Args:
            column(str/list): encode column(s)
            target(str): target
            cv(list): cross validation split
            handle_missing(str): nanをどう変換するか
                                 'value': エンコード対象に加える
                                 'error': raise ValueError.不正な値はすべて'error'入力と見なす
                                          nanが無かった場合はreturn_nanと同様の処理
                                 'return_nan': default.エンコード対象から除外する
            smoothing(float): smoothingの強さ.overfitを避けるため出現数が少ないカテゴリは全体の平均に近づける.
                              ざっくり説明: どんなに大きくしてもそのカテゴリの平均と全体の平均の中間地点で落ち着く.
            min_samples_leaf(int): こちらもsmoothingの強さ.ざっくり説明: 大きいと全体平均に一気に近づく.
        NOTE:
            smoothingの影響をほぼゼロにしたければ、smoothing=1にしてmin_samples_leafを-np.infにすれば良い
        TODO:
            handle_unknownの実装
        
        """
        self.column = [column] if isinstance(column, str) else column
        self.target = target
        self.cv = cv
        self.handle_missing = handle_missing
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.train_df = None
        
    def fit(self, input_df):
        self.train_df = input_df.copy()
        output_df = pd.DataFrame()
        for c in self.column:
            tmp = np.repeat(np.nan, input_df.shape[0])
            for idx_1, idx_2 in self.cv:
                mapping = self._get_mapping(input_df.iloc[idx_1], c)
                if self.handle_missing == 'value':
                    tmp[idx_2] = input_df[c].fillna('NaN').iloc[idx_2].map(mapping)
                else:
                    tmp[idx_2] = input_df[c].iloc[idx_2].map(mapping)
            output_df[f'TE_{c}'] = tmp
        return output_df
    
    def transform(self, input_df):
        """fitみたいなことやっててイケてないけどこっちの方がコードが楽なので…"""
        output_df = pd.DataFrame()
        for c in self.column:
            mapping = self._get_mapping(self.train_df, c)
            if self.handle_missing == 'value':
                output_df[f'TE_{c}'] = input_df[c].fillna('NaN').map(mapping)
            else:
                output_df[f'TE_{c}'] = input_df[c].map(mapping)
        return output_df
        
    def _get_mapping(self, input_df, col):
        if self.handle_missing == 'value':
            _df = pd.DataFrame({col: input_df[col].fillna('NaN'), 'target': input_df[self.target]})
        elif self.handle_missing == 'return_nan':
            _df = pd.DataFrame({col: input_df[col], 'target': input_df[self.target]})
        else:
            if input_df[self.column].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')
            else:
                _df = pd.DataFrame({col: input_df[col], 'target': input_df[self.target]})
            
        prior = _df['target'].mean()
        stats = _df['target'].groupby(_df[col]).agg(['count', 'mean'])
        smoove = 1 / (1 + np.exp(-(stats['count'] - self.min_samples_leaf) / self.smoothing))
        mapping = prior * (1 - smoove) + stats['mean'] * smoove

        return mapping