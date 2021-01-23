import os
import csv


class BaseBlock(object):
    """（基本的に）全Blockの親クラス"""
    def fit(self, input_df, y=None):
        return self.transform(input_df)

    def transform(self, input_df):
        raise NotImplementedError()

    def create_memo(self, col_name, desc):
        """
        BaseBlockを継承したBlockから作成された特徴量のメモをcsvに残す関数
        そのうち使うかも知れないのでとりあえず実装した
        使うなら__init__()かfit()で呼び出し必須にするつもり
        """
        file_path = './_features_memo.csv'
        if not os.path.isfile(file_path):
            with open(file_path,"w", encoding="utf-8-sig", errors='ignore') as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(['Feature', 'Memo'])
        
        with open(file_path, 'r+', encoding="utf-8-sig", errors='ignore') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            
            if isinstance(col_name, str):
                col = [line for line in lines if line.replace('\ufeff','').split(',')[0] == col_name]
                if len(col) != 0:return

                writer = csv.writer(f, lineterminator="\n")
                writer.writerow([col_name, desc])
            
            elif isinstance(col_name, list):
                for c in col_name:
                    col = [line for line in lines if line.replace('\ufeff','').split(',')[0] == c]
                    if len(col) != 0:continue

                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerow([c, desc])
            else:
                raise ValueError(f'col_name must be str/list, input type:{type(col_name)} ')

    def save(self, train_df=None, test_df=None, need_fit=False):
        """
        仮で作ってあるもの。そのうちちゃんと実装したい。
        今のところは全特徴量再計算で良いが、稀に激重の計算をすることがあるので
        もしもの時にとりあえず雑でも作成したoutputをpkl保存するやつ。
                
        """
        if train_df is not None:
            if need_fit:
                self.fit(train_df).to_pickle('temp_feature_train.pkl')
            else:
                self.transform(train_df).to_pickle('temp_feature_train.pkl')
                
            if test_df is not None:
                self.transform(test_df).to_pickle('temp_feature_test.pkl')
        else:
            raise NotImplementedError

            
class WrapperBlock(BaseBlock):
    """関数をBlock処理するためのクラス
    どうしても各コンペ専用の特徴量作成関数を作成する必要があった時に使用
    可能な限り処理を一般化してBlockにするのが基本
    
    """
    def __init__(self, function):
        self.function = function
        
    def transform(self, input_df):
        return self.function(input_df)