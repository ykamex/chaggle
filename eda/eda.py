import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince
from pandasgui import show
from matplotlib_venn import venn2
from pygam import GAM


def display_stats_df(input_df):
    print("===============================================================================================================")
    print('# stats_df')
    print("===============================================================================================================")
    stats = []
    for col in input_df.columns:
        if not input_df[col].isnull().all():
            stats.append((col,
                          input_df[col].count(),
                          input_df[col].isnull().sum() * 100 / input_df.shape[0],
                          input_df[col].nunique(),
                          input_df[col].value_counts().index[0],
                          input_df[col].value_counts().values[0],
                          input_df[col].value_counts(normalize=True, dropna=False).values[0] * 100,
                          input_df[col].dtype))

        else:
            stats.append((col, 0, 100, 0, 0, 0, 100, input_df[col].dtype))
    stats_df = pd.DataFrame(stats, columns=['Feature', 'Non null', 'null%', 'Nunique', 'Most frequent item', 'Freq of most frequent item',  '% of most frequent item', 'Type'])
    stats_df.sort_values('null%', ascending=False)
    display(stats_df.style.background_gradient(cmap='Blues'))

    
def display_null(input_df):
    print("===============================================================================================================")
    print('# null')
    print("===============================================================================================================")
    plt.figure(figsize=(18,4))
    sns.heatmap(input_df.isnull(), cbar=False)
    plt.show()
    

def display_raw(input_df, nrows=10, threshold=1):
    print("===============================================================================================================")
    print('# raw')
    print("===============================================================================================================")
    _df = input_df.copy()
    cat_cols = _df.select_dtypes(['object', 'category', 'datetime']).columns.tolist()
    num_cols = _df.select_dtypes(['number', 'bool']).columns.tolist()
    chk_cols = cat_cols + num_cols

    # カテゴリ変数は出現順にLabelEncoding
    for c in cat_cols:
        _dict = {j:i for i,j in enumerate(_df[c].unique())}
        _df[c] = _df[c].map(_dict)
    
    # 数値変数を上限はthresholdでclipping/下限は逆数でclipping
    for c in num_cols:
        p = _df[c].quantile(threshold)
        inv_p = _df[c].quantile(1-threshold)
        _df[c] = _df[c].clip(inv_p, p)
        
    nrows = nrows
    ncols = len(_df) % nrows

    for c in chk_cols:
        fig,ax = plt.subplots(1, 1, figsize=(18,4))
        sns.heatmap(np.array(_df.loc[ncols:,c]).reshape(nrows,-1))
        ax.set_title(c)
        ax.axis("off")
        plt.show()
    plt.close()
        
        
def display_hist(input_df, ncols=3):
    print("===============================================================================================================")
    print('# histgram')
    print("===============================================================================================================")
    nrows = (len(list(input_df.select_dtypes('number')))) // ncols + 1
    fig, axes = plt.subplots(figsize=(4 * ncols, 4 * nrows), nrows=nrows, ncols=ncols)

    for c, ax in zip(list(input_df.select_dtypes('number')), np.ravel(axes)):
        sns.histplot(input_df[c], ax=ax, label=c)
        ax.set_xlabel('')
        ax.legend()
        ax.grid()

    fig.tight_layout()
    display(fig)
    plt.close()
    

def display_raincloud(input_df, scaling=True):
    print("===============================================================================================================")
    print('# raincloud')
    print("===============================================================================================================")
    _df = input_df.copy()
    if scaling:
        _df = (_df - _df.mean()) / _df.std()
        
    cols = list(_df.select_dtypes('number'))
    fig, ax = plt.subplots(figsize=(18, 6))
    ptitprince.RainCloud(data=_df[cols], ax=ax, orient='v')
    fig.tight_layout()
    display(fig)
    plt.close()
    
    
def display_corr_df(input_df):
    print("===============================================================================================================")
    print('# corr_df')
    print("===============================================================================================================")
    _df = input_df.corr()
    display(_df.style.background_gradient(cmap='Blues'))
    
    
def display_common_df(train_df, test_df):
    print("===============================================================================================================")
    print('# common_df')
    print("===============================================================================================================")
    output_df = pd.DataFrame()
    common_cols = [c for c in list(train_df) if c in list(test_df)]
    output_df['column'] = common_cols

    output_df['notnull_train'] = output_df['column'].map(train_df[common_cols].notnull().sum())
    output_df['notnull_test'] = output_df['column'].map(test_df[common_cols].notnull().sum())

    output_df['nunique_train'] = output_df['column'].map(train_df[common_cols].nunique())
    output_df['nunique_test'] = output_df['column'].map(test_df[common_cols].nunique())

    output_df[['common', 'common_volume_train', 'common_volume_test']] = np.nan
    for i, c in enumerate(common_cols):
        common_set = set(train_df[c].dropna()) & set(test_df[c].dropna())
        output_df.loc[i, 'common'] = len(common_set)
        output_df.loc[i, 'common_volume_train'] = train_df[c].isin(common_set).sum()
        output_df.loc[i, 'common_volume_test'] = test_df[c].isin(common_set).sum()

    output_df[['common', 'common_volume_train', 'common_volume_test']] =\
        output_df[['common', 'common_volume_train', 'common_volume_test']].astype(int)

    output_df['common_per_train'] = output_df['common_volume_train'] / len(train_df)
    output_df['common_per_test'] = output_df['common_volume_test'] / len(test_df)

    display(output_df.style.background_gradient(cmap='Blues'))

    
def display_venn(train_df, test_df, ncols=4):
    print("===============================================================================================================")
    print('# venn')
    print("===============================================================================================================")
    columns = test_df.columns
    nfigs = len(columns)
    nrows = nfigs // ncols + 1

    fig, axes = plt.subplots(figsize=(ncols * 3, nrows * 3), ncols=ncols, nrows=nrows)

    for c, ax in zip(columns, axes.ravel()):
        venn2(
            subsets=(set(train_df[c].unique()), set(test_df[c].unique())),
            set_labels=('Train', 'Test'),
            ax=ax
        )
        ax.set_title(c)

    fig.tight_layout()
    display(fig)
    plt.close()
    
    
def display_gam(input_df, target_col, ncols=5):
    print("===============================================================================================================")
    print('# GAM')
    print("===============================================================================================================")
    target_col = [target_col] if isinstance(target_col, str) else target_col
    key_cols = [c for c in list(input_df.select_dtypes('number')) if c not in target_col]
    _df = input_df[key_cols]
    _df = _df.fillna(_df.median())
    y = input_df[target_col]

    nfigs = len(_df.columns)
    nrows = nfigs // ncols + 1 if nfigs % ncols != 0 else nfigs // ncols
    
    model = GAM()
    model.fit(_df, y)

    fig, axes = plt.subplots(figsize=(ncols * 3, nrows * 2), ncols=ncols, nrows=nrows)
    axes = np.array(axes).flatten()
    for i, (ax, title, p_value) in enumerate(zip(axes, _df.columns, model.statistics_['p_values'])):
        XX = model.generate_X_grid(term=i)
        ax.plot(XX[:, i], model.partial_dependence(term=i, X=XX))
        ax.plot(XX[:, i], model.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
        ax.axhline(0, c='#cccccc')
        ax.set_title("{0:} (p={1:.2})".format(title, p_value))
        ax.set_yticks([])
        ax.grid()

    fig.tight_layout()
    display(fig)
    plt.close()
    
    
def display_all(input_df, train_df=None, test_df=None, target_col=None):
    """
    入力に応じてEDA結果を表示するための関数を適用する関数
        
    Args:
        input_df:
            必須。whole_df, train_df, test_dfどれでも入り得る。
            これに入力があると以下が表示される。
                stats_df  : nuniqueやnullの数などをまとめたDataFrame
                null      : nullの場所を視覚的に表示
                raw       : rawデータの並び順での各列の値のheatmap
                hist      : 数値変数のhistgram
                raincloud : 数値変数を正規化したraincloud。
                            もとのスケールで表示したい場合は別途`display_raincloud`をimportして引数を変更する。
                corr_df   : 数値変数の相関
        train_df, test_df:
            任意。どちらか片方を入力しても意味はない。両方入力されると以下が表示される。
                common_df : train_dfとtest_dfでどれだけデータが似通っているかのDaraFrame
                venn      : 意味としては↑と同じでベン図で表現したもの
        target_col:
            任意。input_dfとともに入力されると以下が表示される。
                gam       : input_dfの各数値変数とtarget_colのGAM
    
    """
    display_stats_df(input_df)
    display_null(input_df)
    display_raw(input_df)
    display_hist(input_df)
    display_raincloud(input_df)
    print('CAUTION: Standardized data is used for raincolud.')
    display_corr_df(input_df)
    
    if (train_df is not None)&(train_df is not None):
        display_common_df(train_df, test_df)
        display_venn(train_df, test_df)
        
    if target_col is not None:
        display_gam(input_df, target_col)


def reminder():
    print('■ how to `sweerviz`:\n\
import sweetviz as sv \n\
my_report = sv.compare([train, "Train"], [test, "Test"], TARGET_COL) \n\
my_report.show_html("OUTPUT_DIR/Report.html")')
    
    print('')
    
    print('■ how to `pandasgui`:\n\
from pandasgui import show \n\
show(df)')