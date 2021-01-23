from scipy import stats
import pandas as pd
import numpy as np

# 主にGroupbyのaggで使う簡単な関数たち

def range_diff(x):
    return x.max() - x.min()


def range_ratio(x):
    try:
        output = x.max() / x.min()
    except ZeroDivisionError:
        output = np.nan
    return output


def third_quartile(x):
    return x.quantile(0.75)


def first_quartile(x):
    return x.quantile(0.25)


def quartile_range(x):
    return x.quantile(0.75) - x.quantile(0.25)


def quartile_ratio(x):
    try:
        output = x.quantile(0.75) / x.quantile(0.25)
    except ZeroDivisionError:
        output = np.nan
    return output


def kurtosis(x):
    return x.kurt()


def over1std_ratio(x):
    return (x > (x.mean() + x.std())).sum() / len(x)


def under1std_ratio(x):
    return (x < (x.mean() - x.std())).sum() / len(x)


def coef_var(x):
    try:
        output = x.std() / x.mean()
    except ZeroDivisionError:
        output = np.nan
    return output


def hl_ratio(x):
    return (x > x.mean()).sum() / (x <= x.mean()).sum()


def shapiro(x):
    w, p  = stats.shapiro(x)
    return p
    

def most_freq_count(x):
    return x.value_counts(dropna=False).sort_values(ascending=False).values[0]


def most_freq_per(x):
    return x.value_counts(normalize=True, dropna=False).sort_values(ascending=False).values[0]