# file: signals.py
# -*- coding: utf-8 -*-
"""
对因子做截面标准化（z-score），再按给定权重合成多因子综合信号（打分）。

输出：
- scores_df: index 为调仓日，columns 为资产，数值为综合打分（越大越好）。
"""

import numpy as np
import pandas as pd


def zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个调仓日做截面 z-score 标准化：
    z_ij = (x_ij - mean_j) / std_j

    df: index 为日期，columns 为资产
    """
    def _zscore(s):
        m = s.mean()
        std = s.std()
        if std == 0 or np.isnan(std):
            return s * 0.0
        return (s - m) / std

    return df.apply(_zscore, axis=1)


def combine_factors_to_scores(
    factors: dict,
    factor_weights: dict | None = None
) -> pd.DataFrame:
    """
    输入：
    - factors: dict[str, DataFrame]，每个因子一个 DataFrame
    - factor_weights: dict[str, float]，因子权重，默认平均权重

    输出：
    - scores_df: DataFrame, index 为调仓日，columns 为资产
    """
    factor_names = list(factors.keys())
    index = list(factors.values())[0].index
    columns = list(factors.values())[0].columns

    # 默认每个因子均匀权重
    if factor_weights is None:
        w = {name: 1.0 / len(factor_names) for name in factor_names}
    else:
        # 归一化权重
        total = sum(factor_weights.values())
        w = {k: v / total for k, v in factor_weights.items()}

    # 对每个因子做截面标准化
    factors_z = {}
    for name, df in factors.items():
        z = zscore_cross_section(df)
        # 关键一行：把 NaN 当 0 处理，避免污染别的因子
        z = z.fillna(0.0)
        factors_z[name] = z

    # 合成综合打分
    scores_df = pd.DataFrame(0.0, index=index, columns=columns, dtype=float)
    for name in factor_names:
        scores_df += w[name] * factors_z[name]

    return scores_df


if __name__ == "__main__":
    from data import generate_fake_market_data
    from factors import build_factors

    prices_df, mcap_df, value_raw_df = generate_fake_market_data()
    factors = build_factors(prices_df, mcap_df, value_raw_df)
    scores = combine_factors_to_scores(factors)
    print(scores.head())
