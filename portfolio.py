# file: portfolio.py
# -*- coding: utf-8 -*-
"""
根据综合打分，在调仓日构建组合权重。

这里实现一个简单的长仓策略：
- 每个调仓日，从有打分的股票里选出前 top_quantile 比例的股票
- 对这些股票等权分配权重
- 其它股票权重为 0

输出：
- weights_rebal: DataFrame, index 为调仓日，columns 为资产，值为目标权重
"""

import pandas as pd


def build_long_only_portfolio(
    scores: pd.DataFrame,
    top_quantile: float = 0.2
) -> pd.DataFrame:
    """
    根据综合打分构建长仓组合权重（等权）。

    参数：
    - scores: DataFrame, index 为调仓日，columns 为资产
    - top_quantile: 选取前多少分位数的股票，比如 0.2 表示选前 20%

    返回：
    - weights: DataFrame, index 为调仓日，columns 为资产
    """
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    for dt, row in scores.iterrows():
        s = row.dropna()
        if s.empty:
            continue

        # 按打分从高到低排序
        s_sorted = s.sort_values(ascending=False)
        n_top = max(int(len(s_sorted) * top_quantile), 1)
        selected = s_sorted.iloc[:n_top].index

        # 等权
        w = 1.0 / n_top
        weights.loc[dt, selected] = w

    return weights


if __name__ == "__main__":
    from data import generate_fake_market_data
    from factors import build_factors
    from signals import combine_factors_to_scores

    prices_df, mcap_df, value_raw_df = generate_fake_market_data()
    factors = build_factors(prices_df, mcap_df, value_raw_df)
    scores = combine_factors_to_scores(factors)
    weights_rebal = build_long_only_portfolio(scores)
    print(weights_rebal.head())
