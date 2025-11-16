# file: backtest.py
# -*- coding: utf-8 -*-
"""
简单的多因子组合回测：

- 已知：
    - 每日价格序列 prices_df
    - 调仓日组合权重 weights_rebal（在调仓日生效，之后持有到下一个调仓日）

- 做法：
    - 计算日度资产收益矩阵 returns_df
    - 将调仓日权重扩展为每日权重（在调仓日之后持仓不变，直到下一个调仓日）
    - 组合日收益 = 每日 (weights_daily * returns_df).sum(axis=1)
    - 生成组合净值曲线

输出：
- port_ret: 组合日度收益 Series
- port_nav: 组合净值曲线 Series（起始为 1.0）
"""

import pandas as pd


def run_backtest(
    prices: pd.DataFrame,
    weights_rebal: pd.DataFrame
):
    """
    根据价格和调仓日权重进行回测。

    参数：
    - prices: (T, N) 收盘价 DataFrame
    - weights_rebal: (Tr, N) 调仓日目标权重（index 是调仓日的日期）

    返回：
    - port_ret: Series, 组合日度收益
    - port_nav: Series, 组合净值曲线（从 1 开始）
    """
    # 计算日度收益
    returns = prices.pct_change().fillna(0.0)

    # 将 rebalance 权重对齐到全时间轴，并向前填充
    weights_daily = weights_rebal.reindex(returns.index)
    # 在第一个非 NA 日期之前，权重为 0
    first_rebal_date = weights_rebal.index.min()
    weights_daily.loc[:first_rebal_date, :] = 0.0
    # 向前填充
    weights_daily = weights_daily.ffill().fillna(0.0)

    # 对齐 columns（万一有差别）
    weights_daily = weights_daily.reindex(columns=returns.columns, fill_value=0.0)

    # 每日组合收益
    port_ret = (weights_daily * returns).sum(axis=1)

    # 组合净值曲线（初始 1.0）
    port_nav = (1.0 + port_ret).cumprod()

    return port_ret, port_nav


if __name__ == "__main__":
    from data import generate_fake_market_data
    from factors import build_factors
    from signals import combine_factors_to_scores
    from portfolio import build_long_only_portfolio

    prices_df, mcap_df, value_raw_df = generate_fake_market_data()
    factors = build_factors(prices_df, mcap_df, value_raw_df)
    scores = combine_factors_to_scores(factors)
    weights_rebal = build_long_only_portfolio(scores)
    port_ret, port_nav = run_backtest(prices_df, weights_rebal)
    print(port_ret.head())
    print(port_nav.tail())
