# file: factors.py
# -*- coding: utf-8 -*-
"""
从原始数据构建多种因子：

- 动量因子：12 个月动量（排除最近 1 个月）
- 价值因子：由 value_raw_df 直接取（越大越便宜）
- 规模因子：Size（这里定义为 -ln(MarketCap)，小市值更优）
- 波动率因子：过去 6 个月日度收益波动率（低波动更优，后面会取负）

输出为每个因子一个 DataFrame：
  index = 调仓日（每月最后一个交易日）
  columns = 资产

注意：这里只是示例，可以按需要扩展更多因子。
"""

import numpy as np
import pandas as pd


def _get_rebalance_dates(prices: pd.DataFrame, freq: str = "ME") -> pd.DatetimeIndex:
    """
    给定价格序列，返回调仓日（例如每月最后一个交易日）。
    """
    # 以月末重采样，然后映射回最近一个真实交易日
    month_ends = prices.resample(freq).last().index
    # month_ends 已经是对齐 trade days 的（因为 last）
    return month_ends


def build_factors(
    prices: pd.DataFrame,
    mcap: pd.DataFrame,
    value_raw: pd.DataFrame,
    rebalance_freq: str = "M",
    lookback_mom: int = 252,
    lookback_vol: int = 126
):
    """
    构建多因子：

    参数：
    - prices: (T, N) 收盘价
    - mcap:   (T, N) 流通市值
    - value_raw: (T, N) 价值类原始指标
    - rebalance_freq: 调仓频率（'M' 表示每月末）
    - lookback_mom: 动量计算所需的回看期（天数）
    - lookback_vol: 波动率计算所需的回看期（天数）

    返回：
    - factors: dict[str, DataFrame]
        key 为因子名（'value', 'momentum', 'size', 'low_vol'）
        每个 DataFrame index 为调仓日，columns 为资产名
    """
    rebal_dates = _get_rebalance_dates(prices, freq=rebalance_freq)

    # 计算日度收益
    returns = prices.pct_change()

    factor_value = pd.DataFrame(index=rebal_dates, columns=prices.columns, dtype=float)
    factor_mom = pd.DataFrame(index=rebal_dates, columns=prices.columns, dtype=float)
    factor_size = pd.DataFrame(index=rebal_dates, columns=prices.columns, dtype=float)
    factor_lowvol = pd.DataFrame(index=rebal_dates, columns=prices.columns, dtype=float)

    for dt in rebal_dates:
        if dt not in prices.index:
            continue

        # 找到当前日期的索引位置
        idx = prices.index.get_loc(dt)

        # --- 动量因子：过去 lookback_mom 天收益（排除最近 21 天）
        start_idx_mom = idx - lookback_mom
        end_idx_mom = idx - 21  # 排除最近一个月（21 个交易日）

        if start_idx_mom >= 0 and end_idx_mom > start_idx_mom:
            price_start = prices.iloc[start_idx_mom, :]
            price_end = prices.iloc[end_idx_mom, :]
            mom = price_end / price_start - 1.0
            factor_mom.loc[dt, :] = mom.values

        # --- 价值因子：直接取 value_raw 当期值
        factor_value.loc[dt, :] = value_raw.loc[dt, :].values

        # --- 规模因子：Size = -ln(市值)，小市值更优
        size_raw = mcap.loc[dt, :]
        factor_size.loc[dt, :] = -np.log(size_raw.values + 1e-8)

        # --- 波动率因子：过去 lookback_vol 天日度收益标准差，低波动更优 → 取负
        start_idx_vol = idx - lookback_vol
        if start_idx_vol >= 0 and idx > start_idx_vol:
            window_ret = returns.iloc[start_idx_vol:idx, :]
            vol = window_ret.std(axis=0)
            # 低波动因子：取负，使得“越大越好”
            factor_lowvol.loc[dt, :] = -vol.values

    factors = {
        "value": factor_value.astype(float),
        "momentum": factor_mom.astype(float),
        "size": factor_size.astype(float),
        "low_vol": factor_lowvol.astype(float),
    }
    return factors


if __name__ == "__main__":
    from data import generate_fake_market_data

    prices_df, mcap_df, value_raw_df = generate_fake_market_data()
    factors = build_factors(prices_df, mcap_df, value_raw_df)
    for name, df in factors.items():
        print(name, df.shape)
