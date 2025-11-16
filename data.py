## 3. `data.py` —— 原始数据生成

# ```python
# file: data.py
# -*- coding: utf-8 -*-
"""
生成用于多因子策略的“原始数据”：
- 股票价格时间序列（收盘价）
- 流通市值时间序列（由价格 * 固定股本模拟）
- 简单模拟一个“价值相关原始指标”（例如 book_to_price）

注意：全部是假数据，只用于演示流程。
"""

import numpy as np
import pandas as pd


def generate_fake_market_data(
    num_assets: int = 30,
    num_days: int = 252 * 3,  # 大约 3 年交易日
    seed: int = 42
):
    """
    生成假数据：
    - prices_df: (T, N) 每日收盘价
    - mcap_df:   (T, N) 每日流通市值
    - value_raw_df: (T, N) “价值原始指标”，可理解成 book_to_price 之类的东西

    返回：
    prices_df, mcap_df, value_raw_df
    """
    rng = np.random.default_rng(seed)

    # 生成交易日（工作日）
    dates = pd.date_range("2018-01-01", periods=num_days, freq="B")

    # 为每个资产生成一个随机的年化漂移和波动率
    # 日度漂移大约 0.0002，波动率 0.01~0.03
    daily_drift = rng.normal(0.0002, 0.0001, size=num_assets)
    daily_vol = rng.uniform(0.01, 0.03, size=num_assets)

    prices = np.empty((num_days, num_assets))
    mkt_caps = np.empty((num_days, num_assets))
    value_raw = np.empty((num_days, num_assets))

    # 初始价格和股本
    init_prices = rng.uniform(20, 200, size=num_assets)
    shares_outstanding = rng.uniform(5e7, 2e8, size=num_assets)  # 固定股本

    # 模拟价格路径 & 市值 & 价值原始指标
    for j in range(num_assets):
        # 随机游走的日度 log return
        shocks = daily_drift[j] + daily_vol[j] * rng.normal(0, 1, size=num_days)
        log_price = np.log(init_prices[j]) + np.cumsum(shocks)
        price_path = np.exp(log_price)

        prices[:, j] = price_path
        mkt_caps[:, j] = price_path * shares_outstanding[j]

        # 模拟一个价值相关原始数据（book_to_price 之类）
        # 这里设为：某个“基本面值” + 噪声，再除以价格
        base_fundamental = rng.uniform(5, 15)  # 某种“账面价值”
        noise = rng.normal(0, 0.5, size=num_days)
        fundamental_series = base_fundamental + noise
        value_raw[:, j] = fundamental_series / price_path  # 类似 B/P

    asset_names = [f"Asset_{i+1:02d}" for i in range(num_assets)]

    prices_df = pd.DataFrame(prices, index=dates, columns=asset_names)
    mcap_df = pd.DataFrame(mkt_caps, index=dates, columns=asset_names)
    value_raw_df = pd.DataFrame(value_raw, index=dates, columns=asset_names)

    return prices_df, mcap_df, value_raw_df


if __name__ == "__main__":
    p, mc, vr = generate_fake_market_data()
    print("prices_df shape:", p.shape)
    print("mcap_df shape:", mc.shape)
    print("value_raw_df shape:", vr.shape)
