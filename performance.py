# file: performance.py
# -*- coding: utf-8 -*-
"""
对组合日度收益计算一些常见绩效指标：

- 年化收益
- 年化波动率
- 夏普比（假设无风险利率为 0）
- 最大回撤
"""

import numpy as np
import pandas as pd


def compute_max_drawdown(nav: pd.Series) -> float:
    """
    计算最大回撤（基于净值曲线 nav）。
    """
    cum_max = nav.cummax()
    drawdown = nav / cum_max - 1.0
    return drawdown.min()


def compute_performance_stats(
    port_ret: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate_annual: float = 0.0
):
    """
    根据组合日度收益，计算绩效指标。

    返回：
    - stats: dict
    """
    # 去掉开头可能的 NaN
    r = port_ret.dropna()
    if len(r) == 0:
        raise ValueError("组合收益为空，无法计算绩效")

    mean_daily = r.mean()
    std_daily = r.std()

    ann_ret = (1.0 + mean_daily) ** periods_per_year - 1.0
    ann_vol = std_daily * np.sqrt(periods_per_year)
    rf_daily = (1.0 + risk_free_rate_annual) ** (1.0 / periods_per_year) - 1.0
    excess_mean_daily = mean_daily - rf_daily
    excess_ann_ret = (1.0 + excess_mean_daily) ** periods_per_year - 1.0

    sharpe = excess_ann_ret / ann_vol if ann_vol > 0 else np.nan

    # 净值曲线
    nav = (1.0 + r).cumprod()
    max_dd = compute_max_drawdown(nav)

    stats = {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }
    return stats


if __name__ == "__main__":
    import pandas as pd

    # 简单自测
    rng = np.random.default_rng(0)
    fake_ret = pd.Series(rng.normal(0.0005, 0.01, size=252))
    stats = compute_performance_stats(fake_ret)
    print(stats)
