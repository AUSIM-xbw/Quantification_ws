## 四、`ff5_data.py` —— 生成假数据

# python
# file: ff5_data.py
# -*- coding: utf-8 -*-
"""
生成用于 Fama-French 五因子模型的假数据。

包含：
- 因子收益时间序列(5 个因子)
- 无风险利率时间序列
- 多只资产的收益时间序列
- 每个资产的“真实” beta 和 alpha(用来对比回归结果)

因子:”MKT_RF, SMB, HML, RMW, CMA
"""

import numpy as np
import pandas as pd


def generate_fake_ff5_data(num_assets: int = 20,
                           num_periods: int = 120,
                           seed: int = 42):
    """
    生成假数据：
    - num_assets: 资产数量
    - num_periods: 时间长度（期数），这里用月度 120 期 ≈ 10 年
    - seed: 随机种子，保证可复现

    返回：
    - returns_df: DataFrame, 形状 (T, N)，资产总收益（含无风险利率）
    - rf_ser:     Series,   形状 (T,), 无风险利率
    - factors_df: DataFrame, 形状 (T, 5)，五因子收益（超额收益）
    - betas_true: np.ndarray, 形状 (N, 5)，真实的 beta
    - alphas_true: np.ndarray, 形状 (N,), 真实的 alpha
    """
    rng = np.random.default_rng(seed)

    # ===== 1. 模拟五个因子收益 F_t =====
    # 因子期望收益（可以理解成 risk premium 的均值）
    # 单位：每期（比如每月）
    mu_f = np.array([0.005, 0.002, 0.002, 0.0015, 0.001])
    # 因子波动率（简单起见，假设因子之间不相关，用对角协方差矩阵）
    sigma_f = np.array([0.04, 0.02, 0.02, 0.015, 0.015])
    cov_f = np.diag(sigma_f)

    # factors: 形状 (T, 5)
    factors = rng.multivariate_normal(mu_f, cov_f, size=num_periods)

    # ===== 2. 模拟无风险利率 RF_t =====
    # 简单地假设每期都是常数 0.001
    rf = np.full(num_periods, 0.001)

    # ===== 3. 为每个资产生成真实 beta 和 alpha =====
    # beta 均值：对市场因子敏感度 ~1，其它因子较小
    beta_mean = np.array([1.0, 0.3, 0.3, 0.2, 0.2])
    beta_std = np.array([0.2, 0.3, 0.3, 0.25, 0.25])

    # betas_true: (N, 5)
    betas_true = rng.normal(beta_mean, beta_std, size=(num_assets, 5))

    # 真实 alpha，设为接近 0 的小数
    alphas_true = rng.normal(0.0, 0.001, size=num_assets)

    # 特质风险（idiosyncratic risk），简单用常数标准差
    epsilon_std = 0.05
    eps = rng.normal(0.0, epsilon_std, size=(num_periods, num_assets))

    # ===== 4. 生成资产收益 =====
    # 模型：R_{i,t} = RF_t + alpha_i + beta_i^T F_t + epsilon_{i,t}
    R = np.empty((num_periods, num_assets))
    for t in range(num_periods):
        # factors[t] 形状 (5,)
        # betas_true.T 形状 (5, N)
        # dot -> 形状 (N,)
        R[t, :] = rf[t] + alphas_true + factors[t].dot(betas_true.T) + eps[t, :]

    # ===== 5. 包装成 pandas 结构 =====
    dates = pd.date_range("2000-01-31", periods=num_periods, freq="M")
    asset_cols = [f"Asset_{i+1:02d}" for i in range(num_assets)]

    returns_df = pd.DataFrame(R, index=dates, columns=asset_cols)
    rf_ser = pd.Series(rf, index=dates, name="RF")

    factor_cols = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
    factors_df = pd.DataFrame(factors, index=dates, columns=factor_cols)

    return returns_df, rf_ser, factors_df, betas_true, alphas_true


if __name__ == "__main__":
    # 简单自测
    r, rf, f, b, a = generate_fake_ff5_data()
    print("returns shape:", r.shape)
    print("factors shape:", f.shape)
    print("true betas shape:", b.shape)
