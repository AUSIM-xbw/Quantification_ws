# file: ff5_regression.py
# -*- coding: utf-8 -*-
"""
Fama-French 五因子模型的估计模块。

包含:
- 时间序列 OLS 回归(对每个资产估计 alpha_i, beta_i)
- 对所有资产跑 OLS,得到 beta_hat 矩阵
- Fama-MacBeth 横截面回归，估计因子风险溢价 lambda_hat
"""

import numpy as np
import pandas as pd


def ols_time_series(y: np.ndarray, X: np.ndarray):
    """
    对单一资产做时间序列 OLS 回归:
        y_t = alpha + beta^T X_t + eps_t

    参数:
    - y: 形状 (T,) 的向量(例如资产超额收益)
    - X: 形状 (T, K) 的矩阵(K 个因子)

    返回:
    - coef: 形状 (K+1,) 的向量，[alpha, beta_1, ..., beta_K]
    - sigma2: 残差方差估计
    - resid: 残差向量
    """
    T = y.shape[0]
    # 设计矩阵:添加截距列
    X_design = np.column_stack([np.ones(T), X])  # (T, K+1)

    # OLS 闭式解: (X'X)^(-1) X'y
    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.inv(XtX)
    Xty = X_design.T @ y
    coef = XtX_inv @ Xty  # (K+1,)

    # 残差及残差方差
    y_hat = X_design @ coef
    resid = y - y_hat
    dof = T - X_design.shape[1]  # 自由度 = 样本数 - 参数个数
    sigma2 = (resid @ resid) / dof

    return coef, sigma2, resid


def estimate_betas_all_assets(returns_df: pd.DataFrame,
                              rf_ser: pd.Series,
                              factors_df: pd.DataFrame):
    """
    对每个资产做时间序列 OLS 回归，估计:
    - alpha_hat_i
    - beta_hat_i (对 5 个因子的暴露)

    返回:
    - betas_hat: 形状 (N, K) 的矩阵
    - alphas_hat: 形状 (N,) 的向量
    - sigma2s: 形状 (N,) 的残差方差估计
    """
    assets = returns_df.columns
    T = len(returns_df)
    K = factors_df.shape[1]

    betas_hat = np.zeros((len(assets), K))
    alphas_hat = np.zeros(len(assets))
    sigma2s = np.zeros(len(assets))

    X = factors_df.values  # (T, K)

    for i, asset in enumerate(assets):
        # 资产超额收益:R_i,t - R_f,t
        y = (returns_df[asset] - rf_ser).values  # (T,)
        coef, sigma2, resid = ols_time_series(y, X)
        alphas_hat[i] = coef[0]
        betas_hat[i, :] = coef[1:]
        sigma2s[i] = sigma2

    return betas_hat, alphas_hat, sigma2s


def fama_macbeth_lambda(returns_df: pd.DataFrame,
                        rf_ser: pd.Series,
                        betas_hat: np.ndarray):
    """
    使用 Fama-MacBeth 的思路估计因子风险溢价 lambda:

    对每个时间 t(横截面)做回归:
        R_{i,t} - R_{f,t} = gamma0_t + gamma_t^T * beta_hat_i + zeta_{i,t}

    然后对 gamma_t 在时间上取平均:
        lambda_hat = mean_t(gamma_t)

    参数:
    - returns_df: (T, N) 的资产收益
    - rf_ser: (T,) 的无风险利率
    - betas_hat: (N, K)，每个资产的 beta_hat

    返回:
    - lambdas_mean: (K+1,) 向量，[lambda0, lambda_1, ..., lambda_K]
    - lambdas_t: (T, K+1) 矩阵，每期的横截面回归系数
    """
    T = len(returns_df)
    assets = returns_df.columns
    N = len(assets)
    K = betas_hat.shape[1]

    lambdas_t = np.zeros((T, K + 1))  # 每期:截距 + K 个因子溢价

    # beta_hat 在时间上假定常数，直接用于每期横截面回归
    X_cs = np.column_stack([np.ones(N), betas_hat])  # (N, K+1)

    # 对每个时间点做横截面回归
    for t in range(T):
        # 当前期所有资产的超额收益 (N,)
        y_t = (returns_df.iloc[t, :].values - rf_ser.iloc[t])
        # OLS: (X'X)^(-1) X'y
        XtX = X_cs.T @ X_cs
        XtX_inv = np.linalg.inv(XtX)
        Xty = X_cs.T @ y_t
        coef_t = XtX_inv @ Xty  # (K+1,)
        lambdas_t[t, :] = coef_t

    lambdas_mean = lambdas_t.mean(axis=0)
    return lambdas_mean, lambdas_t


if __name__ == "__main__":
    # 这里不做自测，因为需要数据，主逻辑放在 main.py
    pass
