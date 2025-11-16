# file: main.py
# -*- coding: utf-8 -*-
"""
主入口:使用假数据复现 Fama-French 五因子模型的完整流程:

1. 生成假因子收益 + 资产收益 + 无风险利率(ff5_data.generate_fake_ff5_data)
2. 对每个资产做时间序列回归，估计 alpha_hat, beta_hat
3. 使用 Fama-MacBeth 横截面回归估计因子风险溢价 lambda_hat
4. 打印部分结果做 sanity check
"""

import numpy as np

from ff5_data import generate_fake_ff5_data
from ff5_regression import estimate_betas_all_assets, fama_macbeth_lambda


def main():
    # === 1. 生成假数据 ===
    num_assets = 20
    num_periods = 120

    returns_df, rf_ser, factors_df, betas_true, alphas_true = generate_fake_ff5_data(
        num_assets=num_assets,
        num_periods=num_periods,
        seed=42
    )

    print("====== 假数据基本信息 ======")
    print("时间长度(期数):", num_periods)
    print("资产数量:", num_assets)
    print("因子列名:", list(factors_df.columns))
    print()

    # === 2. 时间序列回归:估计每个资产的 beta_hat, alpha_hat ===
    betas_hat, alphas_hat, sigma2s = estimate_betas_all_assets(
        returns_df, rf_ser, factors_df
    )

    # 打印前 3 个资产的真实 beta 和估计的 beta 做对比
    print("====== 部分资产的真实 beta vs 回归估计 beta_hat(前 3 个资产)======")
    factor_names = list(factors_df.columns)
    for i in range(min(3, num_assets)):
        print(f"\n资产 Asset_{i+1:02d}:")
        print("  真实 alpha:    {:.6f}".format(alphas_true[i]))
        print("  估计 alpha_hat:{:.6f}".format(alphas_hat[i]))
        print("  因子      | 真实 beta   | 估计 beta_hat")
        for k, name in enumerate(factor_names):
            print("  {:7s} | {:10.4f} | {:10.4f}".format(
                name, betas_true[i, k], betas_hat[i, k]
            ))

    # === 3. Fama-MacBeth 横截面回归:估计因子风险溢价 lambda_hat ===
    lambdas_mean, lambdas_t = fama_macbeth_lambda(
        returns_df, rf_ser, betas_hat
    )

    print("\n====== Fama-MacBeth 估计的因子风险溢价 λ(时间平均)======")
    # lambdas_mean[0] 是截距，其余对应五个因子
    print("lambda0 (截距): {:.6f}".format(lambdas_mean[0]))
    for k, name in enumerate(factor_names):
        print("lambda_{:7s}: {:.6f}".format(name, lambdas_mean[k + 1]))

    print("\n注意:")
    print("- 所有数据都是模拟生成的，因子溢价 λ_hat 数值只用于演示流程。")
    print("- 工程结构已经包含:数据生成 -> 时间序列回归 -> 横截面回归。")
    print("- 你可以把 generate_fake_ff5_data 换成真实数据读取函数，就可以做真实回归。")


if __name__ == "__main__":
    main()
