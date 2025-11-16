# file: main.py
# -*- coding: utf-8 -*-
"""
主入口：一键跑通完整流程

原始数据 → 构建因子 → 合成信号 → 组合构建 → 回测 → 策略表现python main.py
"""

from data import generate_fake_market_data
from factors import build_factors
from signals import combine_factors_to_scores
from portfolio import build_long_only_portfolio
from backtest import run_backtest
from performance import compute_performance_stats


def main():
    # 1. 生成原始数据（价格、流通市值、“价值原始指标”）
    prices_df, mcap_df, value_raw_df = generate_fake_market_data(
        num_assets=30,
        num_days=252 * 3,
        seed=123
    )
    print("原始数据生成完成：")
    print("价格矩阵形状:", prices_df.shape)

    # 2. 构建因子
    factors = build_factors(prices_df, mcap_df, value_raw_df)
    print("\n构建因子完成：")
    for name, df in factors.items():
        print(f"  因子 {name:8s} 形状: {df.shape}")

    # 3. 合成多因子综合信号（打分）
    # 可以自定义因子权重，比如：价值 0.4，动量 0.3，规模 0.2，低波动 0.1
    factor_weights = {
        "value": 0.4,
        "momentum": 0.3,
        "size": 0.2,
        "low_vol": 0.1,
    }
    scores = combine_factors_to_scores(factors, factor_weights=factor_weights)
    print("\n合成综合信号完成，示例前几行：")
    print(scores.head())

    # 4. 根据信号构建组合（长仓，选前 20%）
    weights_rebal = build_long_only_portfolio(scores, top_quantile=0.2)
    print("\n组合权重（调仓日）示例：")
    print(weights_rebal.head())

    # 5. 回测：根据价格 + 调仓权重生成组合日度收益和净值曲线
    port_ret, port_nav = run_backtest(prices_df, weights_rebal)
    print("\n回测完成。组合净值末值:", port_nav.iloc[-1])

    # 6. 绩效分析
    stats = compute_performance_stats(port_ret)
    print("\n策略表现：")
    print(f"  年化收益率: {stats['annual_return']:.2%}")
    print(f"  年化波动率: {stats['annual_vol']:.2%}")
    print(f"  夏普比率  : {stats['sharpe']:.2f}")
    print(f"  最大回撤  : {stats['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
