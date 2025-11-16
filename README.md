# Fama-French 五因子模型（假数据 Demo）

本工程使用完全模拟的假数据，复现了 Fama-French 五因子模型的基本数学流程：

1. 生成五个因子收益：MKT-RF, SMB, HML, RMW, CMA（用高斯分布模拟）
2. 为每个资产生成“真实”的 β 和 α
3. 用五因子线性模型生成资产收益：
   R_i,t = RF_t + α_i + β_i^T F_t + ε_i,t
4. 对每个资产做时间序列 OLS 回归，估计 α_hat, β_hat
5. 使用 Fama-MacBeth 横截面回归估计因子风险溢价 λ_hat

运行方法：

```bash
pip install -r requirements.txt
python main.py
