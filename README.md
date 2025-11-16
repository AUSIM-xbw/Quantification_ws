# Multi-Factor Strategy Demo (Fake Data)

本工程用完全模拟的数据，演示一个完整的多因子策略流水线：

原始数据 → 构建因子 → 合成信号 → 组合构建 → 回测 → 策略表现

模块说明：

- data.py      : 生成假数据（价格、流通市值、“价值原始指标”）
- factors.py   : 用原始数据构建多种因子（动量、价值、规模、波动率）
- signals.py   : 对因子做截面标准化，并按权重合成综合打分
- portfolio.py : 根据综合因子得分，在调仓日生成组合权重（长仓策略）
- backtest.py  : 结合价格和权重，生成组合日度收益和净值曲线
- performance.py: 计算组合的收益、波动率、夏普率、最大回撤等指标
- main.py      : 串起全流程，一键运行

运行：

```bash
pip install -r requirements.txt
python main.py
