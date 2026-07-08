# TradingAgents-Astock-Pro

<p align="center">
  A股多Agent投研框架 · 数据回退 + 回测验证 + AI决策
</p>

<p align="center">
  <a href="https://github.com/TauricResearch/TradingAgents"><img alt="基于" src="https://img.shields.io/badge/基于-TauricResearch%2FTradingAgents%20(65K%20⭐)-orange"/></a>
  <a href="https://github.com/simonlin1212/TradingAgents-astock"><img alt="参考" src="https://img.shields.io/badge/参考-simonlin1212%2Fastock-blue"/></a>
  <a href="https://img.shields.io/badge/License-Apache%202.0-green"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green"/></a>
  <a href="https://img.shields.io/badge/pip%20install%20tradingagents--astock--pro-lightgrey"><img alt="install" src="https://img.shields.io/badge/pip%20install%20tradingagents--astock--pro-lightgrey"/></a>
  <a href="https://img.shields.io/badge/Python-3.10%2B-yellow"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-yellow"/></a>
</p>

<p align="center">
  基于 <a href="https://github.com/TauricResearch/TradingAgents">TauricResearch/TradingAgents (65K ⭐)</a> 和 <a href="https://github.com/simonlin1212/TradingAgents-astock">simonlin1212/TradingAgents-astock</a> 的 A 股深度特化 fork<br>
  全 Apache 2.0 开源 · pip install 即跑 · 零外部服务依赖
</p>

> ⚠️ **免责声明**：本项目仅供学习研究与技术演示，不构成任何投资建议。投资决策请咨询持牌专业机构。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **多 Agent 辩论** | 7 位专业分析师 + 2 位研究员 + 3 位风控 + 1 位交易员 + 1 位组合经理 |
| 📊 **A 股特化数据层** | mootdx → AKShare 自动回退，零 API Key 依赖 |
| 💹 **实时行情** | 东方财富直连 HTTP，获取实时价格 |
| 🔄 **回测验证** | 支持 T+1 / 涨跌停 / 手续费 / 滑点 / 仓位管理 |
| 🤖 **量化信号** | 内置 LightGBM 模型引擎，输出量化预测信号 |
| 📈 **可视化** | 自动生成净值曲线和回撤图表 |

---

## 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    数据层 (DataLayer)                      │
│  mootdx (TCP 7709) ──────→ AKShare 回退                    │
│  东方财富推送 ────────────→ 实时行情                        │
│  新浪财经 / 同花顺 / 财联社 ──→ 数据回退通道                  │
├─────────────────────────────────────────────────────────┤
│                    分析师团队 (Analysts)                   │
│  🏪 Market → 💬 Social → 📰 News → 📊 Fundamentals      │
│  → 🏛️ Policy → 🔥 Hot Money → 🔓 Lockup              │
│  → 📈 Earnings → 🏭 Sector Rotation → 🤖 Quant Signal   │
├─────────────────────────────────────────────────────────┤
│                  研究员团队 (Researchers)                   │
│           Bull Researcher ↔ Bear Researcher              │
│                (多空辩论，最多 N 轮)                        │
├─────────────────────────────────────────────────────────┤
│                 研究总监 (Research Manager)                │
│              (综合双方观点，形成投资计划)                     │
├─────────────────────────────────────────────────────────┤
│                   交易员 (Trader)                          │
│              (A 股约束：T+1/涨跌停/最小手数)                │
├─────────────────────────────────────────────────────────┤
│                风控团队 (Risk Management)                  │
│      Aggressive ↔ Conservative ↔ Neutral                 │
│                   (三方风险辩论)                            │
├─────────────────────────────────────────────────────────┤
│               组合经理 (Portfolio Manager)                 │
│         (最终决策：Buy / Overweight / Hold / Sell)         │
├─────────────────────────────────────────────────────────┤
│                   回测验证 (Backtest)                      │
│              (决策 → 历史回测 → 指标评估)                   │
└─────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/carl800-1/TradingAgents-Astock-Pro.git
cd TradingAgents-Astock-Pro

# 激活 vnpy 环境（或创建新的虚拟环境）
source ~/vnpy_env/bin/activate

# 安装依赖
pip install langchain langchain-core langgraph pandas numpy joblib akshare baostock
```

### 运行回测

```bash
# 单只股票回测
python scripts/run_backtest.py --stock 600519 --name 贵州茅台

# 指定初始资金和仓位
python scripts/run_backtest.py --stock 601006 --name 大秦铁路 --capital 500000 --position 0.5

# 运行所有默认标的
python scripts/run_backtest.py --all

# 不生成图表
python scripts/run_backtest.py --stock 600519 --name 贵州茅台 --no-chart
```

### 调用数据层

```python
from data_layer.data_layer import DataLayer, get_realtime_price, check_data_quality

# 获取实时行情
price = get_realtime_price("600519")
print(f"贵州茅台当前价格: {price['price']}")

# K 线数据（自动回退）
layer = DataLayer()
data, status = layer.get_kline("600519", "20250101", "20261231")
print(f"数据来源: {status}")  # OK_mootdx 或 OK_akshare

# 数据质量检查
quality = check_data_quality(data, source=status)
print(f"质量评分: {quality['quality_score']}")
```

---

## 项目结构

```
TradingAgents-Astock-Pro/
├── agents/                          # 新增 Agent
│   ├── earnings_analyst.py          # 财报季分析师
│   ├── sector_rotation_analyst.py   # 行业轮动分析师
│   └── quant_signal_analyst.py      # 量化信号分析师
├── backtest/                        # 回测模块
│   └── vnpy_backtest.py             # vn.py 回测引擎
├── data_layer/                      # 数据层
│   └── data_layer.py                # 统一数据获取 + 实时行情
├── config/                          # 配置
│   └── backtest_config.py           # 回测参数配置
├── scripts/                         # 运行脚本
│   └── run_backtest.py              # 一键运行回测
├── data/                            # 输出图表
├── README.md
└── pyproject.toml                   # 项目依赖
```

---

## 新增 Agent 说明

### 1. 财报季分析师 (Earnings Analyst)

追踪 A 股业绩预告、实际财报与市场预期差距。

**必采清单**：
- 最近一期业绩预告数据
- 扣非净利润 vs 净利润差异
- 经营现金流量净额及趋势
- 营收同比增长率
- 财报总体评级

### 2. 行业轮动分析师 (Sector Rotation Analyst)

追踪板块轮动方向、热点题材和行业景气度。

**必采清单**：
- 目标公司所属概念板块及涨跌幅
- 主力资金在行业间的流入流出
- 行业景气度评级
- 板块轮动方向判断

### 3. 量化信号分析师 (Quant Signal Analyst)

调用 LightGBM 模型输出量化预测信号。

**输出内容**：
- 模型路径及版本
- 提取的特征及数值
- 买入概率（0-1）
- 信号等级（BUY/HOLD/SELL）
- 模型局限性提示

---

## 回测结果示例

### 贵州茅台（2025-01 ~ 2026-07）

```
标的:        贵州茅台
回测区间:    2025-01-02 ~ 2026-07-07
总收益率:    +10.21%   基准收益率: -20.11%
超额收益:    +30.31%   最大回撤: -3.65%
交易次数:    6 笔      胜率: 0%
综合评估:    🌟 及格（有收益）
```

### 大秦铁路（2025-01 ~ 2026-07）

```
标的:        大秦铁路
回测区间:    2025-01-02 ~ 2026-07-07
总收益率:    -0.51%    基准收益率: -27.42%
超额收益:    +26.92%   最大回撤: -2.38%
交易次数:    5 笔      胜率: 40%
综合评估:    ❌ 不及格（但跑赢基准 27%）
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 多 Agent 框架 | LangGraph + LangChain |
| LLM 调用 | OpenAI / Anthropic / DeepSeek |
| 数据源 | mootdx + AKShare + 东方财富直连 |
| 回测引擎 | 自建（pandas + baostock） |
| 图表生成 | matplotlib |
| 量化模型 | LightGBM |

---

## 接入 TradingAgents-astock

新增 Agent 可集成到 TradingAgents-astock 项目：

1. 复制 `agents/` 下的 3 个 .py 文件到 `tradingagents/agents/analysts/`
2. 在 `agent_states.py` 新增字段（`earnings_report`、`sector_rotation_report`、`quant_signal_report`）
3. 在 `agents/__init__.py` 导出新函数
4. 在 `graph/setup.py` 注册新节点
5. 在 `graph/trading_graph.py` 加入分析师列表

详细步骤见 [集成指南](agents/README.md)

---

## 开发路线

```
Week 1-2:  克隆 TradingAgents-astock，跑通 main.py
Week 3-4:  集成 3 个新 Agent，完成必采清单
Week 5-6:  数据层优化（AKShare 回退 + 实时行情）
Week 7-8:  接入 LightGBM 量化模型
Week 9-10: 完善回测模块，批量验证决策
Week 11+:  Web UI + 定时分析
```

---

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/xxx`)
3. 提交更改 (`git commit -am 'Add feature'`)
4. 推送到分支 (`git push origin feature/xxx`)
5. 发起 Pull Request

---

## 许可证

Apache 2.0

---

## 致谢

- [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) — 多 Agent 投研框架原始项目
- [simonlin1212/TradingAgents-astock](https://github.com/simonlin1212/TradingAgents-astock) — A 股深度特化 fork
- [vn.py](https://github.com/vnpy/vnpy) — A 股实盘交易框架
- [AKShare](https://github.com/akfamily/akshare) — 开源金融数据接口