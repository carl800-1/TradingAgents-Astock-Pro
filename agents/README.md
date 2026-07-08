# Agent 集成指南

## 新增 Agent 清单

| Agent | 文件 | 字段名 | 职责 |
|-------|------|--------|------|
| 📈 **财报季分析师** | `earnings_analyst.py` | `earnings_report` | 追踪业绩预告、实际财报与预期差距 |
| 🏭 **行业轮动分析师** | `sector_rotation_analyst.py` | `sector_rotation_report` | 追踪板块轮动、热点方向、行业景气度 |
| 🤖 **量化信号分析师** | `quant_signal_analyst.py` | `quant_signal_report` | 调用 LightGBM 模型输出量化预测信号 |

---

## 集成到 TradingAgents-astock

### 步骤 1：复制文件

```bash
# 目标项目
cd ~/TradingAgents-astock

# 复制 3 个 Agent
cp /Users/frank/TradingAgents-Pro/agents/earnings_analyst.py tradingagents/agents/analysts/
cp /Users/frank/TradingAgents-Pro/agents/sector_rotation_analyst.py tradingagents/agents/analysts/
cp /Users/frank/TradingAgents-Pro/agents/quant_signal_analyst.py tradingagents/agents/analysts/
```

### 步骤 2：更新 agent_states.py

在 `tradingagents/agents/utils/agent_states.py` 的 `AgentState` 类末尾新增：

```python
class AgentState(MessagesState):
    # ... 现有字段 ...
    
    # === 新增 A 股特化报告 ===
    earnings_report: Annotated[str, "Report from the Earnings Analyst"]
    sector_rotation_report: Annotated[str, "Report from the Sector Rotation Analyst"]
    quant_signal_report: Annotated[str, "Report from the Quant Signal Analyst"]
```

### 步骤 3：更新 agents/__init__.py

在 `tradingagents/agents/__init__.py` 中导入并导出：

```python
from .analysts.earnings_analyst import create_earnings_analyst
from .analysts.sector_rotation_analyst import create_sector_rotation_analyst
from .analysts.quant_signal_analyst import create_quant_signal_analyst

__all__ = [
    # ... 现有导出 ...
    "create_earnings_analyst",
    "create_sector_rotation_analyst",
    "create_quant_signal_analyst",
]
```

### 步骤 4：更新 graph/setup.py

在 `tradingagents/graph/setup.py` 中：

1. 导入新 Agent 工厂函数
2. 在 `create_default_graph` 中注册新节点
3. 在 `create_conditional_logic` 中新增路由

### 步骤 5：更新 graph/trading_graph.py

在分析师列表中注册：

```python
analysts = [
    create_market_analyst(llm),
    create_social_media_analyst(llm),
    create_news_analyst(llm),
    create_fundamentals_analyst(llm),
    create_policy_analyst(llm),
    create_hot_money_tracker(llm),
    create_lockup_watcher(llm),
    # 新增 3 个 Agent
    create_earnings_analyst(llm),
    create_sector_rotation_analyst(llm),
    create_quant_signal_analyst(llm),
]
```

---

## 量化模型配置

`quant_signal_analyst.py` 依赖 LightGBM 模型文件：

```bash
# 训练模型
python train_lgbm_v2.py

# 设置模型路径
export QUANT_MODEL_PATH=/path/to/model_lgbm_v2.pkl
```

模型通过环境变量 `QUANT_MODEL_PATH` 加载，默认读取 `~/.vnpy_project/data/` 下的模型文件。

---

## Agent 输出格式

每个新增 Agent 遵循 TradingAgents 标准模式：

```python
def create_<name>_analyst(llm):
    def <name>_analyst_node(state):
        tools = [...]  # 数据工具
        system_message = "A 股分析框架 + 必采清单 + 中文指令"
        prompt = ChatPromptTemplate.from_messages([...])
        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])
        return {"messages": [result], "<name>_report": report}
    return <name>_analyst_node
```