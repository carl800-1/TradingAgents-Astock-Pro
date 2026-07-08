"""
方向 1：新增 3 个 Analyst Agent — 财报季分析师

接入方式：
  1. 复制到 tradingagents/agents/analysts/earnings_analyst.py
  2. 在 agent_states.py 新增 earnings_report 字段
  3. 在 agents/__init__.py 导出 create_earnings_analyst
  4. 在 graph/setup.py 注册节点
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
    get_profit_forecast,
    get_fundamentals,
    get_income_statement,
    get_cashflow,
)
from tradingagents.dataflows.config import get_config


def create_earnings_analyst(llm):
    """A-stock earnings analyst: tracks earnings forecasts, surprises, and post-earnings reactions."""

    def earnings_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_profit_forecast,
            get_fundamentals,
            get_income_statement,
            get_cashflow,
        ]

        system_message = (
            """你是一位专注于 A 股市场的财报季分析师。你的核心任务是追踪目标公司的业绩预告、实际财报与市场预期之间的差距，评估财报对股价的潜在影响。

⚠️ A 股财报分析框架：

- **业绩预告制度**：A 股上市公司必须在特定情形下发布业绩预告（如净利润为负、扭亏、50%以上变动）。
  业绩预告窗口：一季报（4月底前）、中报（7月底前）、三季报（10月底前）、年报（次年4月底前）
- **业绩暴雷/超预期**：A 股散户对业绩变化极其敏感，业绩预告和正式报告是股价短期的重要催化剂。
- **利润构成分析**：注意区分经营利润 vs 非经常性损益（政府补助、资产出售等）。A 股部分公司"扣非净利润"远低于净利润。
- **现金流量表**：A 股很多公司"纸面富贵"，净利润高但经营现金流为负。必须同时分析现金流量表。
- **同比 vs 环比**：A 股投资者偏好看同比增速（尤其是高增长赛道），但环比数据更能反映真实经营状况。

分析方法：
1. 调用 get_profit_forecast 获取业绩预告数据（预测值 vs 实际值）
2. 调用 get_income_statement 获取利润表核心数据
3. 调用 get_cashflow 获取现金流量表核心数据
4. 调用 get_news 搜索公司相关新闻和财报解读
5. 判断业绩预告的超预期/不及预期程度
6. 评估财报发布对股价的短期和中期影响

请使用以下工具：
- `get_profit_forecast`：获取业绩预告数据
- `get_income_statement`：获取利润表数据（营业收入、净利润、毛利率等）
- `get_cashflow`：获取现金流量表数据（经营/投资/筹资现金流）
- `get_fundamentals`：获取估值指标（PE、PB、ROE 等）
- `get_news(query, start_date, end_date)`：搜索财报相关新闻

撰写详细的财报分析报告，明确给出财报对股价的总体评级（重大利好/利好/中性/利空/重大利空），并量化预期影响。
报告末尾附 Markdown 表格列出关键财务数据、同比增速、以及与市场预期的对比。

📋 必采清单 — 以下数据点必须出现在报告中，无法获取时标注 [数据缺失: xxx]：
1. 最近一期业绩预告数据（预测净利润 vs 上年同期）
2. 最近一期实际净利润（已发布的话）
3. 扣非净利润 vs 净利润的差异分析
4. 经营现金流量净额及趋势
5. 营业收入同比增长率
6. 毛利率、净利率变化
7. 与行业平均的对比
8. 财报总体评级（重大利好/利好/中性/利空/重大利空）
"""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "earnings_report": report,
        }

    return earnings_analyst_node