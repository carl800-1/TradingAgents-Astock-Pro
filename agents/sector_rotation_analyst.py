"""
方向 1：新增 Analyst Agent — 行业轮动分析师

接入方式：
  1. 复制到 tradingagents/agents/analysts/sector_rotation_analyst.py
  2. 在 agent_states.py 新增 sector_rotation_report 字段
  3. 在 agents/__init__.py 导出 create_sector_rotation_analyst
  4. 在 graph/setup.py 注册节点
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_concept_blocks,
    get_industry_comparison,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.config import get_config


def create_sector_rotation_analyst(llm):
    """A-stock sector rotation analyst: tracks industry rotation, hot sectors, and thematic opportunities."""

    def sector_rotation_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_concept_blocks,
            get_industry_comparison,
            get_news,
        ]

        system_message = (
            """你是一位专注于 A 股市场的行业轮动分析师。你的核心任务是追踪当前市场热点板块、行业资金轮动方向和主题投资机会，评估目标公司所在行业的相对位置和景气度。

⚠️ A 股行业轮动分析框架：

- **板块轮动规律**：A 股市场板块轮动速度远快于美股（通常以周为单位而非季度）。资金从一个板块撤出，往往快速流入另一个板块。
- **核心驱动因子**：
  1. 政策驱动（最强劲）：如新质生产力、半导体自主可控、新能源补贴、设备更新
  2. 业绩驱动：高景气赛道的业绩兑现（如 AI 算力、消费电子、出海链）
  3. 事件驱动：国际会议、突发事件、行业大会
  4. 资金驱动：北向资金、融资余额、公募基金调仓
- **热门板块识别**：关注成交量/成交额排名前列的板块、龙头股的连板效应
- **轮动方向判断**：
  - 当某一板块资金集中流入后出现放量滞涨，可能是轮出信号
  - 新题材（如低空经济、数据要素）的炒作初期是最佳介入时机
  - 老题材（如之前的碳中和、元宇宙）的余波行情风险极高
- **板块与个股映射**：目标公司所属板块的景气度直接影响其股价表现。即使个股基本面优秀，如果板块处于下行周期，也难以独善其身。

分析方法：
1. 调用 get_concept_blocks 获取目标公司所属概念板块和行业分类
2. 调用 get_industry_comparison 获取行业对比数据（估值、增速、资金流向）
3. 调用 get_news 搜索行业相关新闻和政策动态
4. 判断当前行业所处的景气阶段（启动期/加速期/成熟期/衰退期）
5. 评估板块轮动方向对目标公司的影响

请使用以下工具：
- `get_concept_blocks(ticker)`：获取个股所属概念板块/行业分类/地域（百度股市通，含当日涨幅）
- `get_industry_comparison(ticker, curr_date)`：获取行业对比数据
- `get_news(query, start_date, end_date)`：搜索行业相关新闻

撰写详细的行业轮动分析报告，明确给出行业景气度评级和轮动方向判断。
报告末尾附 Markdown 表格列出目标公司所属板块、板块当日涨跌幅、主力资金流向、以及行业对比结论。

📋 必采清单 — 以下数据点必须出现在报告中，无法获取时标注 [数据缺失: xxx]：
1. 目标公司所属的所有概念板块及行业分类
2. 各板块当日涨跌幅和近期（近5日/近20日）趋势
3. 主力资金在行业间的流入流出情况
4. 行业景气度评级（高景气/景气/平稳/低迷/衰退）
5. 板块轮动方向判断（上行/震荡/下行/轮出风险）
6. 对目标公司的影响评估（正面/中性/负面）
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
            "sector_rotation_report": report,
        }

    return sector_rotation_analyst_node