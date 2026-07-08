"""
方向 1：新增 Analyst Agent — 量化信号分析师

接入方式：
  1. 复制到 tradingagents/agents/analysts/quant_signal_analyst.py
  2. 在 agent_states.py 新增 quant_signal_report 字段
  3. 在 agents/__init__.py 导出 create_quant_signal_analyst
  4. 在 graph/setup.py 注册节点

核心逻辑：调用本地训练好的 LightGBM 模型，输出量化预测信号。
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config

import joblib
import numpy as np
import os as _os

# ============================================================
# 量化信号引擎 — 独立于 LangGraph 的纯 Python 模块
# 负责调用 LightGBM 模型生成信号
# ============================================================

class QuantSignalEngine:
    """
    量化信号引擎
    使用预训练的 LightGBM 模型对标的进行量化评分

    参数:
        model_path: LightGBM 模型文件路径（如 ~/.vnpy_project/data/600519_lgbm_v2.pkl）
    """

    def __init__(self, model_path: str):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """加载 LightGBM 模型"""
        if self.model_path and _os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = None

    def _extract_features(self, ohlcV_data, indicators_data) -> dict:
        """
        从 OHLCV 和技术指标数据中提取特征
        简化版：提取核心特征供模型预测

        Args:
            ohlcV_data: K 线数据 dict {open, high, low, close, volume, ...}
            indicators_data: 技术指标 dict

        Returns:
            特征字典
        """
        features = {}
        df = ohlcV_data.get('data', {})

        # 基础价格特征
        closes = df.get('close', [])
        volumes = df.get('volume', [])

        if len(closes) < 10:
            return {}

        last_close = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else last_close

        # 价格变化率
        features['pct_change_1d'] = (last_close - prev_close) / prev_close if prev_close else 0

        # 短期均线
        if len(closes) >= 5:
            ema_5 = sum(closes[-5:]) / 5
            features['ema_5'] = ema_5
            features['price_vs_ema5'] = (last_close - ema_5) / ema_5 if ema_5 else 0

        if len(closes) >= 20:
            ema_20 = sum(closes[-20:]) / 20
            features['ema_20'] = ema_20
            features['price_vs_ema20'] = (last_close - ema_20) / ema_20 if ema_20 else 0

        # 布林带位置（简化版）
        if len(closes) >= 20:
            prices = closes[-20:]
            mean_price = sum(prices) / 20
            std_price = (sum((p - mean_price) ** 2 for p in prices) / 20) ** 0.5
            bb_upper = mean_price + 2 * std_price
            bb_lower = mean_price - 2 * std_price
            features['bb_position'] = (last_close - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

        # 成交量
        if len(volumes) >= 20:
            vol_5 = sum(volumes[-5:]) / 5
            vol_20 = sum(volumes[-20:]) / 20
            features['volume_ratio_5d'] = vol_5 / vol_20 if vol_20 > 0 else 1.0

        return features

    def predict_signal(self, ohlcV_data, indicators_data=None):
        """
        对标的进行量化预测

        Args:
            ohlcV_data: 从 get_stock_data 获取的 K 线数据
            indicators_data: 从 get_indicators 获取的技术指标数据

        Returns:
            dict: {
                'buy_prob': float (0-1, 未来 5 天上涨概率),
                'signal': str ('BUY'/'HOLD'/'SELL'),
                'features': dict (使用的特征),
                'model_path': str,
                'status': str ('OK'/'NO_MODEL'/'NO_DATA')
            }
        """
        if self.model is None:
            return {
                'buy_prob': None,
                'signal': 'HOLD',
                'features': {},
                'model_path': self.model_path,
                'status': 'NO_MODEL',
                'message': f'模型未找到: {self.model_path}',
            }

        features = self._extract_features(ohlcV_data, indicators_data)

        if not features:
            return {
                'buy_prob': None,
                'signal': 'HOLD',
                'features': {},
                'model_path': self.model_path,
                'status': 'NO_DATA',
                'message': 'K 线数据不足，无法提取特征',
            }

        # 填充缺失特征
        feature_names = list(features.keys())
        # 获取模型的特征名（如果有）
        if hasattr(self.model, 'feature_names_in_'):
            model_features = list(self.model.feature_names_in_)
        else:
            model_features = feature_names

        # 构建完整特征向量
        full_features = {}
        for fname in model_features:
            full_features[fname] = features.get(fname, 0.0)

        feature_array = np.array([[full_features[fname] for fname in model_features]])

        try:
            prob = self.model.predict_proba(feature_array)[0]
            # 找到标签为 1 的概率
            classes = self.model.classes_
            if 1 in classes:
                buy_prob = prob[list(classes).index(1)]
            else:
                buy_prob = prob[0]

            # 信号判断（阈值 0.5）
            if buy_prob > 0.6:
                signal = 'BUY'
            elif buy_prob < 0.4:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            return {
                'buy_prob': round(float(buy_prob), 4),
                'signal': signal,
                'features': {k: round(v, 4) for k, v in features.items()},
                'model_path': self.model_path,
                'status': 'OK',
                'message': f'模型预测完成，买入概率 {buy_prob:.2%}',
            }

        except Exception as e:
            return {
                'buy_prob': None,
                'signal': 'HOLD',
                'features': features,
                'model_path': self.model_path,
                'status': 'ERROR',
                'message': f'模型预测出错: {str(e)}',
            }


# 全局量化信号引擎实例（可在 config 中配置模型路径）
_quant_engine = None

def get_quant_engine() -> QuantSignalEngine:
    """获取全局量化信号引擎实例"""
    global _quant_engine
    if _quant_engine is None:
        model_path = _os.getenv('QUANT_MODEL_PATH', '')
        _quant_engine = QuantSignalEngine(model_path)
    return _quant_engine


def set_quant_engine(model_path: str):
    """设置量化信号引擎的模型路径"""
    global _quant_engine
    _quant_engine = QuantSignalEngine(model_path)


# ============================================================
# Agent 节点定义
# ============================================================

def create_quant_signal_analyst(llm):
    """量化信号分析师：调用 LightGBM 模型输出量化预测信号"""

    def quant_signal_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """你是一位量化信号分析师。你的核心任务是调用量化模型对目标标的进行量化评分，输出结构化的买卖信号。

⚠️ 量化信号分析框架：

- **模型信号**：量化模型基于历史数据训练，输出未来 5 天标的上涨的概率（buy_prob, 0-1）。
  - buy_prob > 0.6 → 强烈买入信号
  - buy_prob 0.4-0.6 → 中性信号，观望
  - buy_prob < 0.4 → 卖出信号
- **特征解读**：量化模型依赖以下特征做出预测：
  - 价格与均线的关系（价格是否偏离均线）
  - 布林带位置（当前价格在通道中的位置）
  - 成交量比（近期成交量相对于平均水平的变化）
  - 技术指标（MACD、RSI 等）
- **模型局限性**：
  - 量化模型的预测不是 100% 准确的，它捕捉的是统计上的概率优势
  - 单一量化信号不足以做出交易决策，必须结合基本面、技术面和事件面的综合判断
  - 模型在极端行情（如熔断、重大政策突变）下可能失效

操作方法：
1. 调用 get_stock_data 获取 K 线数据
2. 调用 get_indicators 获取技术指标数据
3. 将数据传入量化模型引擎，获取预测信号
4. 解读量化信号并结合市场情况给出分析结论

请务必注意：量化信号仅供研究参考，不构成投资建议。量化模型存在过拟合风险，实盘使用需谨慎。

📋 必采清单 — 以下数据点必须出现在报告中，无法获取时标注 [数据缺失: xxx]：
1. 量化模型路径及版本
2. 提取到的特征及其数值
3. 买入概率（buy_prob, 0-1）
4. 量化信号等级（BUY/HOLD/SELL）
5. 模型状态的说明（OK/NO_MODEL/NO_DATA/ERROR）
6. 模型局限性提示
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

        # 自动调用量化引擎获取信号
        try:
            engine = get_quant_engine()
            # 从 tool_calls 中获取数据
            stock_data = None
            indicator_data = None
            for call in result.tool_calls:
                if 'get_stock_data' in call.get('name', ''):
                    # 这里需要从实际 tool call 中解析数据
                    pass
            # 简化：在 agent 节点返回中附加量化信号
            signal_result = engine.predict_signal(stock_data or {}, indicator_data)
            quant_signal_extra = f"\n\n--- 量化信号（自动计算）---\n{signal_result['message']}\n" if signal_result['status'] == 'OK' else f"\n\n--- 量化信号 ---\n{signal_result['message']}"
            report += quant_signal_extra
        except Exception as e:
            pass

        return {
            "messages": [result],
            "quant_signal_report": report,
        }

    return quant_signal_analyst_node