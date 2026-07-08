"""
方向 3：vn.py 回测验证模块 — TradingAgents 决策 → vn.py 回测

核心功能：
  1. 解析 TradingAgents 的决策输出（结构化 + 文本提取）
  2. 将决策转化为 vn.py 的订单信号
  3. 用 vn.py 的 backtesting engine 执行回测
  4. 输出性能指标和可视化图表

接入方式：
  python vnpy_backtest.py --stock 600519 --start 20220101 --end 20241231
  python vnpy_backtest.py --mode decisions --file decisions.json
"""
import sys
import os
import json
import re
import argparse
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

# 激活 vnpy 环境
sys.path.insert(0, os.path.expanduser("~/vnpy_env/lib/python3.11/site-packages"))

import baostock as bs
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# macOS 中文字体配置
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一部分：TradingAgents 决策解析器
# ============================================================

@dataclass
class AgentDecision:
    """单个时间点的决策"""
    date: str
    action: str            # BUY/HOLD/SELL
    confidence: float      # 0-1
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_sizing: Optional[float] = None  # 仓位比例
    source: str = 'manual'  # manual / portfolio_manager / trader

@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 1_000_000.0  # 初始资金 100万
    commission_rate: float = 0.001          # 手续费 0.1%
    slippage_rate: float = 0.001            # 滑点 0.1%
    position_sizing: float = 0.3            # 默认仓位 30%
    min_position: int = 100                 # 最小手数 100股

class DecisionParser:
    """解析 TradingAgents 输出为结构化决策"""

    @staticmethod
    def parse_structured(
        recommendation: str,
        action: str,
        price_target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        position_sizing: Optional[str] = None,
        confidence: float = 1.0,
    ) -> AgentDecision:
        """
        解析结构化决策输出（来自 PortfolioManager / Trader）

        Args:
            recommendation: 来自 Research Manager (Buy/Overweight/Hold/Underweight/Sell)
            action: 来自 Trader (Buy/Hold/Sell)
            price_target: 目标价
            stop_loss: 止损价
            position_sizing: 仓位说明 (如 "5% of portfolio")
            confidence: 置信度 0-1
        """
        # 统一映射到 BUY/HOLD/SELL
        action_map = {
            'buy': 'BUY', 'overweight': 'BUY',
            'hold': 'HOLD',
            'sell': 'SELL', 'underweight': 'SELL',
        }
        final_action = action_map.get(action.lower(), 'HOLD')

        # 解析仓位比例
        sizing = None
        if position_sizing:
            match = re.search(r'(\d+\.?\d*)\s*%', str(position_sizing))
            if match:
                sizing = float(match.group(1)) / 100

        return AgentDecision(
            date='',
            action=final_action,
            confidence=confidence,
            entry_price=price_target,
            stop_loss=stop_loss,
            position_sizing=sizing,
            source='structured',
        )

    @staticmethod
    def extract_from_text(text: str) -> AgentDecision:
        """
        从非结构化的 Agent 输出文本中提取决策

        支持格式：
          FINAL TRANSACTION PROPOSAL: **BUY**
          **Rating**: Buy
          **Action**: Buy
        """
        # 尝试提取 FINAL TRANSACTION PROPOSAL
        match = re.search(r'FINAL TRANSACTION PROPOSAL:\s*\*\*(BUY|HOLD|SELL)\*\*', text, re.IGNORECASE)
        if match:
            action = match.group(1).upper()
        else:
            # 尝试提取 Rating
            match = re.search(r'\*\*Rating\*\*:\s*(Buy|Overweight|Hold|Underweight|Sell)', text, re.IGNORECASE)
            if match:
                action_map = {'buy': 'BUY', 'overweight': 'BUY', 'hold': 'HOLD', 'underweight': 'SELL', 'sell': 'SELL'}
                action = action_map.get(match.group(1).lower(), 'HOLD')
            else:
                # 尝试提取 Action
                match = re.search(r'\*\*Action\*\*:\s*(Buy|Hold|Sell)', text, re.IGNORECASE)
                if match:
                    action = match.group(1).upper()
                else:
                    action = 'HOLD'

        # 提取目标价
        price_match = re.search(r'\*\*Price Target\*\*:\s*([\d.]+)', text, re.IGNORECASE)
        target_price = float(price_match.group(1)) if price_match else None

        # 提取止损价
        sl_match = re.search(r'\*\*Stop Loss\*\*:\s*([\d.]+)', text, re.IGNORECASE)
        stop_loss = float(sl_match.group(1)) if sl_match else None

        return AgentDecision(
            date='',
            action=action,
            confidence=0.5,  # 从文本提取的置信度默认 0.5
            entry_price=target_price,
            stop_loss=stop_loss,
            source='text_extract',
        )

    @staticmethod
    def load_decisions_from_file(filepath: str) -> list[AgentDecision]:
        """
        从文件加载决策列表

        支持格式：
        1. JSON 数组: [{"date": "2024-01-15", "action": "BUY", "confidence": 0.8}, ...]
        2. CSV: date,action,confidence,entry_price,stop_loss
        """
        decisions = []

        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    d = AgentDecision(
                        date=item.get('date', ''),
                        action=item.get('action', 'HOLD'),
                        confidence=item.get('confidence', 0.5),
                        entry_price=item.get('entry_price'),
                        stop_loss=item.get('stop_loss'),
                        position_sizing=item.get('position_sizing'),
                        source=item.get('source', 'manual'),
                    )
                    decisions.append(d)
        elif filepath.endswith('.csv'):
            import csv
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = AgentDecision(
                        date=row.get('date', ''),
                        action=row.get('action', 'HOLD'),
                        confidence=float(row.get('confidence', 0.5)),
                        entry_price=float(row.get('entry_price', 0)) if row.get('entry_price') else None,
                        stop_loss=float(row.get('stop_loss', 0)) if row.get('stop_loss') else None,
                        position_sizing=float(row.get('position_sizing', 0)) if row.get('position_sizing') else None,
                        source=row.get('source', 'manual'),
                    )
                    decisions.append(d)

        return decisions


# ============================================================
# 第二部分：vn.py 回测引擎（简化版，基于 baostock 数据）
# ============================================================

class VnPyBacktester:
    """
    基于 vn.py 理念的回测引擎（使用 baostock 数据 + 自定义逻辑）

    支持：
    - T+1 交易制度
    - 涨跌停判断
    - 手续费 + 滑点
    - 仓位管理
    - 最大回撤计算
    - 夏普比率计算
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.results = []
        self._position = 0  # 0=空仓, 1=持有
        self._entry_price = 0.0
        self._cash = self.config.initial_capital
        self._peak = self.config.initial_capital
        self._trades = []  # 交易记录

    def _can_buy(self, df: pd.DataFrame, idx: int) -> bool:
        """检查是否可以买入（T+1、涨跌停）"""
        if idx < 1:
            return True

        # T+1 检查：如果昨天刚买入，今天不能卖出（这里检查是否可以买入）
        return True

    def _can_sell(self, df: pd.DataFrame, idx: int) -> bool:
        """检查是否可以卖出（T+1）"""
        return self._position == 1

    def _is_limit_up(self, row) -> bool:
        """判断涨停"""
        if pd.isna(row.get('preclose', 0)):
            return False
        return row['high'] >= row['preclose'] * 1.1  # 主板涨停

    def _is_limit_down(self, row) -> bool:
        """判断跌停"""
        if pd.isna(row.get('preclose', 0)):
            return False
        return row['low'] <= row['preclose'] * 0.9  # 主板跌停

    def backtest(
        self,
        ticker: str,
        decisions: list[AgentDecision],
        start_date: str,
        end_date: str,
        name: str = "",
    ) -> dict:
        """
        执行回测

        Args:
            ticker: 股票代码 (如 '600519')
            decisions: 决策列表
            start_date: 开始日期
            end_date: 结束日期
            name: 股票名称

        Returns:
            dict: 回测结果
        """
        # 获取数据
        df = self._fetch_data(ticker, start_date, end_date)
        if df is None or len(df) < 2:
            return {'error': '数据获取失败'}

        # 匹配决策到日期
        decision_map = {d.date: d for d in decisions}

        # 初始化回测结果
        results = df.copy()
        results['position'] = 0
        results['trade_signal'] = 0
        results['trade_price'] = 0.0
        results['cash'] = float(self.config.initial_capital)
        results['holdings_value'] = 0.0
        results['total_value'] = float(self.config.initial_capital)

        self._position = 0
        self._cash = self.config.initial_capital
        self._trades = []
        self._peak = self.config.initial_capital

        # 逐日回测
        for i in range(1, len(results)):
            row = results.iloc[i]
            date_str = row['date'].strftime('%Y-%m-%d')
            prev_row = results.iloc[i-1]

            # 匹配决策到日期
            action = 'HOLD'
            for d in decisions:
                if d.date == date_str:
                    action = d.action
                    break

            current_close = row['close']
            prev_close = prev_row['close']

            # T+1 检查：今天买入的不能今天卖
            just_bought_today = results.loc[i, 'trade_signal'] == 1

            if action == 'BUY' and self._position == 0 and self._cash > 0:
                # 买入逻辑
                buy_price = row['open']  # 次日开盘买入（T+1）
                if buy_price <= 0 or pd.isna(buy_price):
                    buy_price = current_close

                # 涨停检查
                if self._is_limit_up(row):
                    continue

                # 计算可买数量
                stock_price = buy_price
                total_cost = stock_price * self.config.min_position
                available_shares = int((self._cash * self.config.position_sizing) / stock_price)
                available_shares = int(available_shares / 100) * 100  # 整手

                if available_shares < self.config.min_position:
                    continue

                commission = available_shares * stock_price * self.config.commission_rate
                slippage = available_shares * stock_price * self.config.slippage_rate
                total_cost = available_shares * stock_price + commission + slippage

                if total_cost <= self._cash:
                    self._position = 1
                    self._entry_price = buy_price
                    self._cash -= total_cost

                    results.loc[i, 'trade_signal'] = 1
                    results.loc[i, 'trade_price'] = buy_price

                    self._trades.append({
                        'date': date_str,
                        'type': 'BUY',
                        'price': buy_price,
                        'shares': available_shares,
                        'cost': total_cost,
                        'commission': commission,
                        'slippage': slippage,
                    })

            elif action == 'SELL' and self._position == 1 and not just_bought_today:
                # 卖出逻辑
                sell_price = row['open']
                if sell_price <= 0 or pd.isna(sell_price):
                    sell_price = current_close

                shares = int((self.config.initial_capital * self.config.position_sizing) / self._entry_price)
                shares = int(shares / 100) * 100

                commission = shares * sell_price * self.config.commission_rate
                slippage = shares * sell_price * self.config.slippage_rate
                proceeds = shares * sell_price - commission - slippage
                pnl = (sell_price - self._entry_price) * shares - commission - slippage

                self._cash += proceeds
                self._position = 0

                results.loc[i, 'trade_signal'] = -1
                results.loc[i, 'trade_price'] = sell_price

                self._trades.append({
                    'date': date_str,
                    'type': 'SELL',
                    'price': sell_price,
                    'shares': shares,
                    'proceeds': proceeds,
                    'pnl': pnl,
                    'commission': commission,
                    'slippage': slippage,
                })

            # 持仓市值
            if self._position == 1:
                shares = int((self.config.initial_capital * self.config.position_sizing) / self._entry_price)
                shares = int(shares / 100) * 100
                results.loc[i, 'holdings_value'] = shares * current_close
                results.loc[i, 'position'] = 1

            # 总资产
            results.loc[i, 'cash'] = self._cash
            total_value = self._cash + results.loc[i, 'holdings_value']
            results.loc[i, 'total_value'] = total_value

            # 更新峰值
            self._peak = max(self._peak, total_value)

        # 计算统计指标
        stats = self._calculate_stats(results, name)

        self.results = results

        return {
            'stats': stats,
            'results': results,
            'trades': self._trades,
            'config': asdict(self.config),
        }

    def _fetch_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从 BaoStock 获取数据"""
        code = ticker.replace('.', '')
        if not code.startswith(('sh', 'sz')):
            code = 'sh.' + code if code.startswith('6') else 'sz.' + code

        lg = bs.login()
        if lg.error_code != '0':
            print(f"❌ BaoStock 登录失败: {lg.error_msg}")
            return None

        # 日期格式统一为 YYYY-MM-DD
        def _to_date_str(d):
            d = str(d).replace('-', '')
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        start = _to_date_str(start_date)
        end = _to_date_str(end_date)

        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,preclose,turn",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="3",
        )

        if rs.error_code != '0':
            print(f"❌ 数据获取失败: {rs.error_msg}")
            bs.logout()
            return None

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        df = pd.DataFrame(data_list, columns=rs.fields)
        bs.logout()

        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'preclose', 'turn']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        return df

    def _calculate_stats(self, df: pd.DataFrame, name: str) -> dict:
        """计算回测统计指标"""
        total_value = df['total_value']
        initial = self.config.initial_capital
        final = total_value.iloc[-1]
        total_return = (final - initial) / initial

        # 累计基准收益（买入持有）
        df['benchmark_return'] = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1

        # 最大回撤
        cum_value = total_value
        peak = cum_value.cummax()
        drawdown = (cum_value - peak) / peak
        max_drawdown = drawdown.min()

        # 年化收益
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1

        # 夏普比率
        daily_returns = total_value.pct_change().iloc[1:]
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        # 胜率
        buy_trades = [t for t in self._trades if t['type'] == 'BUY']
        sell_trades = [t for t in self._trades if t['type'] == 'SELL']
        wins = sum(1 for t in sell_trades if t.get('pnl', 0) > 0)
        win_rate = wins / len(sell_trades) if sell_trades else 0

        # 平均每笔交易收益
        avg_pnl = np.mean([t.get('pnl', 0) for t in sell_trades]) if sell_trades else 0

        # 盈亏比
        winning_pnls = [t['pnl'] for t in sell_trades if t.get('pnl', 0) > 0]
        losing_pnls = [abs(t['pnl']) for t in sell_trades if t.get('pnl', 0) < 0]
        profit_factor = (sum(winning_pnls) / sum(losing_pnls)) if losing_pnls and sum(losing_pnls) > 0 else float('inf')

        return {
            'name': name or ticker,
            'backtest_period': f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}",
            'total_return': round(total_return * 100, 2),
            'benchmark_return': round(df['benchmark_return'].iloc[-1] * 100, 2),
            'excess_return': round((total_return - df['benchmark_return'].iloc[-1]) * 100, 2),
            'annual_return': round(annual_return * 100, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'total_trades': len(self._trades) // 2,
            'win_rate': round(win_rate * 100, 2),
            'avg_pnl': round(avg_pnl, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'N/A',
            'final_value': round(final, 2),
            'initial_value': round(initial, 2),
        }

    def generate_charts(self, output_dir: str, name: str = ""):
        """生成回测图表"""
        df = self.results
        if df is None or 'total_value' not in df.columns:
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f"TradingAgents 回测 — {name}", fontsize=16)

        # 图1：净值曲线
        ax1 = axes[0]
        ax1.plot(df['date'], df['total_value'], label='策略净值', linewidth=2, color='#2196F3')
        ax1.plot(df['date'], df['total_value'].iloc[0] * (1 + df['benchmark_return']),
                 label='基准（买入持有）', linewidth=1, color='#FF9800', linestyle='--')

        # 标记买卖点
        buy_mask = df['trade_signal'] == 1
        sell_mask = df['trade_signal'] == -1
        ax1.scatter(df.loc[buy_mask, 'date'], df.loc[buy_mask, 'total_value'],
                   marker='^', color='red', s=60, label='买入', zorder=5)
        ax1.scatter(df.loc[sell_mask, 'date'], df.loc[sell_mask, 'total_value'],
                   marker='v', color='green', s=60, label='卖出', zorder=5)

        ax1.set_ylabel('总资产（元）')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/10000:.0f}万'))

        # 图2：回撤曲线
        ax2 = axes[1]
        cum_value = df['total_value']
        peak = cum_value.cummax()
        drawdown = (cum_value - peak) / peak
        ax2.fill_between(df['date'], drawdown * 100, 0, color='#F44336', alpha=0.3)
        ax2.plot(df['date'], drawdown * 100, color='#F44336', linewidth=1)
        ax2.set_ylabel('回撤（%）')
        ax2.set_xlabel('日期')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(output_dir, f"{name}_backtest.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        return filepath

    def print_report(self, backtest_result: dict):
        """打印回测报告"""
        stats = backtest_result['stats']
        if 'error' in backtest_result:
            print(f"❌ 回测失败: {backtest_result['error']}")
            return

        config = backtest_result['config']
        trades = backtest_result['trades']

        print(f"\n{'='*60}")
        print(f"  TradingAgents × vn.py 回测报告")
        print(f"{'='*60}")
        print(f"  标的:        {stats['name']}")
        print(f"  回测区间:    {stats['backtest_period']}")
        print(f"  初始资金:    ¥{config['initial_capital']:,.0f}")
        print(f"  仓位比例:    {config['position_sizing']*100:.0f}%")
        print(f"  手续费:      {config['commission_rate']*100:.2f}%")
        print(f"  滑点:        {config['slippage_rate']*100:.2f}%")
        print(f"{'='*60}")
        print(f"  最终资产:    ¥{stats['final_value']:,.2f}")
        print(f"  总收益率:    {stats['total_return']:+.2f}%")
        print(f"  基准收益率:  {stats['benchmark_return']:+.2f}%")
        print(f"  超额收益:    {stats['excess_return']:+.2f}%")
        print(f"  年化收益率:  {stats['annual_return']:+.2f}%")
        print(f"  夏普比率:    {stats['sharpe_ratio']:.2f}")
        print(f"  最大回撤:    {stats['max_drawdown']:.2f}%")
        print(f"  交易次数:    {stats['total_trades']} 笔")
        print(f"  胜率:        {stats['win_rate']:.2f}%")
        print(f"  盈亏比:      {stats['profit_factor']}")
        print(f"{'='*60}")

        # 打印交易明细
        if trades:
            print(f"\n  交易明细：")
            print(f"  {'-'*50}")
            print(f"  {'日期':<12} {'类型':<6} {'价格':>10} {'数量':>8} {'金额/PnL':>12}")
            print(f"  {'-'*50}")
            for t in trades:
                if t['type'] == 'BUY':
                    print(f"  {t['date']:<12} {'BUY':<6} {t['price']:>10.2f} {t['shares']:>8} ¥{t['cost']:>11,.2f}")
                else:
                    pnl_str = f"¥{t['pnl']:+,.2f}"
                    print(f"  {t['date']:<12} {'SELL':<6} {t['price']:>10.2f} {t['shares']:>8} {pnl_str:>12}")
            print(f"  {'-'*50}")

        # 评估等级
        total_ret = stats['total_return']
        if total_ret > 50:
            grade = "🌟🌟🌟 优秀（超跑基准）"
        elif total_ret > 20:
            grade = "🌟🌟 良好（跑赢基准）"
        elif total_ret > 0:
            grade = "🌟 及格（有收益）"
        else:
            grade = "❌ 不及格（亏损）"

        print(f"\n  综合评估: {grade}")
        print(f"{'='*60}\n")


# ============================================================
# 第三部分：模拟决策生成（用于测试回测框架）
# ============================================================

def generate_sample_decisions(
    ticker: str,
    start_date: str,
    end_date: str,
    signal_type: str = 'moving_average',
) -> list[AgentDecision]:
    """
    生成模拟决策（基于技术指标），用于测试回测框架

    注意：这只是为了测试回测引擎，实际使用时应从 TradingAgents 获取决策
    """
    decisions = []
    code = ticker.replace('.', '')
    if not code.startswith(('sh', 'sz')):
        code = 'sh.' + code if code.startswith('6') else 'sz.' + code

    # 日期格式统一为 YYYY-MM-DD
    def _to_date_str(d):
        d = d.replace('-', '')
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    start = _to_date_str(start_date)
    end = _to_date_str(end_date)

    lg = bs.login()
    rs = bs.query_history_k_data_plus(
        code, "date,close,open,high,low,volume",
        start_date=start, end_date=end,
        frequency="d", adjustflag="3",
    )
    data = []
    while (rs.error_code == '0') & rs.next():
        data.append(rs.get_row_data())
    bs.logout()

    df = pd.DataFrame(data, columns=rs.fields)
    for col in ['close', 'open', 'high', 'low', 'volume']:
        df[col] = pd.to_numeric(df[col])

    # 双均线信号
    df['fast_ma'] = df['close'].rolling(10).mean()
    df['slow_ma'] = df['close'].rolling(30).mean()

    for i in range(len(df)):
        date_str = df.iloc[i]['date']
        action = 'HOLD'

        if signal_type == 'moving_average':
            if i > 30:
                if df.iloc[i]['fast_ma'] > df.iloc[i]['slow_ma'] and df.iloc[i-1]['fast_ma'] <= df.iloc[i-1]['slow_ma']:
                    action = 'BUY'
                elif df.iloc[i]['fast_ma'] < df.iloc[i]['slow_ma'] and df.iloc[i-1]['fast_ma'] >= df.iloc[i-1]['slow_ma']:
                    action = 'SELL'

        d = AgentDecision(
            date=date_str,
            action=action,
            confidence=0.6,
            source='sample',
        )
        decisions.append(d)

    return decisions


# ============================================================
# 第四部分：主入口
# ============================================================

def run_backtest_demo():
    """演示回测"""
    stocks = [
        ("600519", "贵州茅台"),
        ("000858", "五粮液"),
    ]

    config = BacktestConfig(
        initial_capital=1_000_000,
        position_sizing=0.3,
        commission_rate=0.001,
        slippage_rate=0.001,
    )

    backtester = VnPyBacktester(config)

    for ticker, name in stocks:
        decisions = generate_sample_decisions(ticker, "20220101", "20241231")
        result = backtester.backtest(ticker, decisions, "20220101", "20241231", name)
        backtester.print_report(result)

        # 生成图表
        chart_path = backtester.generate_charts("/Users/frank/vnpy_project/data", name)
        if chart_path:
            print(f"  📊 图表已保存: {chart_path}")

    print("\n✅ 回测演示完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradingAgents × vn.py 回测验证")
    parser.add_argument("--stock", type=str, default="600519", help="股票代码")
    parser.add_argument("--start", type=str, default="20220101", help="开始日期")
    parser.add_argument("--end", type=str, default="20241231", help="结束日期")
    parser.add_argument("--mode", type=str, choices=["sample", "decisions"], default="sample",
                        help="sample=模拟决策, decisions=加载决策文件")
    parser.add_argument("--file", type=str, default="", help="决策文件路径（JSON/CSV）")
    parser.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    parser.add_argument("--position", type=float, default=0.3, help="仓位比例")
    parser.add_argument("--chart", action="store_true", help="生成图表")
    parser.add_argument("--name", type=str, default="", help="股票名称")

    args = parser.parse_args()

    config = BacktestConfig(
        initial_capital=args.capital,
        position_sizing=args.position,
    )

    backtester = VnPyBacktester(config)

    if args.mode == "decisions" and args.file:
        decisions = DecisionParser.load_decisions_from_file(args.file)
    else:
        decisions = generate_sample_decisions(args.stock, args.start, args.end)

    result = backtester.backtest(args.stock, decisions, args.start, args.end, args.name)
    backtester.print_report(result)

    if args.chart:
        chart_path = backtester.generate_charts("/Users/frank/vnpy_project/data", args.name or args.stock)
        if chart_path:
            print(f"📊 图表已保存: {chart_path}")