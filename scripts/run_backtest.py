#!/usr/bin/env python3
"""
回测运行脚本

用法：
  python run_backtest.py                              # 默认参数
  python run_backtest.py --stock 601006 --name 大秦铁路  # 指定股票
  python run_backtest.py --stock 600519 --name 贵州茅台 --capital 500000
  python run_backtest.py --all                         # 运行所有默认标的
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 激活vnpy环境
sys.path.insert(0, os.path.expanduser("~/vnpy_env/lib/python3.11/site-packages"))

from backtest.vnpy_backtest import VnPyBacktester, BacktestConfig, generate_sample_decisions, DecisionParser
from config.backtest_config import BACKTEST_CONFIG, DEFAULT_STOCKS, DEFAULT_PERIOD

def run_single(stock: dict, chart: bool = True):
    """运行单只股票回测"""
    config = BacktestConfig(
        initial_capital=BACKTEST_CONFIG["initial_capital"],
        position_sizing=BACKTEST_CONFIG["position_sizing"],
        commission_rate=BACKTEST_CONFIG["commission_rate"],
        slippage_rate=BACKTEST_CONFIG["slippage_rate"],
    )
    backtester = VnPyBacktester(config)
    decisions = generate_sample_decisions(
        stock["ticker"], DEFAULT_PERIOD["start"], DEFAULT_PERIOD["end"]
    )
    result = backtester.backtest(
        stock["ticker"], decisions, DEFAULT_PERIOD["start"], DEFAULT_PERIOD["end"], stock["name"]
    )
    backtester.print_report(result)

    if chart:
        chart_path = backtester.generate_charts("/Users/frank/TradingAgents-Astock-Pro/data", stock["name"])
        if chart_path:
            print(f"  📊 图表: {chart_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="TradingAgents-Astock-Pro 回测运行")
    parser.add_argument("--stock", type=str, default=None, help="股票代码")
    parser.add_argument("--name", type=str, default=None, help="股票名称")
    parser.add_argument("--capital", type=float, default=None, help="初始资金")
    parser.add_argument("--position", type=float, default=None, help="仓位比例")
    parser.add_argument("--all", action="store_true", help="运行所有默认标的")
    parser.add_argument("--no-chart", action="store_true", help="不生成图表")
    args = parser.parse_args()

    if args.all:
        for stock in DEFAULT_STOCKS:
            print(f"\n{'#'*40}")
            print(f"# {stock['name']} ({stock['ticker']})")
            print(f"{'#'*40}")
            run_single(stock, chart=not args.no_chart)
    elif args.stock:
        stock = {"ticker": args.stock, "name": args.name or args.stock}
        run_single(stock, chart=not args.no_chart)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()