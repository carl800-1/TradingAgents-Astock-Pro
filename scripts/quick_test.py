#!/usr/bin/env python3
"""项目快速验证脚本"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.expanduser("~/vnpy_env/lib/python3.11/site-packages"))

def test_data_layer():
    """测试数据层"""
    print("=" * 50)
    print("  数据层测试")
    print("=" * 50)

    from data_layer.data_layer import get_realtime_price, DataLayer, check_data_quality

    # 实时行情
    price = get_realtime_price("600519")
    print(f"  贵州茅台实时: {price.get('price', 'N/A')} 元 (状态: {price.get('status')})")

    # K线数据
    layer = DataLayer()
    data, status = layer.get_kline("600519", "20250101", "20261231")
    n_points = len(data.get('data', {}).get('close', []))
    print(f"  K线数据: {n_points} 条 (来源: {status})")

    # 数据质量
    quality = check_data_quality(data, source=status)
    print(f"  质量评分: {quality['quality_score']} ({quality['status']})")
    print()

def test_backtest():
    """测试回测"""
    print("=" * 50)
    print("  回测引擎测试")
    print("=" * 50)

    from backtest.vnpy_backtest import VnPyBacktester, generate_sample_decisions

    backtester = VnPyBacktester()
    decisions = generate_sample_decisions("601006", "20250101", "20261231")
    result = backtester.backtest("601006", decisions, "20250101", "20261231", "大秦铁路")
    backtester.print_report(result)
    print()

def main():
    print("\n" + "=" * 50)
    print("  TradingAgents-Pro 快速验证")
    print("=" * 50 + "\n")

    try:
        test_data_layer()
    except Exception as e:
        print(f"  ❌ 数据层测试失败: {e}\n")

    try:
        test_backtest()
    except Exception as e:
        print(f"  ❌ 回测测试失败: {e}\n")

    print("=" * 50)
    print("  ✅ 验证完成")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()