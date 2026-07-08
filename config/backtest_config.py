"""回测配置"""

BACKTEST_CONFIG = {
    "initial_capital": 1_000_000,      # 初始资金
    "commission_rate": 0.001,          # 手续费
    "slippage_rate": 0.001,            # 滑点
    "position_sizing": 0.3,            # 仓位比例
    "min_position": 100,               # 最小手数
    "fast_window": 10,                 # 快线窗口
    "slow_window": 30,                 # 慢线窗口
}

# 默认测试标的
DEFAULT_STOCKS = [
    {"ticker": "600519", "name": "贵州茅台"},
    {"ticker": "601006", "name": "大秦铁路"},
    {"ticker": "000858", "name": "五粮液"},
]

# 默认回测区间
DEFAULT_PERIOD = {
    "start": "20250101",
    "end": "20261231",
}