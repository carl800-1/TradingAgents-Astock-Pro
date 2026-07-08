"""数据层 — 统一数据获取 + 实时行情"""

from .data_layer import (
    DataLayer,
    AKShareSource,
    get_realtime_price,
    get_realtime_prices,
    check_data_quality,
    get_data_with_fallback,
)

__all__ = [
    "DataLayer",
    "AKShareSource",
    "get_realtime_price",
    "get_realtime_prices",
    "check_data_quality",
    "get_data_with_fallback",
]