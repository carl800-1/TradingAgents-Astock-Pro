"""
方向 2：优化数据层 — AKShare 回退 + 实时行情

新增功能：
1. akshare 数据源封装（作为 astock mootdx 的回退通道）
2. 实时行情数据接口（东方财富推送服务）
3. 统一数据获取层（优先 mootdx，失败时回退到 AKShare）

使用方法：
  from data_layer import DataLayer, get_realtime_price
  data_layer = DataLayer()
  df = data_layer.get_kline('600519')  # 自动回退
  price = get_realtime_price('600519')
"""
import time
import requests as _requests
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def _normalize_ticker(code: str) -> str:
    """统一股票代码格式，返回纯6位数字"""
    s = code.strip().upper()
    for suffix in (".SH", ".SZ", ".BJ", ".BJ"):
        if s.endswith(suffix):
            s = s[:-3]
            break
    for prefix in ("SH", "SZ", "BJ"):
        if s.startswith(prefix):
            s = s[2:]
            break
    return s


def _market_prefix(code: str) -> str:
    """返回市场前缀：6/9开头→sh，其他→sz"""
    if code.startswith(("6", "9")):
        return "sh"
    elif code.startswith("8"):
        return "bj"
    return "sz"


# ============================================================
# 1. 实时行情接口（东方财富推送服务）
# ============================================================

def get_realtime_price(ticker: str) -> dict:
    """
    获取实时行情快照

    Args:
        ticker: 股票代码（如 '600519'）

    Returns:
        dict: {
            'code': str,
            'name': str,
            'price': float,           # 最新价
            'open': float,            # 开盘价
            'high': float,            # 最高价
            'low': float,             # 最低价
            'prev_close': float,      # 昨收价
            'volume': float,          # 成交量(股)
            'amount': float,          # 成交额(元)
            'pct_change': float,      # 涨跌幅(%)
            'timestamp': str,         # 更新时间
            'status': str             # OK/ERROR
        }
    """
    code = _normalize_ticker(ticker)
    prefix = _market_prefix(code)
    url = f"http://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": f"{prefix}{code}",
        "fields": "f43,f44,f45,f46,f47,f48,f50,f51,f52,f57,f58,f60,f170,f171",
        "ut": "fa5fd1943c7b386f172d6893dbbd1b60",
    }

    try:
        resp = _requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {})

        if not data:
            return {'status': 'ERROR', 'message': '数据为空', 'code': code}

        return {
            'code': code,
            'name': data.get('f58', ''),
            'price': float(data.get('f43', 0)),
            'open': float(data.get('f44', 0)),
            'high': float(data.get('f45', 0)),
            'low': float(data.get('f46', 0)),
            'prev_close': float(data.get('f47', 0)),
            'volume': int(data.get('f48', 0)),
            'amount': int(data.get('f49', 0)),
            'pct_change': float(data.get('f50', 0)),
            'timestamp': data.get('f50', ''),  # 时间戳
            'status': 'OK',
        }

    except Exception as e:
        logger.warning(f"获取实时行情失败 {code}: {e}")
        return {
            'status': 'ERROR',
            'message': str(e),
            'code': code,
        }


def get_realtime_prices(tickers: list[str]) -> list[dict]:
    """批量获取多只股票的实时行情"""
    return [get_realtime_price(t) for t in tickers]


# ============================================================
# 2. AKShare 数据源封装
# ============================================================

class AKShareSource:
    """AKShare 数据源封装 — 作为 mootdx 的回退通道"""

    def get_kline(
        self,
        ticker: str,
        start_date: str = "20230101",
        end_date: str = "20261231",
        period: str = "daily",
    ) -> tuple:
        """
        获取 K 线数据

        Args:
            ticker: 股票代码
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            period: 'daily' / 'weekly' / 'monthly'

        Returns:
            (data, status) — data 是 K 线 dict，status 是 'OK'/'ERROR'
        """
        code = _normalize_ticker(ticker)
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        symbol = f"{prefix}{code}"

        try:
            import akshare as ak

            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or df.empty:
                return {}, 'ERROR'

            # 转为 TradingAgents 兼容的格式
            kline_data = {
                'open': df['开盘'].tolist(),
                'high': df['最高'].tolist(),
                'low': df['最低'].tolist(),
                'close': df['收盘'].tolist(),
                'volume': df['成交量'].tolist(),
                'amount': df['成交额'].tolist(),
                'date': df['日期'].astype(str).tolist() if '日期' in df.columns else [],
            }

            return {
                'data': kline_data,
                'source': 'akshare',
                'symbol': ticker,
            }, 'OK'

        except ImportError:
            return {}, 'NO_AKSHARE'
        except Exception as e:
            logger.warning(f"AKShare 获取 K 线失败 {ticker}: {e}")
            return {}, 'ERROR'

    def get_stock_info(self, ticker: str) -> tuple:
        """获取股票基本信息（名称、行业等）"""
        code = _normalize_ticker(ticker)
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        symbol = f"{prefix}{code}"

        try:
            import akshare as ak

            df = ak.stock_individual_info_em(symbol=symbol)

            info = {}
            for _, row in df.iterrows():
                if isinstance(row.get('item'), str) and isinstance(row.get('value'), str):
                    info[row['item']] = row['value']

            return info, 'OK'

        except ImportError:
            return {}, 'NO_AKSHARE'
        except Exception as e:
            logger.warning(f"AKShare 获取股票信息失败 {ticker}: {e}")
            return {}, 'ERROR'

    def get_news(self, ticker: str, query: str = None, days: int = 7) -> tuple:
        """
        获取新闻数据（东方财富）
        """
        code = _normalize_ticker(ticker)
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        symbol = f"{prefix}{code}"

        try:
            import akshare as ak

            news_list = []
            # 尝试获取公司个股新闻
            try:
                df = ak.stock_news_em(symbol=symbol)
                if df is not None and not df.empty:
                    for _, row in df.head(10).iterrows():
                        news_list.append({
                            'title': str(row.get('新闻标题', '')),
                            'content': str(row.get('新闻内容', '')),
                            'date': str(row.get('发布时间', '')),
                            'source': str(row.get('新闻来源', '')),
                        })
            except:
                pass

            return {'news': news_list}, 'OK' if news_list else 'OK_EMPTY'

        except ImportError:
            return {}, 'NO_AKSHARE'
        except Exception as e:
            logger.warning(f"AKShare 获取新闻失败 {ticker}: {e}")
            return {}, 'ERROR'


# ============================================================
# 3. 统一数据获取层（mootdx → AKShare 回退）
# ============================================================

class DataLayer:
    """
    统一数据获取层

    优先级：mootdx (astock a_stock.py) → AKShare 直连 → 返回错误

    使用方式：
        layer = DataLayer()
        data, status = layer.get_kline('600519')
    """

    def __init__(self, use_akshare_fallback: bool = True):
        self.use_akshare_fallback = use_akshare_fallback
        self._akshare = AKShareSource() if use_akshare_fallback else None

    def get_kline(
        self,
        ticker: str,
        start_date: str = "20230101",
        end_date: str = "20261231",
        period: str = "daily",
    ) -> tuple:
        """
        获取 K 线数据（自动回退）

        Returns:
            (data_dict, status_str)
            status: 'OK_mootdx' / 'OK_akshare' / 'ERROR'
        """
        code = _normalize_ticker(ticker)

        # 1. 尝试 astock 的 a_stock.py
        try:
            from tradingagents.dataflows.a_stock import get_stock_data
            a_stock_result = get_stock_data(ticker)
            if a_stock_result and a_stock_result.get('data'):
                return a_stock_result, 'OK_mootdx'
        except Exception as e:
            logger.debug(f"mootdx 获取 K 线失败 {code}: {e}")

        # 2. 回退到 AKShare
        if self.use_akshare_fallback and self._akshare:
            data, status = self._akshare.get_kline(ticker, start_date, end_date, period)
            if status == 'OK':
                return data, 'OK_akshare'

        return {}, 'ERROR'

    def get_stock_info(self, ticker: str) -> tuple:
        """获取股票基本信息（自动回退）"""
        # 尝试 astock 的数据
        try:
            from tradingagents.dataflows.a_stock import get_fundamentals
            info = get_fundamentals(ticker)
            if info:
                return info, 'OK_mootdx'
        except Exception:
            pass

        # 回退到 AKShare
        if self.use_akshare_fallback and self._akshare:
            data, status = self._akshare.get_stock_info(ticker)
            if status == 'OK':
                return data, 'OK_akshare'

        return {}, 'ERROR'

    def get_news(self, ticker: str, query: str = None, days: int = 7) -> tuple:
        """获取新闻数据（自动回退）"""
        # 尝试 astock
        try:
            from tradingagents.dataflows.a_stock import get_news
            news_result = get_news(ticker, start_date='', end_date='')
            if news_result:
                return news_result, 'OK_mootdx'
        except Exception:
            pass

        # 回退到 AKShare
        if self.use_akshare_fallback and self._akshare:
            data, status = self._akshare.get_news(ticker, query, days)
            if status in ('OK', 'OK_EMPTY'):
                return data, 'OK_akshare'

        return {}, 'ERROR'

    def get_realtime_price(self, ticker: str) -> dict:
        """获取实时行情"""
        return get_realtime_price(ticker)


# ============================================================
# 4. 数据质量检查工具
# ============================================================

def check_data_quality(data: dict, source: str = 'unknown') -> dict:
    """
    数据质量检查

    Args:
        data: K 线数据 dict
        source: 数据来源

    Returns:
        dict: {
            'quality_score': float (0-1),
            'checks': list,
            'status': str
        }
    """
    checks = []
    score = 1.0

    raw_data = data.get('data', {})
    closes = raw_data.get('close', [])
    volumes = raw_data.get('volume', [])

    # 检查 1：数据量是否充足
    n_points = len(closes)
    if n_points < 10:
        score -= 0.5
        checks.append(f'❌ 数据量不足: 仅 {n_points} 条')
    elif n_points < 30:
        score -= 0.1
        checks.append(f'⚠️ 数据量偏少: {n_points} 条')
    else:
        checks.append(f'✅ 数据量充足: {n_points} 条')

    # 检查 2：是否有缺失值
    if closes and any(c is None or c <= 0 for c in closes):
        score -= 0.2
        checks.append('❌ 存在缺失或无效的价格数据')
    else:
        checks.append('✅ 价格数据完整')

    # 检查 3：成交量数据
    if volumes and all(v > 0 for v in volumes):
        checks.append('✅ 成交量数据有效')
    else:
        score -= 0.1
        checks.append('⚠️ 成交量数据可能有问题')

    # 检查 4：数据时间范围
    dates = raw_data.get('date', [])
    if dates:
        checks.append(f'✅ 数据日期范围: {dates[0]} ~ {dates[-1]}')

    status = 'GOOD' if score >= 0.8 else 'WARNING' if score >= 0.5 else 'POOR'

    return {
        'quality_score': round(score, 2),
        'checks': checks,
        'status': status,
        'source': source,
    }


# ============================================================
# 5. 便捷函数
# ============================================================

def get_data_with_fallback(
    ticker: str,
    data_type: str = 'kline',
    start_date: str = "20230101",
    end_date: str = "20261231",
    period: str = "daily",
) -> tuple:
    """
    统一数据获取入口（带回退机制）

    Args:
        ticker: 股票代码
        data_type: 'kline' / 'info' / 'news'
        start_date / end_date: 时间范围
        period: 'daily' / 'weekly' / 'monthly'

    Returns:
        (data, status, quality)
    """
    layer = DataLayer()

    if data_type == 'kline':
        data, status = layer.get_kline(ticker, start_date, end_date, period)
    elif data_type == 'info':
        data, status = layer.get_stock_info(ticker)
    elif data_type == 'news':
        data, status = layer.get_news(ticker)
    else:
        return {}, 'UNKNOWN_TYPE', None

    quality = check_data_quality(data, source=status) if data else None

    return data, status, quality


if __name__ == "__main__":
    # 测试
    print("===== 实时行情测试 =====")
    result = get_realtime_price("600519")
    print(f"贵州茅台实时: {result}")

    print("\n===== 数据层回退测试 =====")
    layer = DataLayer()
    data, status = layer.get_kline("600519")
    print(f"状态: {status}, 数据点: {len(data.get('data', {}).get('close', []))}")

    print("\n===== 数据质量检查 =====")
    quality = check_data_quality(data, source=status)
    print(f"质量分: {quality['quality_score']}, 状态: {quality['status']}")
    for c in quality['checks']:
        print(f"  {c}")