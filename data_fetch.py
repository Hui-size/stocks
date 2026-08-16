import time
from datetime import date, timedelta

import akshare as ak
import pandas as pd
import requests


class DataFetchError(Exception):
    """数据获取失败时抛出的中文业务异常。"""


_EM_HIST_RETRY_AFTER = 0.0
_EM_HIST_COOLDOWN_SECONDS = 600


def normalize_stock_code(stock_code: str) -> str:
    """清洗用户输入的股票代码，只保留 6 位数字代码。"""
    code = str(stock_code).strip().lower().replace("sh", "").replace("sz", "")
    if not (code.isdigit() and len(code) == 6):
        raise DataFetchError("股票代码格式不正确，请输入 6 位 A 股代码，例如 600519、000001、300750。")
    return code


def code_with_market_prefix(stock_code: str) -> str:
    """根据股票代码规则补充新浪接口需要的市场前缀。"""
    code = normalize_stock_code(stock_code)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def get_default_date_range(days: int = 365) -> tuple[str, str]:
    """生成默认查询日期范围，默认取最近一年自然日。"""
    end = date.today()
    start = end - timedelta(days=days)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _standardize_hist_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """统一 AKShare 不同接口返回的字段名称和数据类型。"""
    if raw_df is None or raw_df.empty:
        raise DataFetchError("接口返回为空，可能是代码不存在、停牌时间过长或数据源暂时不可用。")

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "振幅": "amplitude",
        "换手率": "turnover",
    }

    df = raw_df.rename(columns=rename_map).copy()
    required_columns = ["date", "open", "close", "high", "low", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataFetchError(f"行情数据缺少必要字段：{', '.join(missing_columns)}。")

    df = df[required_columns + [col for col in ["pct_chg", "change", "amount", "turnover"] if col in df.columns]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in [c for c in df.columns if c != "date"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "close", "high", "low"]).sort_values("date")
    if df.empty:
        raise DataFetchError("行情数据清洗后为空，请稍后重试或换一个股票代码。")
    return df.reset_index(drop=True)


def fetch_hist_from_tencent(stock_code: str, days: int) -> tuple[pd.DataFrame, str | None]:
    """通过腾讯单股票接口获取日 K，避免无超时的全量行情请求。"""
    code = normalize_stock_code(stock_code)
    symbol = code_with_market_prefix(code)
    response = requests.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
        params={"param": f"{symbol},day,,,{max(int(days), 30)},qfq"},
        timeout=6,
    )
    response.raise_for_status()
    payload = response.json().get("data", {}).get(symbol, {})
    rows = payload.get("qfqday") or payload.get("day") or []
    if not rows:
        raise DataFetchError("腾讯单股票历史行情接口返回为空。")

    raw_df = pd.DataFrame(
        [row[:6] for row in rows],
        columns=["date", "open", "close", "high", "low", "volume"],
    )
    quote_fields = payload.get("qt", {}).get(symbol) or []
    name = quote_fields[1] if len(quote_fields) > 1 else None
    return _standardize_hist_data(raw_df), name


def fetch_hist_from_sina(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """使用新浪历史行情接口作为备用方案获取 A 股日 K 数据。"""
    symbol = code_with_market_prefix(stock_code)
    raw_df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, end_date=end_date, adjust="")
    if raw_df is None or raw_df.empty:
        raise DataFetchError("备用接口也没有返回历史行情数据。")

    df = raw_df.reset_index().rename(
        columns={
            "date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
        }
    )
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce") / 100
    return _standardize_hist_data(df)


def fetch_stock_history(stock_code: str, days: int = 365) -> tuple[pd.DataFrame, list[str]]:
    """按主接口、备用接口顺序获取股票历史行情，并返回尝试过程中的提示。"""
    global _EM_HIST_RETRY_AFTER
    start_date, end_date = get_default_date_range(days)
    errors = []
    warnings = []

    if time.monotonic() >= _EM_HIST_RETRY_AFTER:
        try:
            history, stock_name = fetch_hist_from_tencent(stock_code, days)
            if stock_name:
                history.attrs["stock_name"] = stock_name
            return history, warnings
        except Exception as exc:
            _EM_HIST_RETRY_AFTER = time.monotonic() + _EM_HIST_COOLDOWN_SECONDS
            errors.append(f"主接口腾讯单股票日 K 获取失败：{exc}")
    else:
        warnings.append("腾讯历史行情接口近期连接失败，本次已直接使用备用接口。")

    try:
        if not warnings:
            warnings.append("腾讯历史行情接口当前连接不稳定，已自动切换到备用接口，历史行情数据已正常加载。")
        return fetch_hist_from_sina(stock_code, start_date, end_date), warnings
    except Exception as exc:
        errors.append(f"备用接口 stock_zh_a_daily 获取失败：{exc}")

    raise DataFetchError("无法获取该股票历史行情。\n" + "\n".join(errors))


def fetch_stock_basic_info(stock_code: str, hist_df: pd.DataFrame | None = None) -> dict:
    """获取股票名称、最新价和涨跌幅，失败时回退到历史收盘价。"""
    code = normalize_stock_code(stock_code)
    fallback_close = None
    fallback_pct = None
    if hist_df is not None and not hist_df.empty:
        fallback_close = float(hist_df.iloc[-1]["close"])
        if len(hist_df) >= 2:
            prev_close = float(hist_df.iloc[-2]["close"])
            fallback_pct = (fallback_close - prev_close) / prev_close * 100 if prev_close else None

    info = {
        "code": code,
        "name": str(hist_df.attrs.get("stock_name") or code) if hist_df is not None else code,
        "latest_price": fallback_close,
        "pct_chg": fallback_pct,
        "source": "历史收盘价",
    }

    return info
