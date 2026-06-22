import akshare as ak
import pandas as pd

from data_fetch import code_with_market_prefix, normalize_stock_code


class RealtimeQuoteError(Exception):
    """实时行情获取失败时抛出的业务异常。"""


def fetch_realtime_quote(stock_code: str) -> dict:
    """使用 AKShare 新浪实时行情接口获取单只股票快照。"""
    code = normalize_stock_code(stock_code)
    symbol = code_with_market_prefix(code)
    try:
        raw_df = ak.stock_zh_a_spot()
    except Exception as exc:
        raise RealtimeQuoteError("实时行情接口暂时不可用，请稍后重试。") from exc

    if raw_df is None or raw_df.empty:
        raise RealtimeQuoteError("实时行情接口返回为空。")

    row = raw_df[raw_df["代码"].astype(str).str.lower() == symbol.lower()]
    if row.empty:
        raise RealtimeQuoteError(f"没有在实时行情中找到 {code}。")

    item = row.iloc[0]
    quote = {
        "code": code,
        "symbol": symbol,
        "name": str(item.get("名称", code)),
        "latest_price": pd.to_numeric(item.get("最新价"), errors="coerce"),
        "change": pd.to_numeric(item.get("涨跌额"), errors="coerce"),
        "pct_chg": pd.to_numeric(item.get("涨跌幅"), errors="coerce"),
        "bid": pd.to_numeric(item.get("买入"), errors="coerce"),
        "ask": pd.to_numeric(item.get("卖出"), errors="coerce"),
        "prev_close": pd.to_numeric(item.get("昨收"), errors="coerce"),
        "open": pd.to_numeric(item.get("今开"), errors="coerce"),
        "high": pd.to_numeric(item.get("最高"), errors="coerce"),
        "low": pd.to_numeric(item.get("最低"), errors="coerce"),
        "volume": pd.to_numeric(item.get("成交量"), errors="coerce"),
        "amount": pd.to_numeric(item.get("成交额"), errors="coerce"),
        "timestamp": str(item.get("时间戳", "")),
        "source": "新浪实时行情",
    }
    return quote


def fetch_realtime_minute(stock_code: str, period: str = "1") -> pd.DataFrame:
    """获取最新交易日分时走势数据。"""
    code = normalize_stock_code(stock_code)
    symbol = code_with_market_prefix(code)
    try:
        raw_df = ak.stock_zh_a_minute(symbol=symbol, period=period, adjust="")
    except Exception as exc:
        raise RealtimeQuoteError("实时分时走势接口暂时不可用，请稍后重试。") from exc

    if raw_df is None or raw_df.empty:
        raise RealtimeQuoteError("实时分时走势接口返回为空。")

    df = raw_df.rename(
        columns={
            "day": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
    ).copy()
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RealtimeQuoteError(f"实时分时走势缺少字段：{', '.join(missing)}。")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in [c for c in df.columns if c != "datetime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).sort_values("datetime").reset_index(drop=True)
    if df.empty:
        raise RealtimeQuoteError("实时分时走势清洗后为空。")

    latest_date = df["datetime"].dt.date.max()
    return df[df["datetime"].dt.date == latest_date].reset_index(drop=True)


def compare_realtime_with_prediction(quote: dict, prediction_result: dict) -> dict:
    """对比实时盘中表现和 1 日预测方向。"""
    first_prediction = prediction_result["predictions"][0] if prediction_result.get("predictions") else {}
    predicted_direction = first_prediction.get("预测方向", "暂无")
    pct_chg = quote.get("pct_chg")
    if pd.isna(pct_chg):
        live_direction = "暂无"
    elif pct_chg > 1:
        live_direction = "上涨"
    elif pct_chg < -1:
        live_direction = "下跌"
    else:
        live_direction = "震荡"

    if predicted_direction == "暂无" or live_direction == "暂无":
        consistency = "暂无"
    elif predicted_direction == live_direction:
        consistency = "方向一致"
    elif "震荡" in [predicted_direction, live_direction]:
        consistency = "部分接近"
    else:
        consistency = "方向相反"

    return {
        "预测方向": predicted_direction,
        "实时方向": live_direction,
        "对比结论": consistency,
        "说明": "实时行情用于观察盘中表现，预测模型基于最新可用历史日线，两者存在时间粒度差异。",
    }
