import pandas as pd
import requests

from data_fetch import code_with_market_prefix, normalize_stock_code


class RealtimeQuoteError(Exception):
    """实时行情获取失败时抛出的业务异常。"""


def fetch_realtime_quote(stock_code: str) -> dict:
    """通过腾讯单股票接口获取实时快照，避免下载整个 A 股市场。"""
    code = normalize_stock_code(stock_code)
    symbol = code_with_market_prefix(code)
    try:
        response = requests.get(
            f"https://qt.gtimg.cn/q={symbol}",
            timeout=4,
        )
        response.raise_for_status()
        fields = response.content.decode("gbk", errors="replace").split("~")
    except Exception as exc:
        raise RealtimeQuoteError("实时行情接口暂时不可用，请稍后重试。") from exc

    if len(fields) < 38 or fields[3] in (None, "", "-"):
        raise RealtimeQuoteError("实时行情接口返回为空。")

    timestamp = fields[30][8:14]
    timestamp = f"{timestamp[:2]}:{timestamp[2:4]}:{timestamp[4:]}" if len(timestamp) == 6 else ""
    quote = {
        "code": code,
        "symbol": symbol,
        "name": fields[1] or code,
        "latest_price": pd.to_numeric(fields[3], errors="coerce"),
        "change": pd.to_numeric(fields[31], errors="coerce"),
        "pct_chg": pd.to_numeric(fields[32], errors="coerce"),
        "bid": pd.to_numeric(fields[9], errors="coerce"),
        "ask": pd.to_numeric(fields[19], errors="coerce"),
        "prev_close": pd.to_numeric(fields[4], errors="coerce"),
        "open": pd.to_numeric(fields[5], errors="coerce"),
        "high": pd.to_numeric(fields[33], errors="coerce"),
        "low": pd.to_numeric(fields[34], errors="coerce"),
        "volume": pd.to_numeric(fields[36], errors="coerce"),
        "amount": pd.to_numeric(fields[37], errors="coerce") * 10_000,
        "timestamp": timestamp,
        "source": "腾讯单股票实时行情",
    }
    return quote


def fetch_realtime_minute(stock_code: str, period: str = "1") -> pd.DataFrame:
    """获取最新交易日分时走势数据。"""
    code = normalize_stock_code(stock_code)
    symbol = code_with_market_prefix(code)
    try:
        response = requests.get(
            "https://web.ifzq.gtimg.cn/appstock/app/minute/query",
            params={"code": symbol},
            timeout=6,
        )
        response.raise_for_status()
        payload = response.json().get("data", {}).get(symbol, {}).get("data", {})
        trade_date = str(payload.get("date") or "")
        rows = [str(row).split()[:4] for row in payload.get("data", [])]
    except Exception as exc:
        raise RealtimeQuoteError("实时分时走势接口暂时不可用，请稍后重试。") from exc

    if not rows or len(trade_date) != 8:
        raise RealtimeQuoteError("实时分时走势接口返回为空。")

    df = pd.DataFrame(rows, columns=["time", "close", "volume_total", "amount_total"])
    df["datetime"] = pd.to_datetime(trade_date + df.pop("time"), format="%Y%m%d%H%M", errors="coerce")
    for col in ["close", "volume_total", "amount_total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open"] = df["close"].iloc[0]
    df["high"] = df["close"].cummax()
    df["low"] = df["close"].cummin()
    df["volume"] = df["volume_total"].diff().fillna(df["volume_total"]).clip(lower=0)
    df["amount"] = df["amount_total"].diff().fillna(df["amount_total"]).clip(lower=0)
    required = ["datetime", "open", "high", "low", "close", "volume"]
    df = df.dropna(subset=required).sort_values("datetime").reset_index(drop=True)
    if df.empty:
        raise RealtimeQuoteError("实时分时走势清洗后为空。")
    return df[["datetime", "open", "high", "low", "close", "volume", "amount"]]


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
