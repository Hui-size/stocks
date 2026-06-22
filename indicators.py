import pandas as pd


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MA5、MA10、MA20、MA60 均线指标。"""
    result = df.copy()
    for window in [5, 10, 20, 60]:
        result[f"MA{window}"] = result["close"].rolling(window=window).mean()
    return result


def add_volume_averages(df: pd.DataFrame) -> pd.DataFrame:
    """计算成交量 5 日和 10 日均量，成交量单位保持为手。"""
    result = df.copy()
    result["VOL_MA5"] = result["volume"].rolling(window=5).mean()
    result["VOL_MA10"] = result["volume"].rolling(window=10).mean()
    return result


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MACD 指标，包括 DIF、DEA 和 MACD 柱。"""
    result = df.copy()
    ema12 = result["close"].ewm(span=12, adjust=False).mean()
    ema26 = result["close"].ewm(span=26, adjust=False).mean()
    result["DIF"] = ema12 - ema26
    result["DEA"] = result["DIF"].ewm(span=9, adjust=False).mean()
    result["MACD"] = (result["DIF"] - result["DEA"]) * 2
    return result


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 RSI 相对强弱指标。"""
    result = df.copy()
    delta = result["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    result["RSI"] = rsi
    return result


def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """计算布林线 BOLL 中轨、上轨和下轨。"""
    result = df.copy()
    middle = result["close"].rolling(window=window).mean()
    std = result["close"].rolling(window=window).std()
    result["BOLL_MID"] = middle
    result["BOLL_UPPER"] = middle + 2 * std
    result["BOLL_LOWER"] = middle - 2 * std
    return result


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算页面需要展示和分析的全部技术指标。"""
    result = add_moving_averages(df)
    result = add_volume_averages(result)
    result = add_macd(result)
    result = add_rsi(result)
    result = add_bollinger_bands(result)
    return result
