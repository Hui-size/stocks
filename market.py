from datetime import date, timedelta

import akshare as ak
import pandas as pd

from analysis import judge_trend
from indicators import add_moving_averages
from utils import first_existing_column, normalize_columns, safe_float


INDEX_SYMBOLS = {
    "上证指数": {"hist": "000001", "daily": "sh000001"},
    "深证成指": {"hist": "399001", "daily": "sz399001"},
    "创业板指": {"hist": "399006", "daily": "sz399006"},
    "科创50": {"hist": "000688", "daily": "sh000688"},
}


def _date_range(days: int = 90) -> tuple[str, str]:
    """生成指数查询日期范围。"""
    end = date.today()
    start = end - timedelta(days=days)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _standardize_index_hist(raw_df: pd.DataFrame) -> pd.DataFrame:
    """统一指数历史行情字段。"""
    if raw_df is None or raw_df.empty:
        raise ValueError("指数历史数据为空。")
    df = normalize_columns(
        raw_df,
        {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
        },
    )
    required = ["date", "open", "close", "high", "low"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"指数数据缺少字段：{', '.join(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in [c for c in df.columns if c != "date"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def fetch_index_history(symbol: str, days: int = 90) -> pd.DataFrame:
    """获取单个指数历史行情。"""
    start_date, end_date = _date_range(days)
    raw_df = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date)
    return _standardize_index_hist(raw_df)


def fetch_index_history_fallback(symbol: str, days: int = 90) -> pd.DataFrame:
    """使用新浪指数日线接口作为备用方案。"""
    start_date, end_date = _date_range(days)
    raw_df = ak.stock_zh_index_daily(symbol=symbol)
    df = _standardize_index_hist(raw_df)
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].reset_index(drop=True)


def get_major_indices(days: int = 90) -> tuple[list[dict], list[str]]:
    """获取主要指数概览数据和失败提示。"""
    results = []
    errors = []
    for name, symbols in INDEX_SYMBOLS.items():
        try:
            try:
                df = fetch_index_history(symbols["hist"], days)
                symbol = symbols["hist"]
            except Exception:
                df = fetch_index_history_fallback(symbols["daily"], days)
                symbol = symbols["hist"]
            if len(df) < 2:
                raise ValueError("指数数据不足。")
            df_with_ma = add_moving_averages(df.assign(volume=df.get("volume", 0)))
            latest = df_with_ma.iloc[-1]
            previous = df_with_ma.iloc[-2]
            pct_chg = latest.get("pct_chg")
            if pd.isna(pct_chg):
                pct_chg = (latest["close"] / previous["close"] - 1) * 100
            results.append(
                {
                    "name": name,
                    "symbol": symbol,
                    "latest": float(latest["close"]),
                    "pct_chg": float(pct_chg),
                    "trend": judge_trend(df_with_ma),
                    "history": df,
                }
            )
        except Exception as exc:
            errors.append(f"{name} 获取失败：{exc}")
    return results, errors


def fetch_market_activity() -> tuple[dict, str | None]:
    """获取市场上涨下跌家数等情绪数据。"""
    try:
        raw_df = ak.stock_market_activity_legu()
        if raw_df is None or raw_df.empty:
            return {}, "市场活跃度接口返回为空。"
        latest = raw_df.iloc[-1]
        up_col = first_existing_column(raw_df, ["上涨", "上涨家数", "rise", "up"])
        down_col = first_existing_column(raw_df, ["下跌", "下跌家数", "fall", "down"])
        return {
            "raw": raw_df,
            "up_count": safe_float(latest.get(up_col)) if up_col else None,
            "down_count": safe_float(latest.get(down_col)) if down_col else None,
        }, None
    except Exception as exc:
        return {}, f"市场活跃度数据暂不可用：{exc}"


def describe_market_sentiment(indices: list[dict], activity: dict | None = None) -> str:
    """根据指数涨跌和市场活跃度生成市场情绪描述。"""
    if not indices:
        return "市场情绪暂无法判断，主要指数数据不可用。"
    pct_values = [item["pct_chg"] for item in indices if item.get("pct_chg") is not None]
    avg_pct = sum(pct_values) / len(pct_values) if pct_values else 0
    strong_count = sum(1 for value in pct_values if value > 0.5)
    weak_count = sum(1 for value in pct_values if value < -0.5)
    up_count = (activity or {}).get("up_count")
    down_count = (activity or {}).get("down_count")

    if up_count and down_count and up_count > down_count * 1.3 and avg_pct > 0:
        return "市场偏强，主要指数和上涨家数共同改善。"
    if up_count and down_count and down_count > up_count * 1.3 and avg_pct < 0:
        return "市场偏弱，下跌家数占优且指数表现承压。"
    if strong_count > 0 and weak_count > 0:
        return "市场情绪分化明显，不同指数表现差异较大。"
    if avg_pct > 0.4:
        return "市场偏强，主要指数整体上涨。"
    if avg_pct < -0.4:
        return "市场偏弱，主要指数整体回落。"
    return "市场震荡，主要指数涨跌幅较小。"
