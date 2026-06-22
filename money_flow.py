import akshare as ak
import pandas as pd

from data_fetch import code_with_market_prefix, normalize_stock_code
from utils import normalize_columns, safe_float


def _market_code(stock_code: str) -> str:
    """生成 AKShare 资金流接口需要的市场代码。"""
    prefixed = code_with_market_prefix(stock_code)
    return "sh" if prefixed.startswith("sh") else "sz"


def fetch_individual_money_flow(stock_code: str) -> pd.DataFrame:
    """获取个股资金流历史数据。"""
    code = normalize_stock_code(stock_code)
    raw_df = ak.stock_individual_fund_flow(stock=code, market=_market_code(code))
    if raw_df is None or raw_df.empty:
        raise ValueError("资金流接口返回为空。")
    df = normalize_columns(
        raw_df,
        {
            "日期": "date",
            "收盘价": "close",
            "涨跌幅": "pct_chg",
            "主力净流入-净额": "main_net",
            "超大单净流入-净额": "super_net",
            "大单净流入-净额": "big_net",
            "中单净流入-净额": "medium_net",
            "小单净流入-净额": "small_net",
        },
    )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in [c for c in df.columns if c != "date"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True) if "date" in df.columns else df


def summarize_money_flow(flow_df: pd.DataFrame) -> dict:
    """提取最近一日资金流摘要。"""
    if flow_df is None or flow_df.empty:
        return {}
    latest = flow_df.iloc[-1]
    return {
        "主力净流入": safe_float(latest.get("main_net")),
        "超大单净流入": safe_float(latest.get("super_net")),
        "大单净流入": safe_float(latest.get("big_net")),
        "中单净流入": safe_float(latest.get("medium_net")),
        "小单净流入": safe_float(latest.get("small_net")),
    }


def analyze_money_flow(flow_df: pd.DataFrame, price_df: pd.DataFrame | None = None) -> str:
    """根据资金流和价格量能关系生成客观分析。"""
    if flow_df is None or flow_df.empty:
        return "资金流数据暂不可用。"
    latest = flow_df.iloc[-1]
    main_net = safe_float(latest.get("main_net"), 0)
    recent_main = flow_df["main_net"].tail(3) if "main_net" in flow_df.columns else pd.Series(dtype=float)
    parts = []
    if len(recent_main) >= 3 and (recent_main > 0).all():
        parts.append("主力资金近 3 个交易日连续净流入，资金关注度较高。")
    elif main_net > 0:
        parts.append("最近一个交易日主力资金为净流入。")
    elif main_net < 0:
        parts.append("最近一个交易日主力资金为净流出。")
    else:
        parts.append("最近一个交易日主力资金流向不明显。")

    if price_df is not None and len(price_df) >= 20:
        latest_price = price_df.iloc[-1]
        avg_volume = price_df["volume"].tail(20).mean()
        price_up = latest_price["close"] > latest_price["open"]
        volume_ratio = latest_price["volume"] / avg_volume if avg_volume else 1
        if main_net < 0 and price_up:
            parts.append("主力资金流出但股价上涨，资金与价格表现存在分歧。")
        if main_net < 0 and not price_up and volume_ratio > 1.3:
            parts.append("放量下跌且主力资金流出，抛压较大。")
        if main_net >= 0 and price_up and volume_ratio < 0.8:
            parts.append("缩量上涨，持续性仍需观察。")
    parts.append("资金流只能反映阶段性资金行为，不构成买卖建议。")
    return " ".join(parts)
