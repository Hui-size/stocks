import akshare as ak
import pandas as pd

from utils import normalize_columns, safe_float


class SectorDataError(Exception):
    """行业板块数据获取失败时抛出的业务异常。"""


SECTOR_COLUMNS = {
    "板块代码": "code",
    "板块名称": "name",
    "最新价": "price",
    "涨跌幅": "pct_chg",
    "换手率": "turnover",
    "上涨家数": "up_count",
    "下跌家数": "down_count",
    "领涨股票": "leading_stock",
    "领涨股票-涨跌幅": "leading_stock_pct",
    "成交额": "amount",
}


def fetch_industry_board() -> pd.DataFrame:
    """第一优先级使用东方财富接口获取 A 股行业板块列表。"""
    try:
        raw_df = ak.stock_board_industry_name_em()
    except Exception as exc:
        raise SectorDataError("东方财富行业板块接口暂时不可用，请稍后重试或检查网络代理。") from exc

    if raw_df is None or raw_df.empty:
        raise SectorDataError("东方财富行业板块接口返回为空，请稍后重试。")

    df = normalize_columns(raw_df, SECTOR_COLUMNS)
    if "name" not in df.columns:
        raise SectorDataError("行业板块数据缺少“板块名称”字段，可能是接口字段发生变化。")

    for col in ["price", "pct_chg", "turnover", "up_count", "down_count", "leading_stock_pct", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ordered_cols = [col for col in ["code", "name", "price", "pct_chg", "turnover", "up_count", "down_count", "leading_stock", "leading_stock_pct", "amount"] if col in df.columns]
    return df[ordered_cols].copy()


def get_sector_rankings(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """生成今日涨幅、跌幅和成交额排行榜。"""
    if df is None or df.empty:
        empty = pd.DataFrame()
        return {"top_gain": empty, "top_loss": empty, "top_amount": empty}

    top_gain = df.sort_values("pct_chg", ascending=False).head(10) if "pct_chg" in df.columns else df.head(0)
    top_loss = df.sort_values("pct_chg", ascending=True).head(10) if "pct_chg" in df.columns else df.head(0)
    top_amount = df.sort_values("amount", ascending=False).head(10) if "amount" in df.columns else df.head(0)
    return {"top_gain": top_gain, "top_loss": top_loss, "top_amount": top_amount}


def fetch_sector_stocks(sector_name: str) -> pd.DataFrame:
    """使用东方财富接口获取行业板块成分股。"""
    if not sector_name:
        raise SectorDataError("请选择一个行业板块。")
    try:
        raw_df = ak.stock_board_industry_cons_em(symbol=sector_name)
    except Exception as exc:
        raise SectorDataError("板块成分股接口暂时不可用，请稍后重试。") from exc
    if raw_df is None or raw_df.empty:
        raise SectorDataError("板块成分股接口返回为空。")

    df = normalize_columns(
        raw_df,
        {
            "代码": "code",
            "名称": "name",
            "最新价": "price",
            "涨跌幅": "pct_chg",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
        },
    )
    return df


def analyze_sector(row: pd.Series | dict) -> str:
    """根据行业板块涨跌、成交额和上涨下跌家数生成客观分析。"""
    pct_chg = safe_float(row.get("pct_chg"))
    amount = safe_float(row.get("amount"))
    up_count = safe_float(row.get("up_count"))
    down_count = safe_float(row.get("down_count"))
    leading_stock = row.get("leading_stock", "暂无")
    leading_stock_pct = safe_float(row.get("leading_stock_pct"))

    parts = []
    if pct_chg is None:
        parts.append("当前板块涨跌幅数据不足，强弱暂无法判断。")
    elif pct_chg >= 2:
        parts.append("当前板块表现偏强。")
    elif pct_chg <= -2:
        parts.append("当前板块表现偏弱。")
    else:
        parts.append("当前板块整体偏震荡。")

    if amount is not None:
        if amount >= 10_000_000_000:
            parts.append("成交额处于较高水平，市场关注度较高。")
        else:
            parts.append("成交额未见明显异常放大。")

    if up_count is not None and down_count is not None:
        if up_count > down_count * 1.5:
            parts.append("板块内上涨家数明显占优。")
        elif down_count > up_count * 1.5:
            parts.append("板块内下跌家数明显占优。")
        else:
            parts.append("板块内部涨跌分化较明显。")

    if leading_stock != "暂无":
        leader_text = f"领涨股票为 {leading_stock}"
        if leading_stock_pct is not None:
            leader_text += f"，涨跌幅 {leading_stock_pct:.2f}%"
        parts.append(leader_text + "。")

    parts.append("风险提示：板块热度可能快速切换，需关注成交额持续性和板块内部个股分化。")
    return " ".join(parts)
