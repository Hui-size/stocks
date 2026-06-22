from datetime import date

import akshare as ak
import pandas as pd

from data_fetch import normalize_stock_code
from utils import normalize_columns


POSITIVE_WORDS = ["增长", "预增", "中标", "回购", "增持", "突破", "合作", "盈利", "利好"]
NEGATIVE_WORDS = ["亏损", "预减", "减持", "处罚", "诉讼", "下滑", "风险", "立案", "利空"]


def classify_news_title(title: str) -> str:
    """根据标题关键词对新闻做简单归类。"""
    text = str(title)
    if any(word in text for word in POSITIVE_WORDS):
        return "利好相关"
    if any(word in text for word in NEGATIVE_WORDS):
        return "利空相关"
    if text.strip():
        return "中性信息"
    return "无法判断"


def fetch_stock_news(stock_code: str) -> pd.DataFrame:
    """获取个股相关新闻。"""
    code = normalize_stock_code(stock_code)
    raw_df = ak.stock_news_em(symbol=code)
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    df = normalize_columns(raw_df, {"关键词": "keyword", "新闻标题": "title", "新闻内容": "content", "发布时间": "time", "文章来源": "source", "新闻链接": "url"})
    if "title" not in df.columns:
        return pd.DataFrame()
    df["分类"] = df["title"].apply(classify_news_title)
    return df


def fetch_stock_announcements(stock_code: str) -> pd.DataFrame:
    """尝试获取股票公告，接口不稳定时返回空表。"""
    code = normalize_stock_code(stock_code)
    try:
        raw_df = ak.stock_notice_report(symbol=code, date=date.today().strftime("%Y%m%d"))
    except Exception:
        return pd.DataFrame()
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    return normalize_columns(raw_df, {"代码": "code", "名称": "name", "公告标题": "title", "公告时间": "time", "网址": "url"})
