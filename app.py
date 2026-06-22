import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

from analysis import build_analysis
from config import load_config, save_config
from data_fetch import DataFetchError, fetch_stock_basic_info, fetch_stock_history, normalize_stock_code
from feedback import apply_feedback_calibration, record_predictions, summarize_feedback, update_prediction_outcomes
from indicators import add_all_indicators
from market import describe_market_sentiment, fetch_market_activity, get_major_indices
from news import fetch_stock_announcements, fetch_stock_news
from prediction import predict_short_term
from realtime import RealtimeQuoteError, compare_realtime_with_prediction, fetch_realtime_minute, fetch_realtime_quote
from report import generate_markdown_report
from scoring import build_score
from sector import SectorDataError, analyze_sector, fetch_industry_board, fetch_sector_stocks, get_sector_rankings
from utils import format_number
from watchlist import add_to_watchlist, load_watchlist, remove_from_watchlist
from backtest import build_backtest_figure, rolling_backtest


def apply_custom_theme() -> None:
    """注入页面样式，优化 Streamlit 默认全白界面。"""
    st.markdown(
        """
<style>
    :root {
        --bg: #f4f6f8;
        --panel: #ffffff;
        --panel-soft: #f8fafc;
        --text: #1f2937;
        --muted: #6b7280;
        --line: #e5e7eb;
        --accent: #2563eb;
        --accent-soft: #dbeafe;
    }

    .stApp {
        background: linear-gradient(180deg, #eef3f8 0%, #f7f9fb 260px, #f4f6f8 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #253043;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] select {
        background: #1f2937 !important;
        border: 1px solid #374151 !important;
        color: #f9fafb !important;
        border-radius: 8px !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 8px 10px;
        margin: 4px 0;
    }

    .block-container {
        max-width: 1360px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        color: #111827;
        font-weight: 800;
        letter-spacing: 0;
        padding-bottom: 0.25rem;
    }

    h2, h3 {
        color: #1f2937;
        letter-spacing: 0;
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.88rem;
    }

    [data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 750;
        font-size: clamp(1.45rem, 2.4vw, 2.3rem);
        line-height: 1.15;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"],
    [data-testid="stPlotlyChart"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }

    .stAlert {
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.35);
    }

    div[data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid #2563eb;
        background: #2563eb;
        color: white;
        font-weight: 650;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #1d4ed8;
        border-color: #1d4ed8;
        color: white;
    }

    .stCaptionContainer,
    caption,
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def style_plotly_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    """统一 Plotly 图表视觉风格。"""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font=dict(color="#1f2937", family="Arial"),
        legend=dict(bgcolor="rgba(255,255,255,0.75)", bordercolor="#e5e7eb", borderwidth=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig


def prepare_chart_data(df):
    """为图表准备交易日等距 X 轴和格式化悬浮文本。"""
    chart_df = df.copy()
    chart_df["date_label"] = chart_df["date"].dt.strftime("%Y-%m-%d")
    chart_df["volume_wan"] = chart_df["volume"] / 10000
    if "VOL_MA5" not in chart_df.columns:
        chart_df["VOL_MA5"] = chart_df["volume"].rolling(window=5).mean()
    if "VOL_MA10" not in chart_df.columns:
        chart_df["VOL_MA10"] = chart_df["volume"].rolling(window=10).mean()
    return chart_df


def parse_stock_codes(text: str) -> tuple[list[str], list[str]]:
    """解析用户输入的多个股票代码，返回合法代码和错误提示。"""
    raw_items = [item.strip() for item in text.replace("，", ",").replace("\n", ",").split(",")]
    codes = []
    errors = []
    for item in raw_items:
        if not item:
            continue
        try:
            code = normalize_stock_code(item)
            if code in codes:
                errors.append(f"{code} 重复输入，已自动忽略。")
                continue
            codes.append(code)
        except DataFetchError as exc:
            errors.append(f"{item}：{exc}")
    return codes, errors


@st.cache_data(ttl=900, show_spinner=False)
def get_stock_dataset(stock_code: str, days: int = 365) -> dict:
    """带缓存获取单只股票行情、指标、基础信息和分析结果。"""
    code = normalize_stock_code(stock_code)
    hist_df, warnings = fetch_stock_history(code, days=days)
    df = add_all_indicators(hist_df)
    basic_info = fetch_stock_basic_info(code, df)
    analysis_result = build_analysis(df, basic_info.get("pct_chg"))
    return {"code": code, "df": df, "info": basic_info, "analysis": analysis_result, "warnings": warnings}


@st.cache_data(ttl=900, show_spinner=False)
def get_market_dataset() -> dict:
    """带缓存获取市场概览数据。"""
    indices, index_errors = get_major_indices(days=90)
    activity, activity_error = fetch_market_activity()
    errors = index_errors + ([activity_error] if activity_error else [])
    return {"indices": indices, "activity": activity, "sentiment": describe_market_sentiment(indices, activity), "errors": errors}


@st.cache_data(ttl=1800, show_spinner=False)
def get_sector_dataset() -> pd.DataFrame:
    """带缓存获取行业板块数据。"""
    return fetch_industry_board()


@st.cache_data(ttl=1800, show_spinner=False)
def get_sector_stocks_dataset(sector_name: str) -> pd.DataFrame:
    """带缓存获取行业板块成分股。"""
    return fetch_sector_stocks(sector_name)


@st.cache_data(ttl=1800, show_spinner=False)
def get_news_dataset(stock_code: str) -> dict:
    """带缓存获取新闻和公告数据。"""
    code = normalize_stock_code(stock_code)
    return {"news": fetch_stock_news(code), "announcements": fetch_stock_announcements(code)}


@st.cache_data(ttl=900, show_spinner=False)
def get_prediction_dataset(stock_code: str, threshold: float) -> dict:
    """带缓存获取 1-3 日短线预测结果。"""
    return predict_short_term(stock_code, threshold=threshold)


@st.cache_data(ttl=1800, show_spinner=False)
def get_backtest_dataset(stock_code: str, threshold: float, test_days: int = 120) -> dict:
    """带缓存获取滚动回测结果。"""
    return rolling_backtest(stock_code, threshold=threshold, test_days=test_days)


@st.cache_data(ttl=30, show_spinner=False)
def get_realtime_quote_dataset(stock_code: str) -> dict:
    """带短缓存获取实时行情快照。"""
    return fetch_realtime_quote(stock_code)


@st.cache_data(ttl=30, show_spinner=False)
def get_realtime_minute_dataset(stock_code: str) -> pd.DataFrame:
    """带短缓存获取实时分时走势。"""
    return fetch_realtime_minute(stock_code)


def build_price_volume_figure(df, show_boll: bool = True):
    """构建包含 K 线、均线、布林线和成交量的 Plotly 图表。"""
    chart_df = prepare_chart_data(df)
    x = chart_df["date_label"]
    volume_max = chart_df["volume"].max()
    volume_range_top = volume_max * 1.2 if pd.notna(volume_max) and volume_max > 0 else 1

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
        subplot_titles=("K 线与技术指标", "成交量"),
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="K线",
            increasing_line_color="#ef5350",
            increasing_fillcolor="#ef5350",
            decreasing_line_color="#26a69a",
            decreasing_fillcolor="#26a69a",
            hovertext=(
                "日期：" + chart_df["date_label"]
                + "<br>开盘：" + chart_df["open"].round(2).astype(str)
                + "<br>最高：" + chart_df["high"].round(2).astype(str)
                + "<br>最低：" + chart_df["low"].round(2).astype(str)
                + "<br>收盘：" + chart_df["close"].round(2).astype(str)
                + "<br>成交量：" + chart_df["volume_wan"].round(2).astype(str) + " 万手"
            ),
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )

    ma_styles = {"MA5": "#42a5f5", "MA10": "#ff5252", "MA20": "#ff9e9e", "MA60": "#26a69a"}
    for ma, color in ma_styles.items():
        fig.add_trace(go.Scatter(x=x, y=chart_df[ma], mode="lines", name=ma, line=dict(color=color, width=1.5)), row=1, col=1)

    if show_boll:
        boll_styles = {
            "BOLL_UPPER": ("BOLL上轨", "#66bb6a"),
            "BOLL_MID": ("BOLL中轨", "#ff9800"),
            "BOLL_LOWER": ("BOLL下轨", "#ff9800"),
        }
        for col, (name, color) in boll_styles.items():
            fig.add_trace(go.Scatter(x=x, y=chart_df[col], mode="lines", name=name, line=dict(color=color, dash="dot", width=1.4)), row=1, col=1)

    volume_colors = ["#ef5350" if close >= open_ else "#26a69a" for open_, close in zip(chart_df["open"], chart_df["close"])]
    fig.add_trace(
        go.Bar(
            x=x,
            y=chart_df["volume"],
            name="成交量",
            marker_color=volume_colors,
            hovertemplate="日期：%{x}<br>成交量：%{customdata:.2f} 万手<extra></extra>",
            customdata=chart_df["volume_wan"],
        ),
        row=2,
        col=1,
    )
    fig.add_trace(go.Scatter(x=x, y=chart_df["VOL_MA5"], mode="lines", name="VOL_MA5", line=dict(color="#42a5f5", width=1.3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=chart_df["VOL_MA10"], mode="lines", name="VOL_MA10", line=dict(color="#ff9800", width=1.3)), row=2, col=1)

    fig.update_layout(
        height=760,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        dragmode="zoom",
        bargap=0.18,
        modebar=dict(orientation="h"),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量（手）", range=[0, volume_range_top], rangemode="tozero", row=2, col=1)
    fig.update_xaxes(type="category", nticks=12, row=1, col=1)
    fig.update_xaxes(type="category", nticks=12, rangeslider=dict(visible=True, thickness=0.08), row=2, col=1)
    return style_plotly_figure(fig)


def build_macd_figure(df):
    """构建 MACD 指标图表。"""
    chart_df = prepare_chart_data(df)
    colors = ["#ef5350" if value >= 0 else "#26a69a" for value in chart_df["MACD"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart_df["date_label"], y=chart_df["MACD"], name="MACD", marker_color=colors))
    fig.add_trace(go.Scatter(x=chart_df["date_label"], y=chart_df["DIF"], mode="lines", name="DIF"))
    fig.add_trace(go.Scatter(x=chart_df["date_label"], y=chart_df["DEA"], mode="lines", name="DEA"))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h"), bargap=0.18)
    fig.update_xaxes(type="category", nticks=8)
    return style_plotly_figure(fig)


def build_rsi_figure(df):
    """构建 RSI 指标图表。"""
    chart_df = prepare_chart_data(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["date_label"], y=chart_df["RSI"], mode="lines", name="RSI", line=dict(color="#5c6bc0")))
    fig.add_hline(y=70, line_dash="dot", line_color="#ef5350")
    fig.add_hline(y=30, line_dash="dot", line_color="#26a69a")
    fig.update_layout(height=260, yaxis=dict(range=[0, 100]), margin=dict(l=20, r=20, t=30, b=20))
    fig.update_xaxes(type="category", nticks=6)
    return style_plotly_figure(fig)


def build_compare_figure(series_map: dict[str, pd.DataFrame], period: int) -> go.Figure:
    """构建多股票区间涨跌幅对比图。"""
    fig = go.Figure()
    for label, df in series_map.items():
        period_df = df.tail(period).copy()
        if period_df.empty:
            continue
        base_close = period_df.iloc[0]["close"]
        if not base_close:
            continue
        period_df["return_pct"] = (period_df["close"] / base_close - 1) * 100
        period_df["date_label"] = period_df["date"].dt.strftime("%Y-%m-%d")
        fig.add_trace(go.Scatter(x=period_df["date_label"], y=period_df["return_pct"], mode="lines", name=label))
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    fig.update_layout(
        height=520,
        yaxis_title="区间涨跌幅（%）",
        xaxis_title="交易日",
        hovermode="x unified",
        dragmode="zoom",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(type="category", nticks=12, rangeslider=dict(visible=True, thickness=0.08))
    return style_plotly_figure(fig)


def build_index_trend_figure(indices: list[dict]) -> go.Figure:
    """构建主要指数最近 30 日走势对比图。"""
    fig = go.Figure()
    for item in indices:
        df = item["history"].tail(30).copy()
        if df.empty:
            continue
        base = df.iloc[0]["close"]
        if not base:
            continue
        df["return_pct"] = (df["close"] / base - 1) * 100
        df["date_label"] = df["date"].dt.strftime("%Y-%m-%d")
        fig.add_trace(go.Scatter(x=df["date_label"], y=df["return_pct"], mode="lines", name=item["name"]))
    fig.add_hline(y=0, line_dash="dot", line_color="#888")
    fig.update_layout(height=420, yaxis_title="近 30 日涨跌幅（%）", hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20))
    fig.update_xaxes(type="category", nticks=10)
    return style_plotly_figure(fig)


def build_realtime_minute_figure(minute_df: pd.DataFrame, quote: dict | None = None) -> go.Figure:
    """构建实时分时走势和分钟成交量图。"""
    chart_df = minute_df.copy()
    chart_df["time_label"] = chart_df["datetime"].dt.strftime("%H:%M")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=("实时分时走势", "分钟成交量"),
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["time_label"],
            y=chart_df["close"],
            mode="lines",
            name="实时价格",
            line=dict(color="#2563eb", width=2),
            hovertemplate="时间：%{x}<br>价格：%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    prev_close = quote.get("prev_close") if quote else None
    if prev_close is not None and not pd.isna(prev_close):
        fig.add_hline(y=prev_close, line_dash="dot", line_color="#64748b", annotation_text="昨收", row=1, col=1)
    colors = ["#ef5350" if close >= open_ else "#26a69a" for open_, close in zip(chart_df["open"], chart_df["close"])]
    fig.add_trace(
        go.Bar(
            x=chart_df["time_label"],
            y=chart_df["volume"],
            name="分钟成交量",
            marker_color=colors,
            hovertemplate="时间：%{x}<br>成交量：%{y:.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=58, b=20), hovermode="x unified", bargap=0.1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_xaxes(type="category", nticks=12)
    return style_plotly_figure(fig)


def build_compare_analysis_row(dataset: dict, period: int) -> dict:
    """整理多股对比中的单只股票技术分析摘要。"""
    df = dataset["df"]
    info = dataset["info"]
    analysis_result = dataset["analysis"]
    period_df = df.tail(period)
    if len(period_df) >= 2 and period_df.iloc[0]["close"]:
        period_return = (period_df.iloc[-1]["close"] / period_df.iloc[0]["close"] - 1) * 100
    else:
        period_return = None
    latest = df.iloc[-1]
    risks = analysis_result.get("风险提示列表", [])
    return {
        "股票代码": info.get("code", dataset.get("code", "")),
        "股票名称": info.get("name", ""),
        f"近{period}日涨跌幅(%)": round(period_return, 2) if period_return is not None else None,
        "最近收盘价": round(float(latest["close"]), 2),
        "当前趋势": analysis_result.get("趋势判断", "暂无"),
        "后市倾向": analysis_result.get("后市倾向", "暂无"),
        "技术评分": analysis_result.get("后市评分", 0),
        "判断依据": analysis_result.get("后市判断依据", "暂无"),
        "主要风险": risks[0] if risks else "暂无明显风险提示",
    }


def build_recent_data_table(df):
    """整理最近行情数据表，使用中文列名和更易读的成交量单位。"""
    table_df = df.tail(30).sort_values("date", ascending=False).copy()
    table_df["日期"] = table_df["date"].dt.strftime("%Y-%m-%d")
    table_df["成交量(万手)"] = table_df["volume"] / 10000
    if "amount" in table_df.columns:
        table_df["成交额(亿元)"] = table_df["amount"] / 100000000
    columns = {
        "open": "开盘",
        "close": "收盘",
        "high": "最高",
        "low": "最低",
        "pct_chg": "涨跌幅(%)",
        "MA5": "MA5",
        "MA10": "MA10",
        "MA20": "MA20",
        "MA60": "MA60",
        "成交量(万手)": "成交量(万手)",
        "成交额(亿元)": "成交额(亿元)",
    }
    visible_columns = ["日期"] + [col for col in columns if col in table_df.columns]
    table_df = table_df[visible_columns].rename(columns=columns)
    number_columns = [col for col in table_df.columns if col != "日期"]
    table_df[number_columns] = table_df[number_columns].round(2)
    return table_df


def build_watchlist_row(code: str, days: int) -> tuple[dict | None, str | None]:
    """获取单只自选股摘要行，失败时返回错误文本。"""
    try:
        dataset = get_stock_dataset(code, days)
        df = dataset["df"]
        info = dataset["info"]
        latest = df.iloc[-1]
        return {
            "股票代码": code,
            "股票名称": info.get("name", code),
            "最新价/收盘价": round(float(info.get("latest_price") if not pd.isna(info.get("latest_price")) else latest["close"]), 2),
            "涨跌幅(%)": round(float(info.get("pct_chg")) if not pd.isna(info.get("pct_chg")) else 0, 2),
            "成交量(万手)": round(float(latest["volume"]) / 10000, 2),
            "趋势判断": dataset["analysis"].get("趋势判断", "暂无"),
        }, None
    except Exception as exc:
        return None, f"{code} 获取失败：{exc}"


def render_analysis_report(analysis_result: dict) -> None:
    """在页面中渲染报告化文字分析。"""
    st.subheader("简短分析报告")
    st.write(analysis_result.get("简短报告", "暂无分析结果。"))

    st.subheader("风险提示")
    for risk in analysis_result.get("风险提示列表", []):
        st.warning(risk)

    conclusion = analysis_result["综合分析结论"]
    st.subheader("综合分析结论")
    st.markdown(
        f"""
- **趋势判断：** {conclusion["趋势判断"]}
- **技术面信号：** {conclusion["技术面信号"]}
- **风险点：** {conclusion["风险点"]}
- **{conclusion["免责声明"]}**
"""
    )


def format_metric_value(value, suffix: str = "") -> str:
    """格式化指标卡片数值，空值时显示暂无数据。"""
    if value is None or pd.isna(value):
        return "暂无数据"
    return f"{float(value):.2f}{suffix}"


def format_percent_interval(interval) -> str:
    """格式化涨跌幅区间。"""
    if not interval or len(interval) != 2:
        return "暂无"
    return f"{float(interval[0]):.2%} ~ {float(interval[1]):.2%}"


def format_price_interval(interval) -> str:
    """格式化价格区间。"""
    if not interval or len(interval) != 2:
        return "暂无"
    low, high = sorted([float(interval[0]), float(interval[1])])
    return f"{low:.2f} ~ {high:.2f}"


def get_plotly_config() -> dict:
    """返回 Plotly 图表交互配置，支持滚轮缩放和模式栏操作。"""
    return {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        "toImageButtonOptions": {"format": "png", "filename": "stock_analysis_chart", "scale": 2},
    }


def render_sidebar() -> tuple[str, str, int, list[str], dict]:
    """渲染左侧边栏，并返回当前功能、选中代码、历史范围和自选股。"""
    app_config = load_config()
    with st.sidebar:
        st.header("短线研究助手")
        default_code = st.session_state.get("selected_stock", "600519")
        selected_code = st.text_input("股票代码", value=default_code, placeholder="例如 600519、000001、300750")
        days = st.selectbox("分析周期", [30, 60, 120, 250], index=3)
        threshold_percent = st.slider("预测阈值（%）", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
        feature = st.radio(
            "功能选择",
            ["个股技术分析", "1-3 日走势预测", "模型回测", "分析报告"],
        )

        st.divider()
        st.subheader("自选股")
        new_code = st.text_input("添加自选股", placeholder="例如 600519")
        if st.button("添加到自选股", type="primary"):
            ok, message = add_to_watchlist(new_code)
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.warning(message)

        watchlist = load_watchlist()
        st.subheader("自选股列表")
        if not watchlist:
            st.caption("暂无自选股。")
        else:
            for code in watchlist:
                col_select, col_delete = st.columns([3, 1])
                if col_select.button(code, key=f"select_{code}"):
                    st.session_state["selected_stock"] = code
                    st.rerun()
                if col_delete.button("删除", key=f"delete_{code}"):
                    ok, message = remove_from_watchlist(code)
                    if ok:
                        st.success(message)
                        st.rerun()
                    else:
                        st.warning(message)

        app_config["prediction_threshold"] = threshold_percent / 100
    return feature, selected_code, days, watchlist, app_config


def render_market_overview() -> None:
    """渲染市场概览页面。"""
    st.header("市场概览")
    with st.spinner("正在加载主要指数和市场情绪..."):
        dataset = get_market_dataset()
    for error in dataset["errors"]:
        st.warning(error)
    indices = dataset["indices"]
    if not indices:
        st.info("当前没有获取到主要指数数据，请稍后重试。")
        return

    cols = st.columns(min(4, len(indices)))
    for col, item in zip(cols, indices):
        col.metric(item["name"], format_number(item["latest"]), format_number(item["pct_chg"], suffix="%"))
        col.caption(f"趋势：{item['trend']}")
    st.subheader("最近 30 日主要指数走势")
    st.plotly_chart(build_index_trend_figure(indices), use_container_width=True, config=get_plotly_config())
    st.subheader("市场情绪")
    st.info(dataset["sentiment"])


def render_sector_page() -> None:
    """渲染行业板块分析页面。"""
    st.header("行业板块分析")
    try:
        with st.spinner("正在加载行业板块数据..."):
            sector_df = get_sector_dataset()
    except SectorDataError as exc:
        st.error(str(exc))
        st.info("当前行业板块数据源为东方财富行业板块接口：ak.stock_board_industry_name_em()。如果网络代理阻断该接口，请稍后重试。")
        return
    except Exception as exc:
        st.error("行业板块数据暂不可用，请稍后重试。")
        return
    if sector_df.empty:
        st.info("行业板块接口返回为空。")
        return

    column_names = {
        "code": "板块代码",
        "name": "板块名称",
        "price": "最新价",
        "pct_chg": "涨跌幅",
        "turnover": "换手率",
        "up_count": "上涨家数",
        "down_count": "下跌家数",
        "leading_stock": "领涨股票",
        "leading_stock_pct": "领涨股票-涨跌幅",
        "amount": "成交额",
    }
    rankings = get_sector_rankings(sector_df)
    tab_gain, tab_loss, tab_amount = st.tabs(["今日涨幅前 10", "今日跌幅前 10", "成交额前 10"])
    display_cols = [col for col in ["code", "name", "price", "pct_chg", "turnover", "up_count", "down_count", "leading_stock", "leading_stock_pct", "amount"] if col in sector_df.columns]
    with tab_gain:
        st.dataframe(rankings["top_gain"][display_cols].rename(columns=column_names), use_container_width=True, hide_index=True)
    with tab_loss:
        st.dataframe(rankings["top_loss"][display_cols].rename(columns=column_names), use_container_width=True, hide_index=True)
    with tab_amount:
        st.dataframe(rankings["top_amount"][display_cols].rename(columns=column_names), use_container_width=True, hide_index=True)

    sector_names = sector_df["name"].dropna().astype(str).tolist()
    selected = st.selectbox("选择板块查看成分股", sector_names)
    row = sector_df[sector_df["name"].astype(str) == selected].iloc[0]
    st.subheader(f"{selected} 板块分析")
    st.write(analyze_sector(row))
    try:
        with st.spinner("正在加载板块成分股..."):
            stocks_df = get_sector_stocks_dataset(selected)
        if stocks_df.empty:
            st.info("没有获取到板块成分股。")
        else:
            st.dataframe(stocks_df.head(30), use_container_width=True, hide_index=True)
    except SectorDataError as exc:
        st.warning(str(exc))
    except Exception:
        st.warning("板块成分股暂不可用，请稍后重试。")


def render_news_page(stock_code: str) -> None:
    """渲染新闻公告页面。"""
    st.header("新闻与公告")
    try:
        code = normalize_stock_code(stock_code)
    except DataFetchError as exc:
        st.error(str(exc))
        return
    try:
        with st.spinner(f"正在加载 {code} 新闻公告..."):
            dataset = get_news_dataset(code)
    except Exception as exc:
        st.error(f"新闻公告数据暂不可用：{exc}")
        return
    news_df = dataset["news"]
    announcements_df = dataset["announcements"]
    st.subheader("相关新闻")
    if news_df.empty:
        st.info("当前没有获取到相关新闻，或新闻接口暂不可用。")
    else:
        show_cols = [col for col in ["title", "time", "source", "分类", "url"] if col in news_df.columns]
        st.dataframe(news_df[show_cols].head(20), use_container_width=True, hide_index=True)
    st.subheader("公告列表")
    if announcements_df.empty:
        st.info("当前没有获取到公告，后续可接入更稳定公告源。")
    else:
        st.dataframe(announcements_df.head(20), use_container_width=True, hide_index=True)


def render_settings_page() -> None:
    """渲染系统设置页面。"""
    st.header("系统设置")
    current = load_config()
    with st.form("settings_form"):
        default_period = st.selectbox("默认分析周期", [30, 60, 120, 250], index=[30, 60, 120, 250].index(int(current.get("default_period", 250))))
        default_pool = st.radio("默认股票池", ["自选股", "用户手动输入"], index=0 if current.get("default_pool") == "自选股" else 1, horizontal=True)
        st.subheader("高级指标显示")
        show_macd = st.checkbox("MACD", value=bool(current.get("show_macd", True)))
        show_rsi = st.checkbox("RSI", value=bool(current.get("show_rsi", True)))
        show_boll = st.checkbox("BOLL", value=bool(current.get("show_boll", True)))
        submitted = st.form_submit_button("保存设置", type="primary")
    if submitted:
        save_config(
            {
                "default_period": default_period,
                "default_pool": default_pool,
                "show_macd": show_macd,
                "show_rsi": show_rsi,
                "show_boll": show_boll,
            }
        )
        st.success("设置已保存。")


def render_watchlist_dashboard(watchlist: list[str], days: int) -> None:
    """渲染自选股看板页面。"""
    st.header("自选股看板")
    if not watchlist:
        st.info("请先在左侧边栏添加自选股。")
        return

    rows = []
    errors = []
    progress = st.progress(0, text="正在加载自选股行情...")
    for index, code in enumerate(watchlist, start=1):
        row, error = build_watchlist_row(code, days)
        if row:
            rows.append(row)
        if error:
            errors.append(error)
        progress.progress(index / len(watchlist), text=f"正在加载 {code}...")
    progress.empty()

    for error in errors:
        st.warning(error)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_multi_compare(days: int) -> None:
    """渲染多股票区间涨跌幅对比页面。"""
    st.header("多股票对比")
    code_text = st.text_input("输入多只股票代码", value="600519, 000001, 300750", help="用英文逗号或中文逗号分隔。")
    period = st.radio("对比周期", options=[30, 60, 120], index=1, horizontal=True, format_func=lambda x: f"最近 {x} 个交易日")
    if st.button("开始对比", type="primary"):
        codes, parse_errors = parse_stock_codes(code_text)
        for error in parse_errors:
            st.warning(error)
        if not codes:
            st.error("请输入至少一只有效的 6 位股票代码。")
            return

        series_map = {}
        analysis_rows = []
        detail_datasets = []
        errors = []
        with st.spinner("正在获取多只股票行情..."):
            for code in codes:
                try:
                    dataset = get_stock_dataset(code, max(days, 180))
                    info = dataset["info"]
                    series_map[f"{info.get('name', code)}（{code}）"] = dataset["df"]
                    analysis_rows.append(build_compare_analysis_row(dataset, period))
                    detail_datasets.append(dataset)
                except Exception as exc:
                    errors.append(f"{code} 获取失败：{exc}")

        for error in errors:
            st.warning(error)
        if not series_map:
            st.error("没有成功获取到可对比的股票数据。")
            return
        st.plotly_chart(build_compare_figure(series_map, period), use_container_width=True, config=get_plotly_config())
        st.subheader("多股技术面分析")
        st.caption("后市倾向为技术指标综合判断，不代表确定性涨跌预测。仅供学习和研究使用，不构成投资建议。")
        st.dataframe(pd.DataFrame(analysis_rows), use_container_width=True, hide_index=True)

        with st.expander("查看每只股票的详细分析依据"):
            for dataset in detail_datasets:
                info = dataset["info"]
                analysis_result = dataset["analysis"]
                st.markdown(f"#### {info.get('name', dataset['code'])}（{dataset['code']}）")
                st.write(f"**后市倾向：** {analysis_result.get('后市倾向', '暂无')}，技术评分：{analysis_result.get('后市评分', 0)}")
                st.write(f"**判断依据：** {analysis_result.get('后市判断依据', '暂无')}")
                st.write(f"**均线：** {analysis_result.get('均线状态', '暂无')}")
                st.write(f"**MACD：** {analysis_result.get('MACD 状态', '暂无')}")
                st.write(f"**RSI：** {analysis_result.get('RSI 状态', '暂无')}")
                st.write(f"**BOLL：** {analysis_result.get('BOLL 状态', '暂无')}")
                for risk in analysis_result.get("风险提示列表", []):
                    st.warning(risk)


def render_score_panel(score_result: dict) -> None:
    """渲染综合评分面板。"""
    st.subheader("综合评分")
    col_score, col_grade = st.columns(2)
    col_score.metric("综合分数", format_number(score_result.get("综合分数")))
    col_grade.metric("等级", score_result.get("等级", "暂无"))
    detail_df = pd.DataFrame(
        [{"维度": key, "得分": value, "说明": "；".join(score_result.get("评分解释", {}).get(key, []))} for key, value in score_result.get("评分明细", {}).items()]
    )
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
    st.caption("评分仅供学习和研究使用，不构成投资建议。")


def render_stock_detail(stock_code: str, days: int, show_export: bool = False, app_config: dict | None = None) -> None:
    """渲染个股详细分析页面，可选择显示报告导出按钮。"""
    app_config = app_config or load_config()
    st.header("个股详细分析" if not show_export else "分析报告导出")
    try:
        code = normalize_stock_code(stock_code)
    except DataFetchError as exc:
        st.error(str(exc))
        return

    try:
        with st.spinner(f"正在加载 {code} 行情和指标..."):
            dataset = get_stock_dataset(code, days)
    except Exception as exc:
        st.error(f"{code} 数据获取失败：{exc}")
        return

    for warning in dataset["warnings"]:
        st.warning(warning)

    df = dataset["df"]
    info = dataset["info"]
    analysis_result = dataset["analysis"]
    latest_price = info.get("latest_price")
    pct_chg = info.get("pct_chg")
    score_result = build_score(df)

    st.subheader(f"{info.get('name', code)}（{code}）")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前价格 / 最近收盘价", format_metric_value(latest_price))
    col2.metric("涨跌幅", format_metric_value(pct_chg, "%"))
    col3.metric("成交量", f"{df.iloc[-1]['volume'] / 10000:.2f} 万手")
    col4.metric("趋势判断", analysis_result.get("趋势判断", "暂无"))

    try:
        quote = get_realtime_quote_dataset(code)
        minute_df = get_realtime_minute_dataset(code)
        st.subheader("实时行情与分时走势")
        realtime_cols = st.columns(5)
        realtime_cols[0].metric("实时最新价", format_metric_value(quote.get("latest_price")))
        realtime_cols[1].metric("实时涨跌幅", format_metric_value(quote.get("pct_chg"), "%"))
        realtime_cols[2].metric("今日最高", format_metric_value(quote.get("high")))
        realtime_cols[3].metric("今日最低", format_metric_value(quote.get("low")))
        realtime_cols[4].metric("实时成交额", f"{(quote.get('amount') or 0) / 100000000:.2f} 亿元")
        st.caption(f"实时来源：{quote.get('source')}，时间戳：{quote.get('timestamp', '暂无')}。")
        st.plotly_chart(build_realtime_minute_figure(minute_df, quote), use_container_width=True, config=get_plotly_config())
    except RealtimeQuoteError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.warning(f"实时分时走势暂不可用：{exc}")

    if show_export:
        markdown = generate_markdown_report(info, df, analysis_result, score_result=score_result)
        st.download_button(
            "导出分析报告（Markdown）",
            data=markdown,
            file_name=f"{code}_analysis_report.md",
            mime="text/markdown",
            type="primary",
        )
        st.caption("PDF 导出暂作为后续功能保留，当前版本优先提供稳定的 Markdown 报告。")
        with st.expander("预览报告内容", expanded=True):
            st.markdown(markdown)

    st.caption("图表支持鼠标滚轮缩放、拖拽框选放大、双击复位；也可以拖动底部范围条查看局部行情。")
    st.plotly_chart(build_price_volume_figure(df, show_boll=app_config.get("show_boll", True)), use_container_width=True, config=get_plotly_config())

    if app_config.get("show_macd", True) or app_config.get("show_rsi", True):
        col_macd, col_rsi = st.columns([2, 1])
        with col_macd:
            if app_config.get("show_macd", True):
                st.subheader("MACD")
                st.plotly_chart(build_macd_figure(df), use_container_width=True, config=get_plotly_config())
        with col_rsi:
            if app_config.get("show_rsi", True):
                st.subheader("RSI")
                st.plotly_chart(build_rsi_figure(df), use_container_width=True, config=get_plotly_config())

    render_score_panel(score_result)

    with st.expander("查看最近行情数据"):
        st.dataframe(build_recent_data_table(df), use_container_width=True, hide_index=True)

    render_analysis_report(analysis_result)


def render_short_term_prediction(stock_code: str, threshold: float) -> None:
    """渲染未来 1-3 日走势预测页面。"""
    st.header("未来 1-3 日走势预测")
    try:
        code = normalize_stock_code(stock_code)
    except DataFetchError as exc:
        st.error(str(exc))
        return

    try:
        with st.spinner(f"正在获取 {code} 前复权历史行情并训练短线模型..."):
            result = get_prediction_dataset(code, threshold)
    except Exception as exc:
        st.error(f"短线预测暂不可用：{exc}")
        return

    try:
        update_prediction_outcomes(code, threshold, hist_df=result.get("data"))
        result = apply_feedback_calibration(code, result)
    except Exception:
        pass

    if result.get("low_reliability"):
        st.warning("历史数据不足 250 个交易日，预测可靠性较低。")
    stale_dates = [
        item.get("预测日期")
        for item in result["predictions"]
        if item.get("预测日期") and pd.to_datetime(item["预测日期"]).date() < pd.Timestamp.today().date()
    ]
    if stale_dates:
        st.warning(
            f"当前行情数据最新到 {result.get('last_trade_date', '暂无')}，部分预测日期早于今天。"
            "这说明数据源可能尚未更新到最新交易日，预测结果按最新可用交易日向后推算。"
        )

    try:
        with st.spinner("正在获取实时行情用于对比..."):
            quote = get_realtime_quote_dataset(code)
        comparison = compare_realtime_with_prediction(quote, result)
        st.subheader("实时行情对比")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("实时最新价", format_metric_value(quote.get("latest_price")))
        col2.metric("实时涨跌幅", format_metric_value(quote.get("pct_chg"), "%"))
        col3.metric("成交额", f"{(quote.get('amount') or 0) / 100000000:.2f} 亿元")
        col4.metric("预测 vs 实时", comparison["对比结论"])
        st.caption(
            f"实时来源：{quote.get('source')}，时间戳：{quote.get('timestamp', '暂无')}。"
            f"1日预测方向：{comparison['预测方向']}，实时盘中方向：{comparison['实时方向']}。{comparison['说明']}"
        )
        with st.expander("查看实时分时走势图", expanded=True):
            minute_df = get_realtime_minute_dataset(code)
            st.plotly_chart(build_realtime_minute_figure(minute_df, quote), use_container_width=True, config=get_plotly_config())
    except RealtimeQuoteError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.warning(f"实时行情对比暂不可用：{exc}")

    rows = []
    for item in result["predictions"]:
        rows.append(
            {
                "预测日期": item.get("预测日期", ""),
                "周期": item["周期"],
                "预测走势": item["预测方向"],
                "上涨概率": f"{item['上涨概率']:.1%}",
                "震荡概率": f"{item['震荡概率']:.1%}",
                "下跌概率": f"{item['下跌概率']:.1%}",
                "预测涨跌幅区间": format_percent_interval(item.get("预测涨跌幅区间")),
                "预测价位区间": format_price_interval(item.get("预测价位区间")),
                "置信度": item["置信度"],
                "模型": item["模型"],
            }
        )
    st.caption(
        f"最近一个已获取交易日：{result.get('last_trade_date', '暂无')}。"
        f"预测日期来源：{result.get('calendar_source', '暂无')}。"
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    next_day = next((item for item in result["predictions"] if item.get("周期") == "1日"), None)
    if next_day and next_day.get("区间估算"):
        ranges = next_day["区间估算"]
        st.subheader("次日涨跌幅与价位区间预测")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("预测交易日", next_day.get("预测日期", "暂无"))
        col2.metric("基准收盘价", format_metric_value(ranges.get("基准价")))
        col3.metric("模型方向", next_day.get("预测方向", "暂无"))
        col4.metric("方向价位区间", format_price_interval(next_day.get("预测价位区间")))

        scenario_rows = [
            {
                "情景": "如果上涨",
                "概率": f"{next_day['上涨概率']:.1%}",
                "涨跌幅区间": format_percent_interval(ranges.get("上涨幅度区间")),
                "对应价位区间": format_price_interval(ranges.get("上涨价位区间")),
            },
            {
                "情景": "如果震荡",
                "概率": f"{next_day['震荡概率']:.1%}",
                "涨跌幅区间": format_percent_interval(ranges.get("震荡幅度区间")),
                "对应价位区间": format_price_interval(ranges.get("震荡价位区间")),
            },
            {
                "情景": "如果下跌",
                "概率": f"{next_day['下跌概率']:.1%}",
                "涨跌幅区间": format_percent_interval(ranges.get("下跌幅度区间")),
                "对应价位区间": format_price_interval(ranges.get("下跌价位区间")),
            },
        ]
        st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True, hide_index=True)
        st.caption(
            f"{ranges.get('区间说明', '')} 区间会随最新历史行情、阈值和复盘校准变化；不代表确定目标价。"
        )

    try:
        record_predictions(code, result)
        update_prediction_outcomes(code, threshold, hist_df=result.get("data"))
        feedback_summary = summarize_feedback(code)
        st.subheader("预测复盘与自我修正")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("已记录预测", feedback_summary["total"])
        col2.metric("已完成复盘", feedback_summary["evaluated"])
        col3.metric("等待结果", feedback_summary["pending"])
        accuracy_text = "暂无" if feedback_summary["accuracy"] is None else f"{feedback_summary['accuracy']:.1%}"
        col4.metric("历史准确率", accuracy_text)

        if feedback_summary["by_horizon"]:
            horizon_df = pd.DataFrame(feedback_summary["by_horizon"])
            horizon_df["准确率"] = horizon_df["准确率"].map(lambda value: f"{value:.1%}")
            st.write("**分周期复盘表现**")
            st.dataframe(horizon_df, use_container_width=True, hide_index=True)

        if feedback_summary["bias"]:
            st.write("**常见错判类型**")
            st.dataframe(pd.DataFrame(feedback_summary["bias"]), use_container_width=True, hide_index=True)

        if feedback_summary["mistakes"]:
            mistake_rows = []
            for item in feedback_summary["mistakes"]:
                mistake_rows.append(
                    {
                        "预测基准日": item.get("base_date", ""),
                        "目标日期": item.get("target_date", ""),
                        "周期": item.get("horizon", ""),
                        "预测": item.get("predicted", ""),
                        "实际": item.get("actual", ""),
                        "实际涨跌幅": f"{item.get('actual_return', 0):.2%}",
                        "模型": item.get("model", ""),
                    }
                )
            st.write("**最近预测错误记录**")
            st.dataframe(pd.DataFrame(mistake_rows), use_container_width=True, hide_index=True)

        st.info(feedback_summary["summary"])
        st.caption(
            "复盘机制会在后续交易日行情可用后自动核对结果；样本越多，错误总结越有参考价值，但不会保证未来预测一定准确。"
        )
    except Exception as exc:
        st.warning(f"预测复盘暂不可用：{exc}")

    st.subheader("预测依据")
    for item in result["predictions"]:
        with st.expander(f"{item['周期']}预测依据：{item['预测方向']}"):
            st.write(f"**模型判断：** 短线{item['预测方向']}概率相对较高。")
            st.write(f"**预测依据：** {item['预测依据']}")
            st.write("**主要支撑信号：**")
            for signal in item.get("主要支撑信号", []):
                st.write(f"- {signal}")
            st.write("**主要风险信号：**")
            for signal in item.get("主要风险信号", []):
                st.write(f"- {signal}")
            st.warning(item["风险提示"])

    st.subheader("模型说明")
    st.info(result["model_note"])
    st.caption("免责声明：模型输出仅用于学习和研究，不代表未来一定上涨或下跌，不构成任何投资建议。")


def render_model_backtest(stock_code: str, threshold: float) -> None:
    """渲染模型滚动回测页面。"""
    st.header("模型回测")
    try:
        code = normalize_stock_code(stock_code)
    except DataFetchError as exc:
        st.error(str(exc))
        return

    try:
        with st.spinner(f"正在执行 {code} 滚动回测，这可能需要一些时间..."):
            result = get_backtest_dataset(code, threshold, test_days=120)
    except Exception as exc:
        st.error(f"模型回测暂不可用：{exc}")
        return

    col1, col2 = st.columns(2)
    col1.metric("最近 60 个交易日预测准确率", "暂无" if result["accuracy_60"] is None else f"{result['accuracy_60']:.1%}")
    col2.metric("最近 120 个交易日预测准确率", "暂无" if result["accuracy_120"] is None else f"{result['accuracy_120']:.1%}")

    accuracy_rows = []
    for name, value in result["accuracy_by_horizon"].items():
        accuracy_rows.append({"指标": name, "准确率": "暂无" if value is None else f"{value:.1%}"})
    st.subheader("分周期准确率")
    st.dataframe(pd.DataFrame(accuracy_rows), use_container_width=True, hide_index=True)

    st.subheader("混淆矩阵")
    st.dataframe(result["confusion_matrix"], use_container_width=True)

    st.subheader("实际走势 vs 预测走势")
    st.plotly_chart(style_plotly_figure(build_backtest_figure(result["results"])), use_container_width=True, config=get_plotly_config())
    st.info(result["note"])
    st.caption("回测结果只说明历史表现，不代表未来一定准确。")


def main():
    """Streamlit 应用入口，负责看板导航和页面分发。"""
    st.set_page_config(page_title="A股短线研究助手", layout="wide")
    apply_custom_theme()
    st.title("A股短线研究助手")
    st.markdown(
        """
        <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;padding:14px 18px;margin:4px 0 22px 0;box-shadow:0 8px 24px rgba(15,23,42,.05);">
            <div style="color:#374151;font-size:15px;line-height:1.7;">
                面向 A 股短线研究的本地分析工具，覆盖技术分析、1-3 日概率预测、滚动回测和 Markdown 报告导出。
                仅供学习和研究使用，不包含自动交易，不提供买入或卖出建议。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature, selected_code, days, watchlist, app_config = render_sidebar()
    threshold = app_config.get("prediction_threshold", 0.01)

    if feature == "个股技术分析":
        render_stock_detail(selected_code, days, app_config=app_config)
    elif feature == "1-3 日走势预测":
        render_short_term_prediction(selected_code, threshold)
    elif feature == "模型回测":
        render_model_backtest(selected_code, threshold)
    elif feature == "分析报告":
        render_stock_detail(selected_code, days, show_export=True, app_config=app_config)


if __name__ == "__main__":
    main()
