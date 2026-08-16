import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from time import perf_counter

from ai_interpreter import AIInterpreterError, generate_stock_interpretation
from analysis import build_analysis, detect_anomaly_flags, estimate_support_resistance
from config import load_config, save_config
from data_fetch import DataFetchError, fetch_stock_basic_info, fetch_stock_history, normalize_stock_code
from feedback import (
    build_feedback_export_table,
    apply_feedback_calibration,
    get_today_prediction_record_status,
    record_historical_replay,
    record_predictions,
    summarize_feedback,
    update_prediction_outcomes,
)
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


THEME_COOKIE_NAME = "short_term_research_theme"
THEME_COOKIE_MAX_AGE = 60 * 60 * 24 * 365


def get_saved_theme() -> str:
    """读取当前浏览器保存的主题偏好。"""
    try:
        saved_theme = st.context.cookies.get(THEME_COOKIE_NAME)
    except Exception:
        saved_theme = None
    return "深色" if saved_theme == "dark" else "明亮"


def persist_theme_preference(theme_mode: str) -> None:
    """把主题偏好写入当前浏览器，有效期一年。"""
    theme_value = "dark" if theme_mode == "深色" else "light"
    components.html(
        f"""
        <script>
        (() => {{
            const preference = "{THEME_COOKIE_NAME}={theme_value}; Path=/; Max-Age={THEME_COOKIE_MAX_AGE}; SameSite=Lax";
            try {{
                window.parent.document.cookie = preference;
            }} catch (error) {{
                document.cookie = preference;
            }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def get_theme_palette(theme_mode: str | None = None) -> dict[str, str]:
    """返回明亮或深色研究终端的统一颜色令牌。"""
    if (theme_mode or st.session_state.get("ui_theme")) == "深色":
        return {
            "bg": "#080a0d",
            "panel": "#101318",
            "panel_soft": "#15191f",
            "sidebar": "#0c0f13",
            "text": "#f4f6f8",
            "muted": "#a4abb5",
            "line": "#242a32",
            "line_strong": "#343c46",
            "accent": "#d7dde5",
            "accent_hover": "#ffffff",
            "accent_soft": "#1b222b",
            "input": "#11151a",
            "button_text": "#101318",
            "chart": "#0d1117",
            "legend": "rgba(16,19,24,0.90)",
            "grid": "#252c35",
            "link": "#d8dee7",
            "color_scheme": "dark",
        }
    return {
        "bg": "#ffffff",
        "panel": "#ffffff",
        "panel_soft": "#f6f7f8",
        "sidebar": "#f3f4f6",
        "text": "#111318",
        "muted": "#5f6670",
        "line": "#e2e5e9",
        "line_strong": "#cbd1d8",
        "accent": "#334155",
        "accent_hover": "#1f2937",
        "accent_soft": "#e8edf2",
        "input": "#ffffff",
        "button_text": "#ffffff",
        "chart": "#fbfcfd",
        "legend": "rgba(255,255,255,0.90)",
        "grid": "#e8ebef",
        "link": "#334155",
        "color_scheme": "light",
    }


def apply_custom_theme(theme_mode: str | None = None) -> None:
    """注入可切换主题，保持研究工作台的克制感和可读性。"""
    palette = get_theme_palette(theme_mode)
    css = """
<style>
    :root {
        --bg: __BG__;
        --panel: __PANEL__;
        --panel-soft: __PANEL_SOFT__;
        --sidebar: __SIDEBAR__;
        --text: __TEXT__;
        --muted: __MUTED__;
        --line: __LINE__;
        --line-strong: __LINE_STRONG__;
        --accent: __ACCENT__;
        --accent-hover: __ACCENT_HOVER__;
        --accent-soft: __ACCENT_SOFT__;
        --input: __INPUT__;
        --button-text: __BUTTON_TEXT__;
        --link: __LINK__;
    }

    html, body,
    [data-testid="stAppViewContainer"],
    .stApp {
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    [data-testid="stHeader"] {
        background: color-mix(in srgb, var(--bg) 92%, transparent) !important;
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar) !important;
        border-right: 1px solid var(--line-strong);
    }

    /* Keep the sidebar toggle visible on both light and dark themes. */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stExpandSidebarButton"] {
        color: var(--text) !important;
    }

    [data-testid="stExpandSidebarButton"] * {
        color: var(--text) !important;
    }

    [data-testid="stSidebarCollapseButton"] button svg,
    [data-testid="stSidebarCollapsedControl"] button svg,
    [data-testid="stSidebarCollapseButton"] button svg path,
    [data-testid="stSidebarCollapsedControl"] button svg path {
        color: var(--text) !important;
        fill: var(--text) !important;
        stroke: var(--text) !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="stExpandSidebarButton"]:hover {
        background: var(--accent-soft) !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] summary {
        color: var(--text) !important;
    }

    input[type="radio"],
    input[type="range"] {
        accent-color: var(--accent);
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] select,
    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    [data-baseweb="textarea"] > div {
        background: var(--input) !important;
        border: 1px solid var(--line-strong) !important;
        color: var(--text) !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }

    input, textarea, select {
        color: var(--text) !important;
        caret-color: var(--text) !important;
    }

    input::placeholder, textarea::placeholder {
        color: var(--muted) !important;
        opacity: 1;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background: var(--panel) !important;
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: var(--panel-soft);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 7px 9px;
        margin: 3px 0;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text) !important;
    }

    .block-container {
        max-width: 1360px;
        padding-top: 1.2rem;
        padding-bottom: 2.4rem;
    }

    h1, h2, h3, h4, h5, h6,
    p, li, label,
    [data-testid="stMarkdownContainer"] {
        color: var(--text);
    }

    h1 {
        font-weight: 760;
        letter-spacing: 0;
        padding-bottom: 0.1rem;
        font-size: clamp(1.65rem, 2vw, 2.15rem);
    }

    h2, h3 {
        letter-spacing: 0;
        font-weight: 700;
    }

    .terminal-header {
        border-top: 1px solid var(--line-strong);
        border-bottom: 1px solid var(--line-strong);
        padding: 10px 0 12px 0;
        margin: 2px 0 20px 0;
        display: flex;
        gap: 18px;
        align-items: center;
        flex-wrap: wrap;
        color: var(--muted);
        font-size: 0.92rem;
    }

    .terminal-header strong {
        color: var(--text);
        font-weight: 700;
    }

    .terminal-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--accent);
        margin-right: 7px;
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 13px 14px;
        box-shadow: none;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.82rem;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-weight: 720;
        font-size: clamp(1.18rem, 1.8vw, 1.76rem);
        line-height: 1.15;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"],
    [data-testid="stPlotlyChart"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 8px;
        box-shadow: none;
        color-scheme: __COLOR_SCHEME__;
    }

    [data-testid="stTable"] table,
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        background: var(--panel) !important;
        color: var(--text) !important;
        border-color: var(--line) !important;
    }

    .stAlert {
        background: var(--panel-soft) !important;
        color: var(--text) !important;
        border-radius: 6px;
        border: 1px solid var(--line-strong);
    }

    div[data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 6px;
        box-shadow: none;
    }

    div[data-testid="stExpander"] summary,
    [data-testid="stStatusWidget"] {
        color: var(--text) !important;
        background: var(--panel) !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 6px;
        border: 1px solid var(--accent);
        background: var(--accent);
        color: var(--button-text);
        font-weight: 650;
        box-shadow: none;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: var(--accent-hover);
        border-color: var(--accent-hover);
        color: var(--button-text);
    }

    .stButton > button *,
    .stDownloadButton > button *,
    [data-testid="stSidebar"] .stButton > button *,
    [data-testid="stSidebar"] .stDownloadButton > button * {
        color: var(--button-text) !important;
    }

    .stCaptionContainer,
    caption,
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    a {
        color: var(--link) !important;
    }

    code, pre {
        background: var(--panel-soft) !important;
        color: var(--text) !important;
        border-color: var(--line) !important;
    }

    hr {
        border-color: var(--line) !important;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 0.8rem;
    }
</style>
"""
    for token, value in palette.items():
        css = css.replace(f"__{token.upper()}__", value)
    st.markdown(css, unsafe_allow_html=True)


def style_plotly_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    """统一 Plotly 图表视觉风格。"""
    palette = get_theme_palette()
    dark_mode = st.session_state.get("ui_theme") == "深色"
    fig.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=palette["chart"],
        font=dict(color=palette["text"], family="Arial"),
        legend=dict(bgcolor=palette["legend"], bordercolor=palette["line"], borderwidth=1),
        margin=dict(l=20, r=20, t=38, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor=palette["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=palette["grid"], zeroline=False)
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
def get_prediction_dataset(stock_code: str, threshold: float, threshold_mode: str = "manual") -> dict:
    """带缓存获取 1-3 日短线预测结果。"""
    return predict_short_term(stock_code, threshold=threshold, threshold_mode=threshold_mode)


@st.cache_data(ttl=1800, show_spinner=False)
def get_backtest_dataset(
    stock_code: str,
    threshold: float,
    test_days: int = 120,
    threshold_mode: str = "manual",
) -> dict:
    """带缓存获取滚动回测结果。"""
    return rolling_backtest(
        stock_code,
        threshold=threshold,
        test_days=test_days,
        threshold_mode=threshold_mode,
    )


@st.cache_data(ttl=20, show_spinner=False)
def get_realtime_quote_dataset(stock_code: str) -> dict:
    """带短缓存获取实时行情快照。"""
    return fetch_realtime_quote(stock_code)


@st.cache_data(ttl=120, show_spinner=False)
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


def summarize_prediction_basis(item: dict) -> str:
    """把预测依据压缩成表格里容易扫读的一句话。"""
    direction = item.get("预测方向")
    support = [text for text in item.get("主要支撑信号", []) if text]
    risk = [text for text in item.get("主要风险信号", []) if text]
    if direction == "上涨":
        signals = support[:2] or [item.get("预测依据", "偏强信号相对占优")]
    elif direction == "下跌":
        signals = risk[:2] or [item.get("预测依据", "偏弱风险相对占优")]
    else:
        signals = ["多空信号接近"]
        if support:
            signals.append(support[0])
        if risk:
            signals.append(risk[0])
    return "，".join(str(signal) for signal in signals[:3] if signal)[:42]


def get_plotly_config() -> dict:
    """返回 Plotly 图表交互配置，支持滚轮缩放和模式栏操作。"""
    return {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        "toImageButtonOptions": {"format": "png", "filename": "stock_analysis_chart", "scale": 2},
    }


def update_run_status(status, label: str, state: str = "running", expanded: bool = True) -> None:
    """更新运行状态面板，避免状态提示失败影响主流程。"""
    try:
        status.update(label=label, state=state, expanded=expanded)
    except Exception:
        pass


def clear_run_status(status) -> None:
    """任务完成后移除临时状态面板，保持结果页面简洁。"""
    try:
        status.empty()
    except Exception:
        pass


def render_sidebar() -> tuple[str, str, int, list[str], dict]:
    """渲染左侧边栏，并返回当前功能、选中代码、历史范围和自选股。"""
    app_config = load_config()
    with st.sidebar:
        st.header("短线研究助手")
        selected_theme = st.radio(
            "界面主题",
            ["明亮", "深色"],
            horizontal=True,
            key="ui_theme",
            help="明亮模式为白底黑字；深色模式为黑底白字。系统会在当前浏览器中记住你的选择。",
        )
        persist_theme_preference(selected_theme)
        default_code = st.session_state.get("selected_stock", "600519")
        selected_code = st.text_input("股票代码", value=default_code, placeholder="例如 600519、000001、300750")
        days = st.selectbox("分析周期", [30, 60, 120, 250], index=3)
        threshold_choice = st.radio("阈值模式", ["自动（按波动调整）", "手动固定"], index=0)
        threshold_mode = "adaptive" if threshold_choice.startswith("自动") else "manual"
        if threshold_mode == "manual":
            threshold_percent = st.slider("固定预测阈值（%）", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
            st.caption(
                f"未来涨跌幅超过 +{threshold_percent:.1f}% 记为上涨，"
                f"低于 -{threshold_percent:.1f}% 记为下跌。"
            )
        else:
            threshold_percent = 1.0
            st.caption("根据个股近20日波动率和1/2/3日周期分别计算，范围限制在0.6%～4.0%。")
        feature = st.radio(
            "功能选择",
            ["个股技术分析", "1-3 日走势预测", "批量样本积累", "模型回测", "分析报告"],
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
        if not watchlist:
            st.caption("自选股列表：暂无股票。")
        else:
            with st.expander(f"自选股列表（{len(watchlist)}）", expanded=False):
                st.caption("点击股票代码可切换当前研究对象。")
                for code in watchlist:
                    col_select, col_delete = st.columns([3, 1])
                    if col_select.button(code, key=f"select_{code}", use_container_width=True):
                        st.session_state["selected_stock"] = code
                        st.rerun()
                    if col_delete.button("删除", key=f"delete_{code}", use_container_width=True):
                        ok, message = remove_from_watchlist(code)
                        if ok:
                            st.success(message)
                            st.rerun()
                        else:
                            st.warning(message)

        app_config["prediction_threshold"] = threshold_percent / 100
        app_config["threshold_mode"] = threshold_mode
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


def render_anomaly_panel(df: pd.DataFrame) -> None:
    """渲染异常波动检测结果。"""
    flags = detect_anomaly_flags(df)
    st.subheader("异常波动检测")
    if not flags:
        st.info("最近一个交易日未触发明显异常波动信号。")
        return
    st.dataframe(pd.DataFrame(flags), use_container_width=True, hide_index=True)
    st.caption("异常检测只提示客观量价变化，不代表未来走势判断，也不构成买卖建议。")


def render_support_resistance_panel(df: pd.DataFrame) -> None:
    """渲染支撑位和压力位分析。"""
    levels = estimate_support_resistance(df)
    st.subheader("支撑位与压力位")
    if not levels:
        st.info("历史数据不足，暂无法估算支撑位和压力位。")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("当前收盘价", format_metric_value(levels.get("current_price")))
    support_gap = levels.get("support_gap")
    resistance_gap = levels.get("resistance_gap")
    col2.metric(
        "短线支撑位",
        format_metric_value(levels.get("support")),
        "暂无" if support_gap is None else f"下方 {support_gap:.2%}",
    )
    col3.metric(
        "短线压力位",
        format_metric_value(levels.get("resistance")),
        "暂无" if resistance_gap is None else f"上方 {resistance_gap:.2%}",
    )

    dense_text = "、".join(f"{level:.2f}" for level in levels.get("dense_levels", [])) or "暂无"
    detail_rows = [
        {"项目": "近20日低点", "价位": f"{levels['near_20_low']:.2f}"},
        {"项目": "近20日高点", "价位": f"{levels['near_20_high']:.2f}"},
        {"项目": "近60日低点", "价位": f"{levels['near_60_low']:.2f}"},
        {"项目": "近60日高点", "价位": f"{levels['near_60_high']:.2f}"},
        {"项目": "成交密集区", "价位": dense_text},
    ]
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    st.caption(f"{levels.get('explanation', '')} 支撑/压力只是短线观察参考，不构成买卖建议。")


def render_prediction_precheck(df: pd.DataFrame) -> None:
    """在预测页提示异常波动对模型可靠性的影响。"""
    flags = detect_anomaly_flags(df)
    if not flags:
        return
    risk_flags = [item for item in flags if item.get("级别") == "风险"]
    title = "当前存在异常波动，模型预测可靠性可能下降。"
    detail = "；".join(item.get("类型", "") for item in flags[:4])
    if risk_flags:
        st.warning(f"{title} 触发项：{detail}。建议结合异常波动和支撑压力观察，不要只看概率。")
    else:
        st.info(f"当前有量价异动需要留意：{detail}。模型仍可参考，但短线波动可能放大。")


def build_ai_interpretation_context(code: str, result: dict) -> dict:
    """整理智能解读所需的结构化上下文。"""
    data = result.get("data")
    anomaly_flags = detect_anomaly_flags(data) if isinstance(data, pd.DataFrame) else []
    support_resistance = estimate_support_resistance(data) if isinstance(data, pd.DataFrame) else {}
    predictions = []
    for item in result.get("predictions", []):
        predictions.append(
            {
                "周期": item.get("周期"),
                "预测日期": item.get("预测日期"),
                "预测方向": item.get("预测方向"),
                "上涨概率": item.get("上涨概率"),
                "震荡概率": item.get("震荡概率"),
                "下跌概率": item.get("下跌概率"),
                "置信度": item.get("置信度"),
                "置信度说明": item.get("置信度说明"),
                "主要依据": summarize_prediction_basis(item),
                "风险提示": item.get("风险提示"),
            }
        )
    similar_patterns = result.get("similar_patterns", [])[:5]
    return {
        "股票代码": code,
        "最近交易日": result.get("last_trade_date"),
        "最近收盘价": result.get("last_close"),
        "预测结果": predictions,
        "异常波动": anomaly_flags,
        "支撑压力": support_resistance,
        "历史相似走势": similar_patterns,
        "模型说明": result.get("model_note"),
    }


def render_ai_interpretation_panel(code: str, result: dict) -> None:
    """渲染智能解读按钮和结果。"""
    st.subheader("智能解读")
    st.caption("点击后才会调用模型；解读只基于页面已有数据，不提供买卖建议。")
    if st.button("生成智能解读", type="primary"):
        context = build_ai_interpretation_context(code, result)
        try:
            with st.spinner("正在生成智能解读..."):
                interpretation = generate_stock_interpretation(context)
            st.markdown(interpretation)
        except AIInterpreterError as exc:
            st.warning(str(exc))
            st.code(
                "DASHSCOPE_API_KEY=你的API Key\n"
                "DASHSCOPE_MODEL=qwen3.7-plus\n"
                "DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                language="ini",
            )
            st.caption("请把以上内容保存到项目根目录 F:\\Codex\\project6\\.env，填入真实 API Key 后重启 Streamlit。")
        except Exception as exc:
            st.warning(f"智能解读暂不可用：{exc}")


def render_stock_detail(stock_code: str, days: int, show_export: bool = False, app_config: dict | None = None) -> None:
    """渲染个股详细分析页面，可选择显示报告导出按钮。"""
    app_config = app_config or load_config()
    st.header("个股详细分析" if not show_export else "分析报告导出")
    try:
        code = normalize_stock_code(stock_code)
    except DataFetchError as exc:
        st.error(str(exc))
        return

    run_status = st.status("正在获取历史行情和基础信息...", expanded=True)
    try:
        dataset = get_stock_dataset(code, days)
        update_run_status(run_status, "历史行情、技术指标和基础分析已完成。", state="running")
    except Exception as exc:
        update_run_status(run_status, "历史行情加载失败。", state="error")
        st.error(f"{code} 数据获取失败：{exc}")
        return

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
    trend_strength = int(analysis_result.get("趋势强度", 0) or 0)
    col4.metric(
        "趋势判断",
        analysis_result.get("趋势判断", "暂无"),
        delta=f"{trend_strength:+d} · {analysis_result.get('趋势置信度', '低')}置信度",
        help=analysis_result.get("趋势判断依据", "综合多周期价格、均线、波动率与趋势效率判断。"),
    )

    render_anomaly_panel(df)
    render_support_resistance_panel(df)

    try:
        update_run_status(run_status, "正在加载实时行情...", state="running")
        quote_started_at = perf_counter()
        quote = get_realtime_quote_dataset(code)
        quote_elapsed = perf_counter() - quote_started_at
        update_run_status(run_status, "实时行情已加载。", state="running")
        st.subheader("实时行情")
        realtime_cols = st.columns(5)
        realtime_cols[0].metric("实时最新价", format_metric_value(quote.get("latest_price")))
        realtime_cols[1].metric("实时涨跌幅", format_metric_value(quote.get("pct_chg"), "%"))
        realtime_cols[2].metric("今日最高", format_metric_value(quote.get("high")))
        realtime_cols[3].metric("今日最低", format_metric_value(quote.get("low")))
        realtime_cols[4].metric("实时成交额", f"{(quote.get('amount') or 0) / 100000000:.2f} 亿元")
        st.caption(
            f"实时来源：{quote.get('source')}，时间戳：{quote.get('timestamp', '暂无')}，"
            f"加载耗时：{quote_elapsed:.2f} 秒。"
        )
        show_minute_chart = st.toggle(
            "显示分时走势",
            value=False,
            key=f"show_realtime_minute_{code}",
            help="分时数据按需加载，避免拖慢技术分析主体。",
        )
        if show_minute_chart:
            minute_started_at = perf_counter()
            with st.spinner("正在加载分时走势..."):
                minute_df = get_realtime_minute_dataset(code)
            minute_elapsed = perf_counter() - minute_started_at
            st.caption(f"分时数据加载耗时：{minute_elapsed:.2f} 秒；缓存 120 秒。")
            st.plotly_chart(build_realtime_minute_figure(minute_df, quote), use_container_width=True, config=get_plotly_config())
        else:
            st.caption("分时图已改为按需加载；需要查看时打开上方开关。")
    except RealtimeQuoteError as exc:
        update_run_status(run_status, "实时行情暂不可用，继续展示历史分析。", state="running")
        st.warning(str(exc))
    except Exception as exc:
        update_run_status(run_status, "实时分时走势暂不可用，继续展示历史分析。", state="running")
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
    clear_run_status(run_status)


def render_short_term_prediction(
    stock_code: str,
    threshold: float,
    threshold_mode: str = "manual",
) -> None:
    """渲染未来 1-3 日走势预测页面。"""
    st.header("未来 1-3 日走势预测")
    try:
        code = normalize_stock_code(stock_code)
    except DataFetchError as exc:
        st.error(str(exc))
        return

    run_status = st.status("正在获取历史行情并训练短线模型...", expanded=True)
    try:
        result = get_prediction_dataset(code, threshold, threshold_mode)
        update_run_status(run_status, "短线模型训练和 1-3 日预测已完成。", state="running")
    except Exception as exc:
        update_run_status(run_status, "短线预测加载失败。", state="error")
        st.error(f"短线预测暂不可用：{exc}")
        return

    try:
        update_run_status(run_status, "正在更新复盘结果并应用校准...", state="running")
        update_prediction_outcomes(code, threshold, hist_df=result.get("data"))
        result = apply_feedback_calibration(code, result)
        update_run_status(run_status, "复盘更新和概率校准已完成。", state="running")
    except Exception:
        update_run_status(run_status, "复盘校准暂不可用，继续展示本次预测。", state="running")
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

    render_prediction_precheck(result.get("data"))

    st.subheader("实时行情对比（可选）")
    show_realtime_comparison = st.toggle(
        "加载实时行情对比",
        value=False,
        key=f"show_prediction_realtime_{code}",
        help="预测结果会优先显示；需要盘中对比时再加载实时行情。",
    )
    if show_realtime_comparison:
        try:
            update_run_status(run_status, "正在加载实时行情用于对比...", state="running")
            quote = get_realtime_quote_dataset(code)
            comparison = compare_realtime_with_prediction(quote, result)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("实时最新价", format_metric_value(quote.get("latest_price")))
            col2.metric("实时涨跌幅", format_metric_value(quote.get("pct_chg"), "%"))
            col3.metric("成交额", f"{(quote.get('amount') or 0) / 100000000:.2f} 亿元")
            col4.metric("预测 vs 实时", comparison["对比结论"])
            st.caption(
                f"实时来源：{quote.get('source')}，时间戳：{quote.get('timestamp', '暂无')}。"
                f"1日预测方向：{comparison['预测方向']}，实时盘中方向：{comparison['实时方向']}。{comparison['说明']}"
            )
            show_prediction_minute = st.toggle(
                "显示实时分时走势",
                value=False,
                key=f"show_prediction_minute_{code}",
            )
            if show_prediction_minute:
                update_run_status(run_status, "正在加载实时分时走势图...", state="running")
                minute_df = get_realtime_minute_dataset(code)
                st.plotly_chart(
                    build_realtime_minute_figure(minute_df, quote),
                    use_container_width=True,
                    config=get_plotly_config(),
                )
            update_run_status(run_status, "实时行情对比已完成。", state="running")
        except RealtimeQuoteError as exc:
            update_run_status(run_status, "实时行情暂不可用，继续展示预测结果。", state="running")
            st.warning(str(exc))
        except Exception as exc:
            update_run_status(run_status, "实时行情对比暂不可用，继续展示预测结果。", state="running")
            st.warning(f"实时行情对比暂不可用：{exc}")
    else:
        st.caption("实时对比和分时图已改为按需加载，不再阻塞预测结果。")

    rows = []
    for item in result["predictions"]:
        rows.append(
            {
                "预测日期": item.get("预测日期", ""),
                "周期": item["周期"],
                "预测走势": item["预测方向"],
                "判断阈值": f"±{item.get('判断阈值', threshold):.2%}",
                "上涨概率": f"{item['上涨概率']:.1%}",
                "震荡概率": f"{item['震荡概率']:.1%}",
                "下跌概率": f"{item['下跌概率']:.1%}",
                "预测涨跌幅区间": format_percent_interval(item.get("预测涨跌幅区间")),
                "预测价位区间": format_price_interval(item.get("预测价位区间")),
                "置信度": item["置信度"],
                "置信度说明": item.get("置信度说明", "暂无说明"),
                "主要依据": summarize_prediction_basis(item),
            }
        )
    st.caption(
        f"最近一个已获取交易日：{result.get('last_trade_date', '暂无')}。"
        f"预测日期来源：{result.get('calendar_source', '暂无')}。"
    )
    if result.get("threshold_mode") == "adaptive":
        st.info("当前使用波动自适应阈值：每个周期和每个历史时点都按当时近20日波动计算，不使用未来数据。")
    else:
        st.info(f"当前使用手动固定阈值：±{threshold:.2%}。")
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

    similar_patterns = result.get("similar_patterns", [])
    if similar_patterns:
        with st.expander("历史相似走势参考", expanded=False):
            similar_rows = []
            for item in similar_patterns:
                similar_rows.append(
                    {
                        "相似日期": item.get("日期", ""),
                        "相似度": f"{item.get('相似度', 0):.0%}",
                        "当日收盘": f"{item.get('当日收盘', 0):.2f}",
                        "1日后": f"{item.get('1日后走势', '')}（{item.get('1日后涨跌幅', 0):.2%}）",
                        "2日后": f"{item.get('2日后走势', '')}（{item.get('2日后涨跌幅', 0):.2%}）",
                        "3日后": f"{item.get('3日后走势', '')}（{item.get('3日后涨跌幅', 0):.2%}）",
                    }
                )
            st.dataframe(pd.DataFrame(similar_rows), use_container_width=True, hide_index=True)
            st.caption(
                "相似走势基于当前量价、均线、动能、波动率和K线形态特征匹配历史样本。"
                "它只是辅助参考，不参与买卖建议，也不保证当前会重复历史走势。"
            )

    render_ai_interpretation_panel(code, result)

    try:
        update_run_status(run_status, "正在记录本次预测并汇总复盘样本...", state="running")
        record_predictions(code, result)
        update_prediction_outcomes(code, threshold, hist_df=result.get("data"))
        feedback_summary = summarize_feedback(code)
        update_run_status(run_status, "预测记录和复盘汇总已完成。", state="running")
        st.subheader("预测复盘与自我修正")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("已记录预测", feedback_summary["total"])
        col2.metric("已完成复盘", feedback_summary["evaluated"])
        col3.metric("等待结果", feedback_summary["pending"])
        accuracy_text = "暂无" if feedback_summary["accuracy"] is None else f"{feedback_summary['accuracy']:.1%}"
        col4.metric("综合历史准确率", accuracy_text)

        if feedback_summary.get("by_source"):
            source_df = pd.DataFrame(feedback_summary["by_source"])
            source_df["准确率"] = source_df["准确率"].map(lambda value: "暂无" if pd.isna(value) else f"{value:.1%}")
            source_df["校准权重"] = source_df["校准权重"].map(lambda value: f"{value:.2f}")
            with st.expander("查看样本来源", expanded=False):
                st.dataframe(source_df, use_container_width=True, hide_index=True)

        if feedback_summary.get("by_label_version"):
            version_df = pd.DataFrame(feedback_summary["by_label_version"])
            version_df["准确率"] = version_df["准确率"].map(lambda value: "暂无" if pd.isna(value) else f"{value:.1%}")
            with st.expander("查看标签口径", expanded=False):
                st.dataframe(version_df, use_container_width=True, hide_index=True)
                st.caption("固定阈值与波动自适应阈值分别统计；概率校准只使用同一标签版本的已复盘样本。")

        if feedback_summary["by_horizon"]:
            horizon_df = pd.DataFrame(feedback_summary["by_horizon"])
            horizon_df["准确率"] = horizon_df["准确率"].map(lambda value: f"{value:.1%}")
            st.write("**分周期复盘表现**")
            st.dataframe(horizon_df, use_container_width=True, hide_index=True)

        if feedback_summary["bias"] or feedback_summary.get("error_reasons") or feedback_summary["mistakes"]:
            with st.expander("展开查看复盘细节", expanded=False):
                if feedback_summary["bias"]:
                    st.write("**常见错判类型**")
                    st.dataframe(pd.DataFrame(feedback_summary["bias"]), use_container_width=True, hide_index=True)

                if feedback_summary.get("error_reasons"):
                    st.write("**常见错误原因**")
                    st.dataframe(pd.DataFrame(feedback_summary["error_reasons"]), use_container_width=True, hide_index=True)

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
                                "样本来源": item.get("source", "日常预测"),
                                "错误原因": item.get("error_reason", "暂未归类"),
                                "模型": item.get("model", ""),
                            }
                        )
                    st.write("**最近预测错误记录**")
                    st.dataframe(pd.DataFrame(mistake_rows), use_container_width=True, hide_index=True)

        export_df = build_feedback_export_table(code)
        if not export_df.empty:
            st.download_button(
                "导出复盘样本（CSV）",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{code}_prediction_feedback.csv",
                mime="text/csv",
            )

        st.info(feedback_summary["summary"])
        st.caption(
            "复盘机制会在后续交易日行情可用后自动核对结果；样本越多，错误总结越有参考价值，但不会保证未来预测一定准确。"
        )
    except Exception as exc:
        update_run_status(run_status, "复盘模块暂不可用，预测结果已展示。", state="running")
        st.warning(f"预测复盘暂不可用：{exc}")

    st.subheader("预测依据")
    for item in result["predictions"]:
        with st.expander(f"{item['周期']}预测依据：{item['预测方向']}"):
            st.write(f"**模型判断：** 短线{item['预测方向']}概率相对较高。")
            st.write(f"**置信度说明：** {item.get('置信度说明', '暂无说明')}")
            st.write(f"**模型来源：** {item.get('模型', '暂无')}")
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
    clear_run_status(run_status)


def render_batch_prediction_collector(
    watchlist: list[str],
    threshold: float,
    threshold_mode: str = "manual",
) -> None:
    """批量生成自选股预测并记录复盘样本，避免加载图表和实时行情。"""
    st.header("批量样本积累")
    st.caption("该入口只生成 1-3 日预测并写入复盘记录，不加载实时行情、分时图和技术图表。")

    source = st.radio("股票来源", ["自选股", "手动输入"], horizontal=True)
    if source == "自选股":
        if not watchlist:
            st.info("自选股列表为空，请先在左侧边栏添加股票，或切换为手动输入。")
            return
        selected_codes = st.multiselect("选择要批量处理的股票", watchlist, default=watchlist[: min(5, len(watchlist))])
    else:
        code_text = st.text_area("输入股票代码", value="600519, 000001, 300750", help="用英文逗号、中文逗号或换行分隔。")
        selected_codes, parse_errors = parse_stock_codes(code_text)
        for error in parse_errors:
            st.warning(error)

    max_count = st.slider("单次最多处理数量", min_value=1, max_value=30, value=10, step=1)
    skip_today_recorded = st.checkbox(
        "跳过今日已完整记录的股票",
        value=True,
        help="如果本地复盘记录里今天已经有该股票 1日、2日、3日 三条预测，就不再重新训练和写入。",
    )
    selected_codes = selected_codes[:max_count]
    st.info(
        "批量处理会逐只请求历史行情并训练模型，速度取决于 AKShare 数据源和网络状态。"
        "建议先从 5-10 只开始，确认稳定后再增加数量。"
    )

    if not selected_codes:
        st.warning("请至少选择或输入一只有效股票。")
        return

    if not st.button(f"开始积累 {len(selected_codes)} 只股票样本", type="primary"):
        return

    rows = []
    errors = []
    progress = st.progress(0, text="准备开始批量预测...")
    run_status = st.status("批量任务准备开始...", expanded=True)
    for index, code in enumerate(selected_codes, start=1):
        progress.progress((index - 1) / len(selected_codes), text=f"正在处理 {code}（{index}/{len(selected_codes)}）...")
        try:
            if skip_today_recorded:
                record_status = get_today_prediction_record_status(code)
                if record_status["has_full_today_record"]:
                    update_run_status(run_status, f"{code}：今日已完整记录，跳过训练。", state="running")
                    summary = summarize_feedback(code)
                    rows.append(
                        {
                            "股票代码": code,
                            "处理状态": "已跳过",
                            "最近交易日": record_status.get("latest_base_date", ""),
                            "次日预测日期": "",
                            "次日预测": "今日已记录",
                            "次日置信度": "",
                            "已记录": summary.get("total", 0),
                            "已复盘": summary.get("evaluated", 0),
                            "等待复盘": summary.get("pending", 0),
                            "历史准确率": "暂无" if summary.get("accuracy") is None else f"{summary['accuracy']:.1%}",
                        }
                    )
                    progress.progress(index / len(selected_codes), text=f"已完成 {index}/{len(selected_codes)}")
                    continue

            update_run_status(run_status, f"{code}：正在获取历史行情并训练模型...", state="running")
            result = get_prediction_dataset(code, threshold, threshold_mode)
            update_run_status(run_status, f"{code}：正在复盘已有样本并应用校准...", state="running")
            update_prediction_outcomes(code, threshold, hist_df=result.get("data"))
            result = apply_feedback_calibration(code, result)
            update_run_status(run_status, f"{code}：正在写入本次预测记录...", state="running")
            record_predictions(code, result)
            update_prediction_outcomes(code, threshold, hist_df=result.get("data"))
            summary = summarize_feedback(code)
            first_prediction = next((item for item in result.get("predictions", []) if item.get("周期") == "1日"), {})
            rows.append(
                {
                    "股票代码": code,
                    "处理状态": "已处理",
                    "最近交易日": result.get("last_trade_date", ""),
                    "次日预测日期": first_prediction.get("预测日期", ""),
                    "次日预测": first_prediction.get("预测方向", ""),
                    "次日置信度": first_prediction.get("置信度", ""),
                    "次日判断阈值": f"±{first_prediction.get('判断阈值', threshold):.2%}",
                    "已记录": summary.get("total", 0),
                    "已复盘": summary.get("evaluated", 0),
                    "等待复盘": summary.get("pending", 0),
                    "历史准确率": "暂无" if summary.get("accuracy") is None else f"{summary['accuracy']:.1%}",
                }
            )
        except Exception as exc:
            errors.append(f"{code} 处理失败：{exc}")
        progress.progress(index / len(selected_codes), text=f"已完成 {index}/{len(selected_codes)}")
    progress.empty()
    update_run_status(run_status, "批量任务已完成。", state="complete", expanded=False)

    if rows:
        result_df = pd.DataFrame(rows)
        processed_count = int((result_df["处理状态"] == "已处理").sum()) if "处理状态" in result_df.columns else len(result_df)
        skipped_count = int((result_df["处理状态"] == "已跳过").sum()) if "处理状态" in result_df.columns else 0
        st.success(f"本次处理 {processed_count} 只股票，跳过 {skipped_count} 只今日已记录股票。")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.download_button(
            "导出本次批量结果（CSV）",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="batch_prediction_collection.csv",
            mime="text/csv",
        )
    if errors:
        st.warning("部分股票处理失败，可能是数据接口暂时不可用或网络超时。")
        for error in errors:
            st.caption(error)

    st.caption("批量入口用于积累复盘样本，不构成任何投资建议；样本数量增加只会让校准更有依据，不保证未来预测一定准确。")


def render_model_backtest(
    stock_code: str,
    threshold: float,
    threshold_mode: str = "manual",
) -> None:
    """渲染模型滚动回测页面。"""
    st.header("模型回测")
    try:
        code = normalize_stock_code(stock_code)
    except DataFetchError as exc:
        st.error(str(exc))
        return

    test_days = st.select_slider(
        "历史回放范围",
        options=[60, 120, 180, 250],
        value=120,
        help="每个交易日会分别生成 1日、2日、3日 三条历史推演结果。范围越大，计算时间越长。",
    )
    st.caption(f"本次最多可生成约 {test_days * 3} 条历史推演样本；重复日期会自动跳过。")
    st.caption(
        "阈值口径：波动自适应（按每个历史时点计算）。"
        if threshold_mode == "adaptive"
        else f"阈值口径：手动固定 ±{threshold:.2%}。"
    )
    st.caption("为控制本地计算耗时，系统每天生成推演结果，但模型每 20 个交易日重新训练一次。")

    run_status = st.status("正在获取历史行情并执行滚动回测...", expanded=True)
    try:
        result = get_backtest_dataset(code, threshold, test_days=test_days, threshold_mode=threshold_mode)
        update_run_status(run_status, "滚动回测已完成，正在整理统计结果...", state="running")
    except Exception as exc:
        update_run_status(run_status, "模型回测失败。", state="error")
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

    st.subheader("历史样本回放")
    st.write(
        "可把本次滚动回测结果写入复盘系统，用已有历史走势快速扩充样本。"
        "这些样本会标记为“历史滚动推演”，并只按日常预测样本的 0.35 权重参与后续校准。"
    )
    if st.button(f"写入本次 {len(result['results'])} 条历史推演样本", type="primary"):
        replay_result = record_historical_replay(code, result["results"], threshold)
        if replay_result["added"]:
            st.success(
                f"已新增 {replay_result['added']} 条历史推演样本，"
                f"折合校准样本权重约 {replay_result['effective_added']:.1f}；"
                f"另有 {replay_result['skipped']} 条重复或无效记录已跳过。"
            )
        else:
            st.info(f"本次没有新增样本，{replay_result['skipped']} 条记录均已存在或无效。")
    st.caption("历史回放能加快发现模型偏差，但它不是新的市场信息，不能替代后续真实预测复盘。")

    backtest_results = result["results"]
    if backtest_results.empty or not {"actual", "predicted"}.issubset(backtest_results.columns):
        mistakes = pd.DataFrame()
    else:
        mistakes = backtest_results[backtest_results["actual"] != backtest_results["predicted"]].copy()
    st.subheader("错误样本复盘表")
    if mistakes.empty:
        st.success("本次回测样本中暂未发现错判记录。")
    else:
        mistakes["date"] = pd.to_datetime(mistakes["date"]).dt.strftime("%Y-%m-%d")
        mistakes["错判类型"] = mistakes["predicted"] + " -> " + mistakes["actual"]
        type_options = ["全部"] + sorted(mistakes["错判类型"].dropna().unique().tolist())
        selected_type = st.selectbox("错判类型筛选", type_options)
        display_df = mistakes if selected_type == "全部" else mistakes[mistakes["错判类型"] == selected_type]
        display_df = display_df.sort_values("date", ascending=False).rename(
            columns={
                "date": "预测日期",
                "horizon": "周期（日）",
                "predicted": "预测走势",
                "actual": "实际走势",
            }
        )
        st.dataframe(
            display_df[["预测日期", "周期（日）", "预测走势", "实际走势", "错判类型"]].head(80),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("错误样本按日期倒序展示，便于结合当时行情环境人工复盘。")

    st.info(result["note"])
    st.caption("回测结果只说明历史表现，不代表未来一定准确。")
    clear_run_status(run_status)


def main():
    """Streamlit 应用入口，负责看板导航和页面分发。"""
    st.set_page_config(page_title="A股短线研究助手", layout="wide")
    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = get_saved_theme()
    apply_custom_theme(st.session_state["ui_theme"])
    st.title("短线研究工作台")
    st.markdown(
        """
        <div class="terminal-header">
            <span><span class="terminal-dot"></span><strong>本地运行</strong></span>
            <span>技术分析 / 概率预测 / 滚动回测 / 复盘样本</span>
            <span>不含自动交易与买卖建议</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    feature, selected_code, days, watchlist, app_config = render_sidebar()
    threshold = app_config.get("prediction_threshold", 0.01)
    threshold_mode = app_config.get("threshold_mode", "adaptive")

    if feature == "个股技术分析":
        render_stock_detail(selected_code, days, app_config=app_config)
    elif feature == "1-3 日走势预测":
        render_short_term_prediction(selected_code, threshold, threshold_mode)
    elif feature == "批量样本积累":
        render_batch_prediction_collector(watchlist, threshold, threshold_mode)
    elif feature == "模型回测":
        render_model_backtest(selected_code, threshold, threshold_mode)
    elif feature == "分析报告":
        render_stock_detail(selected_code, days, show_export=True, app_config=app_config)


if __name__ == "__main__":
    main()
