from datetime import date, timedelta
from math import sqrt

import akshare as ak
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_fetch import DataFetchError, normalize_stock_code


LABELS = ["上涨", "震荡", "下跌"]
ADAPTIVE_THRESHOLD_VERSION = "volatility_20d_v1"
ADAPTIVE_THRESHOLD_MIN = 0.006
ADAPTIVE_THRESHOLD_MAX = 0.04
ADAPTIVE_THRESHOLD_FACTOR = 0.80
FEATURE_COLUMNS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "dev_ma5",
    "dev_ma10",
    "dev_ma20",
    "dev_ma60",
    "MA5",
    "MA10",
    "MA20",
    "MA60",
    "ma5_slope",
    "ma10_slope",
    "ma20_slope",
    "ma_bull",
    "ma_bear",
    "DIF",
    "DEA",
    "MACD",
    "macd_change",
    "RSI6",
    "RSI12",
    "RSI24",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "amplitude",
    "high_low_range",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_up",
    "volume_down",
    "shrink_up",
    "shrink_down",
    "body_size",
    "upper_shadow",
    "lower_shadow",
    "bullish_candle",
    "bearish_candle",
    "gap_up",
    "gap_down",
    "close_position",
    "volume_change_1d",
    "volume_change_5d",
    "consecutive_up_3d",
    "consecutive_down_3d",
    "new_high_20d",
    "new_low_20d",
]


def fetch_trade_calendar() -> pd.Series:
    """使用 AKShare 获取 A 股交易日历。"""
    calendar_df = ak.tool_trade_date_hist_sina()
    if calendar_df is None or calendar_df.empty or "trade_date" not in calendar_df.columns:
        raise DataFetchError("A 股交易日历接口返回为空。")
    return pd.to_datetime(calendar_df["trade_date"], errors="coerce").dropna().sort_values().reset_index(drop=True)


def next_trade_dates(last_date, count: int = 3) -> tuple[list[pd.Timestamp], str]:
    """根据 A 股交易日历生成后续交易日，接口失败时退回跳过周末方案。"""
    last_ts = pd.to_datetime(last_date)
    try:
        calendar = fetch_trade_calendar()
        dates = calendar[calendar > last_ts].head(count).tolist()
        if len(dates) >= count:
            return dates, "A股交易日历"
    except Exception:
        pass

    current = pd.to_datetime(last_date)
    dates = []
    while len(dates) < count:
        current = current + pd.Timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current)
    return dates, "周末过滤备用规则"


def fetch_prediction_history(stock_code: str, years: int = 3) -> pd.DataFrame:
    """使用 AKShare 获取至少最近三年的前复权日 K 数据。"""
    code = normalize_stock_code(stock_code)
    end = date.today()
    start = end - timedelta(days=365 * years + 120)
    try:
        raw_df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
    except Exception as exc:
        try:
            market = "sh" if code.startswith(("6", "9")) else "sz"
            raw_df = ak.stock_zh_a_daily(
                symbol=f"{market}{code}",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq",
            )
        except Exception as fallback_exc:
            raise DataFetchError("前复权历史行情获取失败，请稍后重试或检查网络代理。") from fallback_exc

    if raw_df is None or raw_df.empty:
        raise DataFetchError("前复权历史行情为空，无法进行短线预测。")
    df = raw_df.reset_index() if "date" not in raw_df.columns and raw_df.index.name == "date" else raw_df.copy()
    df = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
        }
    ).copy()
    required = ["date", "open", "close", "high", "low", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataFetchError(f"预测行情数据缺少必要字段：{', '.join(missing)}。")
    df = df[required + [col for col in ["amount", "pct_chg"] if col in df.columns]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in [c for c in df.columns if c != "date"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise DataFetchError("预测行情清洗后为空。")
    return df


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """计算指定周期 RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """基于历史 K 线构造短线预测特征。"""
    data = df.copy()
    close = data["close"]
    for window in [5, 10, 20, 60]:
        data[f"MA{window}"] = close.rolling(window).mean()
        data[f"dev_ma{window}"] = close / data[f"MA{window}"] - 1
    for window in [1, 3, 5, 10, 20]:
        data[f"ret_{window}d"] = close / close.shift(window) - 1

    data["ma5_slope"] = data["MA5"].pct_change(3)
    data["ma10_slope"] = data["MA10"].pct_change(3)
    data["ma20_slope"] = data["MA20"].pct_change(5)
    data["ma_bull"] = ((data["MA5"] > data["MA10"]) & (data["MA10"] > data["MA20"])).astype(int)
    data["ma_bear"] = ((data["MA5"] < data["MA10"]) & (data["MA10"] < data["MA20"])).astype(int)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["DIF"] = ema12 - ema26
    data["DEA"] = data["DIF"].ewm(span=9, adjust=False).mean()
    data["MACD"] = (data["DIF"] - data["DEA"]) * 2
    data["macd_change"] = data["MACD"].diff()
    data["RSI6"] = _rsi(close, 6)
    data["RSI12"] = _rsi(close, 12)
    data["RSI24"] = _rsi(close, 24)

    returns = close.pct_change()
    for window in [5, 10, 20]:
        data[f"volatility_{window}d"] = returns.rolling(window).std()
    data["amplitude"] = (data["high"] - data["low"]) / close
    data["high_low_range"] = data["high"] / data["low"] - 1

    data["volume_ratio_5d"] = data["volume"] / data["volume"].rolling(5).mean()
    data["volume_ratio_20d"] = data["volume"] / data["volume"].rolling(20).mean()
    data["volume_up"] = ((data["close"] > data["open"]) & (data["volume_ratio_20d"] > 1.2)).astype(int)
    data["volume_down"] = ((data["close"] < data["open"]) & (data["volume_ratio_20d"] > 1.2)).astype(int)
    data["shrink_up"] = ((data["close"] > data["open"]) & (data["volume_ratio_20d"] < 0.8)).astype(int)
    data["shrink_down"] = ((data["close"] < data["open"]) & (data["volume_ratio_20d"] < 0.8)).astype(int)

    data["body_size"] = (data["close"] - data["open"]).abs() / close
    data["upper_shadow"] = (data["high"] - data[["open", "close"]].max(axis=1)) / close
    data["lower_shadow"] = (data[["open", "close"]].min(axis=1) - data["low"]) / close
    data["bullish_candle"] = (data["close"] > data["open"]).astype(int)
    data["bearish_candle"] = (data["close"] < data["open"]).astype(int)
    data["gap_up"] = (data["open"] > data["high"].shift(1)).astype(int)
    data["gap_down"] = (data["open"] < data["low"].shift(1)).astype(int)
    intraday_range = (data["high"] - data["low"]).replace(0, pd.NA)
    data["close_position"] = (data["close"] - data["low"]) / intraday_range
    data["volume_change_1d"] = data["volume"].pct_change(1)
    data["volume_change_5d"] = data["volume"] / data["volume"].rolling(5).mean() - 1
    up_days = (data["close"].diff() > 0).astype(int)
    down_days = (data["close"].diff() < 0).astype(int)
    data["consecutive_up_3d"] = up_days.rolling(3).sum()
    data["consecutive_down_3d"] = down_days.rolling(3).sum()
    data["new_high_20d"] = (data["close"] >= data["close"].rolling(20).max()).astype(int)
    data["new_low_20d"] = (data["close"] <= data["close"].rolling(20).min()).astype(int)
    return data


def _label_threshold_series(
    data: pd.DataFrame,
    horizon: int,
    threshold: float,
    threshold_mode: str,
) -> pd.Series:
    """生成每个历史时点可用的标签阈值，不使用未来数据。"""
    fallback = max(ADAPTIVE_THRESHOLD_MIN, min(ADAPTIVE_THRESHOLD_MAX, float(threshold)))
    if threshold_mode != "adaptive":
        return pd.Series(float(threshold), index=data.index, dtype="float64")

    if "volatility_20d" in data.columns:
        daily_volatility = pd.to_numeric(data["volatility_20d"], errors="coerce")
    else:
        daily_volatility = pd.to_numeric(data["close"], errors="coerce").pct_change().rolling(20).std()
    adaptive = daily_volatility * sqrt(horizon) * ADAPTIVE_THRESHOLD_FACTOR
    return adaptive.clip(ADAPTIVE_THRESHOLD_MIN, ADAPTIVE_THRESHOLD_MAX).fillna(fallback)


def add_prediction_labels(
    df: pd.DataFrame,
    threshold: float = 0.01,
    threshold_mode: str = "manual",
) -> pd.DataFrame:
    """构造未来 1、2、3 日分类标签，支持按当时波动率自适应。"""
    data = df.copy()
    for horizon in [1, 2, 3]:
        future_return = data["close"].shift(-horizon) / data["close"] - 1
        label_threshold = _label_threshold_series(data, horizon, threshold, threshold_mode)
        data[f"future_return_{horizon}d"] = future_return
        data[f"label_threshold_{horizon}d"] = label_threshold
        labels = pd.Series("震荡", index=data.index, dtype="object")
        labels.loc[future_return > label_threshold] = "上涨"
        labels.loc[future_return < -label_threshold] = "下跌"
        data[f"label_{horizon}d"] = labels
        data.loc[future_return.isna(), f"label_{horizon}d"] = pd.NA
    return data


def _quantile_range(values: pd.Series, low: float, high: float, fallback: tuple[float, float]) -> tuple[float, float]:
    """从历史收益率序列中提取稳健分位区间，样本不足时使用备用值。"""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 8:
        return fallback
    left = float(clean.quantile(low))
    right = float(clean.quantile(high))
    return (min(left, right), max(left, right))


def estimate_return_price_ranges(data: pd.DataFrame, last_close: float, horizon: int, threshold: float) -> dict:
    """估算指定周期的上涨、下跌和震荡收益率区间及对应价格区间。"""
    return_col = f"future_return_{horizon}d"
    if return_col not in data.columns or not last_close:
        return {}

    returns = pd.to_numeric(data[return_col], errors="coerce").dropna().tail(500)
    recent_vol = float(pd.to_numeric(data["ret_1d"], errors="coerce").dropna().tail(60).std() or 0.015)
    base_move = max(threshold, min(0.08, recent_vol * (horizon ** 0.5)))

    up_fallback = (threshold, max(threshold * 1.6, base_move * 1.4))
    down_fallback = (-max(threshold * 1.6, base_move * 1.4), -threshold)
    flat_fallback = (-threshold, threshold)

    up_returns = returns[returns > threshold]
    if len(up_returns) < 8:
        up_returns = returns[returns > 0]
    down_returns = returns[returns < -threshold]
    if len(down_returns) < 8:
        down_returns = returns[returns < 0]
    flat_returns = returns[(returns >= -threshold) & (returns <= threshold)]

    up_low, up_high = _quantile_range(up_returns, 0.25, 0.75, up_fallback)
    down_low, down_high = _quantile_range(down_returns, 0.25, 0.75, down_fallback)
    flat_low, flat_high = _quantile_range(flat_returns, 0.2, 0.8, flat_fallback)

    up_low = max(0.001, up_low)
    up_high = max(up_low, up_high)
    down_low = min(down_low, down_high)
    down_high = min(-0.001, down_high)

    return {
        "基准价": float(last_close),
        "上涨幅度区间": (up_low, up_high),
        "上涨价位区间": (last_close * (1 + up_low), last_close * (1 + up_high)),
        "下跌幅度区间": (down_low, down_high),
        "下跌价位区间": (last_close * (1 + down_low), last_close * (1 + down_high)),
        "震荡幅度区间": (flat_low, flat_high),
        "震荡价位区间": (last_close * (1 + flat_low), last_close * (1 + flat_high)),
        "区间说明": f"基于最近历史 {horizon} 日收益率分布估算，样本量 {len(returns)}，不是确定目标价。",
    }


def _attach_range_estimate(result: dict, data: pd.DataFrame, latest_feature: pd.Series, horizon: int, threshold: float) -> dict:
    """把涨跌幅和价位区间补充到预测结果中。"""
    ranges = estimate_return_price_ranges(data, float(latest_feature["close"]), horizon, threshold)
    if not ranges:
        return result
    direction = result.get("预测方向")
    if direction == "上涨":
        result["预测涨跌幅区间"] = ranges["上涨幅度区间"]
        result["预测价位区间"] = ranges["上涨价位区间"]
    elif direction == "下跌":
        result["预测涨跌幅区间"] = ranges["下跌幅度区间"]
        result["预测价位区间"] = ranges["下跌价位区间"]
    else:
        result["预测涨跌幅区间"] = ranges["震荡幅度区间"]
        result["预测价位区间"] = ranges["震荡价位区间"]
    result["区间估算"] = ranges
    return result


def _label_from_threshold(value: float, threshold: float) -> str:
    """根据阈值把收益率转换为上涨、震荡、下跌。"""
    if value > threshold:
        return "上涨"
    if value < -threshold:
        return "下跌"
    return "震荡"


def find_similar_patterns(data: pd.DataFrame, threshold: float, limit: int = 8) -> list[dict]:
    """查找历史上与当前技术特征相似的样本，并给出随后 1-3 日实际走势。"""
    usable = data.dropna(subset=FEATURE_COLUMNS + ["date", "close"]).copy()
    if len(usable) < 90:
        return []

    latest = usable.iloc[-1]
    history = usable.iloc[:-3].copy()
    if history.empty:
        return []

    feature_median = history[FEATURE_COLUMNS].median(numeric_only=True)
    feature_std = history[FEATURE_COLUMNS].std(numeric_only=True).replace(0, 1)
    latest_vector = ((latest[FEATURE_COLUMNS] - feature_median) / feature_std).astype(float)
    history_vectors = ((history[FEATURE_COLUMNS] - feature_median) / feature_std).astype(float)
    distances = ((history_vectors - latest_vector) ** 2).mean(axis=1) ** 0.5

    candidates = history.assign(_distance=distances).dropna(subset=["_distance"]).nsmallest(limit, "_distance")
    rows = []
    for idx, row in candidates.iterrows():
        item = {
            "日期": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
            "相似度": max(0.0, min(1.0, 1 / (1 + float(row["_distance"])))),
            "当日收盘": float(row["close"]),
        }
        for horizon in [1, 2, 3]:
            future_idx = idx + horizon
            if future_idx >= len(data) or pd.isna(data.iloc[future_idx].get("close")):
                continue
            future_return = float(data.iloc[future_idx]["close"] / row["close"] - 1)
            item[f"{horizon}日后涨跌幅"] = future_return
            historical_label = row.get(f"label_{horizon}d")
            item[f"{horizon}日后走势"] = (
                str(historical_label)
                if pd.notna(historical_label)
                else _label_from_threshold(future_return, threshold)
            )
        rows.append(item)
    return rows


def _rule_prediction(feature_row: pd.Series, horizon: int, low_reliability: bool = False, target_date=None) -> dict:
    """使用规则模型生成备用预测。"""
    probs, strong, neutral, weak = _rule_probabilities(feature_row, horizon)
    direction = max(probs, key=probs.get)
    confidence = "低" if low_reliability or max(probs.values()) < 0.45 else "中"
    model_label = "规则模型"
    return {
        "周期": f"{horizon}日",
        "预测日期": pd.to_datetime(target_date).strftime("%Y-%m-%d") if target_date is not None else "",
        "预测方向": direction,
        "上涨概率": probs["上涨"],
        "震荡概率": probs["震荡"],
        "下跌概率": probs["下跌"],
        "置信度": confidence,
        "置信度说明": _confidence_reason(probs, None, low_reliability, model_label),
        "预测依据": "；".join((strong + neutral + weak)[:6]) or "指标信号不充分，按规则模型中性处理。",
        "主要支撑信号": strong,
        "主要风险信号": weak,
        "风险提示": "历史数据不足，预测可靠性较低。" if low_reliability else "短线预测受市场情绪、消息面和流动性影响较大。",
        "模型": model_label,
    }


def _rule_probabilities(feature_row: pd.Series, horizon: int) -> tuple[dict, list[str], list[str], list[str]]:
    """根据技术规则输出概率和信号列表。"""
    strong = []
    weak = []
    neutral = []
    close_above_ma = all(feature_row.get(f"dev_ma{w}", 0) > 0 for w in [5, 10, 20])
    close_below_ma = all(feature_row.get(f"dev_ma{w}", 0) < 0 for w in [5, 10, 20])
    if close_above_ma:
        strong.append("收盘价站上 MA5、MA10、MA20")
    if close_below_ma:
        weak.append("收盘价跌破 MA5、MA10、MA20")
    if feature_row.get("ma_bull", 0) == 1:
        strong.append("MA5 > MA10 > MA20")
    if feature_row.get("ma_bear", 0) == 1:
        weak.append("MA5 < MA10 < MA20")
    if feature_row.get("DIF", 0) > feature_row.get("DEA", 0):
        strong.append("DIF 位于 DEA 上方")
    elif feature_row.get("DIF", 0) < feature_row.get("DEA", 0):
        weak.append("DIF 位于 DEA 下方")
    if feature_row.get("macd_change", 0) > 0:
        strong.append("MACD 柱状图改善")
    elif feature_row.get("macd_change", 0) < 0:
        weak.append("MACD 柱状图走弱")
    rsi = feature_row.get("RSI12", 50)
    if 50 <= rsi <= 70:
        strong.append("RSI 位于 50 到 70")
    elif rsi < 45:
        weak.append("RSI 低于 45")
    elif 45 <= rsi <= 55:
        neutral.append("RSI 位于 45 到 55")
    if feature_row.get("volume_up", 0) == 1:
        strong.append("放量上涨")
    if feature_row.get("volume_down", 0) == 1:
        weak.append("放量下跌")
    if abs(feature_row.get("dev_ma20", 0)) < 0.015:
        neutral.append("股价围绕 MA20 波动")
    if abs(feature_row.get("MACD", 0)) < 0.02:
        neutral.append("MACD 靠近 0 轴")

    score = len(strong) - len(weak)
    if score >= 2:
        probs = {"上涨": 0.52, "震荡": 0.31, "下跌": 0.17}
    elif score <= -2:
        probs = {"上涨": 0.17, "震荡": 0.31, "下跌": 0.52}
    else:
        probs = {"上涨": 0.27, "震荡": 0.48, "下跌": 0.25}
    decay = {1: 1.0, 2: 0.88, 3: 0.78}[horizon]
    probs = {k: 1 / 3 + (v - 1 / 3) * decay for k, v in probs.items()}
    return _normalize_probs(probs), strong, neutral, weak


def _normalize_probs(probs: dict) -> dict:
    """归一化三分类概率，确保三个方向相加为 1。"""
    clean = {label: max(0.0, float(probs.get(label, 0.0))) for label in LABELS}
    total = sum(clean.values())
    if total <= 0:
        return {label: 1 / 3 for label in LABELS}
    return {label: clean[label] / total for label in LABELS}


def _recent_label_probs(train_df: pd.DataFrame, label_col: str, window: int = 120) -> dict:
    """计算近期标签分布，用于校准模型概率。"""
    recent = train_df[label_col].dropna().tail(window)
    if recent.empty:
        return {label: 1 / 3 for label in LABELS}
    counts = recent.value_counts(normalize=True)
    smoothed = {label: float(counts.get(label, 0.0)) + 0.08 for label in LABELS}
    return _normalize_probs(smoothed)


def _blend_probabilities(model_probs: dict, recent_probs: dict, rule_probs: dict, model_score: float | None) -> dict:
    """融合模型概率、近期分布和规则概率，降低单模型过拟合影响。"""
    if model_score is None:
        model_weight = 0.55
    elif model_score >= 0.5:
        model_weight = 0.68
    elif model_score >= 0.42:
        model_weight = 0.58
    else:
        model_weight = 0.45
    recent_weight = 0.2
    rule_weight = 1 - model_weight - recent_weight
    blended = {
        label: model_probs[label] * model_weight + recent_probs[label] * recent_weight + rule_probs[label] * rule_weight
        for label in LABELS
    }
    return _normalize_probs(blended)


def _candidate_models() -> list[tuple[str, object]]:
    """返回短线预测候选模型。"""
    return [
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=220,
                max_depth=6,
                min_samples_leaf=6,
                random_state=42,
                class_weight="balanced_subsample",
            ),
        ),
        (
            "ExtraTreesClassifier",
            ExtraTreesClassifier(
                n_estimators=240,
                max_depth=7,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",
            ),
        ),
        ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.04, random_state=42)),
        ("LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))),
    ]


def _time_series_score(model, x_train: pd.DataFrame, y_train: pd.Series) -> float | None:
    """用时间序列切分评估候选模型，使用平衡准确率减少类别偏斜影响。"""
    if len(x_train) < 260 or y_train.nunique() < 2:
        return None
    splits = min(5, max(2, len(x_train) // 140))
    scores = []
    for train_idx, valid_idx in TimeSeriesSplit(n_splits=splits).split(x_train):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        if y_tr.nunique() < 2:
            continue
        model.fit(x_tr, y_tr)
        pred = model.predict(x_val)
        scores.append(balanced_accuracy_score(y_val, pred))
    return float(sum(scores) / len(scores)) if scores else None


def _extract_model_probs(model, latest_feature: pd.Series) -> dict:
    """从训练好的模型中提取三分类概率。"""
    proba = model.predict_proba(latest_feature[FEATURE_COLUMNS].to_frame().T)[0]
    classes = list(model.classes_)
    probs = {label: 0.0 for label in LABELS}
    for cls, value in zip(classes, proba):
        probs[cls] = float(value)
    return _normalize_probs(probs)


def _confidence_from_probs(probs: dict, cv_score: float | None, low_reliability: bool) -> str:
    """根据概率分布和交叉验证结果判断置信度。"""
    max_prob = max(probs.values())
    if low_reliability or max_prob < 0.43:
        return "低"
    if max_prob >= 0.58 and (cv_score is None or cv_score >= 0.45):
        return "高"
    return "中"


def _confidence_reason(probs: dict, cv_score: float | None, low_reliability: bool, model_name: str) -> str:
    """用简短中文解释置信度来源，避免用户只看到低/中/高。"""
    sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    top_label, top_prob = sorted_probs[0]
    second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0
    gap = top_prob - second_prob

    parts = []
    if gap >= 0.18:
        parts.append(f"{top_label}概率领先较明显")
    elif gap >= 0.08:
        parts.append(f"{top_label}概率略占优")
    else:
        parts.append("三类概率接近")

    if cv_score is None:
        parts.append("历史验证样本偏少")
    elif cv_score >= 0.50:
        parts.append("历史验证表现尚可")
    elif cv_score >= 0.42:
        parts.append("历史验证一般")
    else:
        parts.append("历史验证偏弱")

    parts.append("历史数据不足" if low_reliability else "历史数据充足")
    parts.append("已启用复盘校准" if "复盘校准" in model_name else "未启用复盘校准")
    return "；".join(parts) + "。"


def _train_predict(data: pd.DataFrame, horizon: int, threshold: float, low_reliability: bool, target_date=None) -> dict:
    """训练指定周期模型并预测最新一日的未来走势。"""
    label_col = f"label_{horizon}d"
    train_df = data.dropna(subset=FEATURE_COLUMNS + [label_col]).copy()
    latest_feature = data.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    if len(train_df) < 180 or train_df[label_col].nunique() < 2:
        return _rule_prediction(latest_feature, horizon, low_reliability=True, target_date=target_date)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[label_col]
    recent_probs = _recent_label_probs(train_df, label_col)
    rule_probs, strong_signals, _, weak_signals = _rule_probabilities(latest_feature, horizon)
    models = _candidate_models()
    last_error = None
    best = None
    for model_name, model in models:
        try:
            cv_score = _time_series_score(model, x_train, y_train)
            rank_score = cv_score if cv_score is not None else 0.0
            if best is None or rank_score > best["score"]:
                best = {"name": model_name, "model": model, "score": rank_score, "cv_score": cv_score}
        except Exception as exc:
            last_error = exc

    if best is not None:
        try:
            model = best["model"]
            model.fit(x_train, y_train)
            raw_probs = _extract_model_probs(model, latest_feature)
            probs = _blend_probabilities(raw_probs, recent_probs, rule_probs, best["cv_score"])
            direction = max(probs, key=probs.get)
            model_label = f"{best['name']} + 概率校准"
            return {
                "周期": f"{horizon}日",
                "预测日期": pd.to_datetime(target_date).strftime("%Y-%m-%d") if target_date is not None else "",
                "预测方向": direction,
                "上涨概率": probs["上涨"],
                "震荡概率": probs["震荡"],
                "下跌概率": probs["下跌"],
                "置信度": _confidence_from_probs(probs, best["cv_score"], low_reliability),
                "置信度说明": _confidence_reason(probs, best["cv_score"], low_reliability, model_label),
                "预测依据": _build_prediction_reason(latest_feature, direction, best["cv_score"], best["name"]),
                "主要支撑信号": strong_signals or _support_signals(latest_feature),
                "主要风险信号": weak_signals or _risk_signals(latest_feature),
                "风险提示": "历史数据不足，预测可靠性较低。" if low_reliability else "模型仅基于历史量价特征，无法覆盖突发消息和市场环境变化。",
                "模型": model_label,
            }
        except Exception as exc:
            last_error = exc
    result = _rule_prediction(latest_feature, horizon, low_reliability=True, target_date=target_date)
    result["风险提示"] += f" 机器学习模型训练失败，已使用规则模型。"
    result["模型错误"] = str(last_error)
    return result


def _support_signals(row: pd.Series) -> list[str]:
    """提取偏强支撑信号。"""
    signals = []
    if row.get("dev_ma20", 0) > 0:
        signals.append("收盘价位于 MA20 上方")
    if row.get("ma_bull", 0) == 1:
        signals.append("短期均线多头排列")
    if row.get("DIF", 0) > row.get("DEA", 0):
        signals.append("DIF 位于 DEA 上方")
    if 50 <= row.get("RSI12", 50) <= 70:
        signals.append("RSI12 处于偏强但未极端区间")
    if row.get("volume_up", 0) == 1:
        signals.append("放量上涨")
    return signals or ["未出现明显偏强信号"]


def _risk_signals(row: pd.Series) -> list[str]:
    """提取偏弱风险信号。"""
    signals = []
    if row.get("dev_ma20", 0) < 0:
        signals.append("收盘价位于 MA20 下方")
    if row.get("ma_bear", 0) == 1:
        signals.append("短期均线空头排列")
    if row.get("DIF", 0) < row.get("DEA", 0):
        signals.append("DIF 位于 DEA 下方")
    if row.get("RSI12", 50) < 45:
        signals.append("RSI12 偏弱")
    if row.get("volume_down", 0) == 1:
        signals.append("放量下跌")
    return signals or ["未出现明显偏弱风险信号"]


def _build_prediction_reason(row: pd.Series, direction: str, cv_score: float | None, model_name: str | None = None) -> str:
    """生成模型预测依据说明。"""
    parts = []
    if direction == "上涨":
        parts.extend(_support_signals(row)[:3])
    elif direction == "下跌":
        parts.extend(_risk_signals(row)[:3])
    else:
        parts.append("多空信号接近，模型更偏向震荡判断")
    if model_name:
        parts.append(f"历史验证后选择 {model_name}")
    if cv_score is not None:
        parts.append(f"时间序列平衡准确率约 {cv_score:.2%}")
    return "；".join(parts)


def predict_short_term(
    stock_code: str,
    threshold: float = 0.01,
    threshold_mode: str = "manual",
) -> dict:
    """输出未来 1-3 个交易日短线走势预测。"""
    hist_df = fetch_prediction_history(stock_code)
    normalized_mode = "adaptive" if threshold_mode == "adaptive" else "manual"
    feature_df = add_prediction_labels(
        build_features(hist_df),
        threshold=threshold,
        threshold_mode=normalized_mode,
    )
    low_reliability = len(hist_df) < 250
    target_dates, calendar_source = next_trade_dates(hist_df.iloc[-1]["date"], 3)
    predictions = []
    latest_feature = feature_df.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    current_thresholds = {}
    for horizon in [1, 2, 3]:
        horizon_threshold = float(latest_feature.get(f"label_threshold_{horizon}d", threshold))
        current_thresholds[horizon] = horizon_threshold
        item = _train_predict(
            feature_df,
            horizon,
            horizon_threshold,
            low_reliability,
            target_date=target_dates[horizon - 1],
        )
        item["判断阈值"] = horizon_threshold
        item["阈值模式"] = "波动自适应" if normalized_mode == "adaptive" else "手动固定"
        item["标签版本"] = ADAPTIVE_THRESHOLD_VERSION if normalized_mode == "adaptive" else "fixed_v1"
        predictions.append(_attach_range_estimate(item, feature_df, latest_feature, horizon, horizon_threshold))
    similar_patterns = find_similar_patterns(feature_df, threshold=threshold, limit=8)
    return {
        "code": normalize_stock_code(stock_code),
        "rows": len(hist_df),
        "last_close": float(hist_df.iloc[-1]["close"]),
        "last_trade_date": pd.to_datetime(hist_df.iloc[-1]["date"]).strftime("%Y-%m-%d"),
        "calendar_source": calendar_source,
        "low_reliability": low_reliability,
        "data": feature_df,
        "predictions": predictions,
        "similar_patterns": similar_patterns,
        "threshold_mode": normalized_mode,
        "label_version": ADAPTIVE_THRESHOLD_VERSION if normalized_mode == "adaptive" else "fixed_v1",
        "current_thresholds": current_thresholds,
        "model_note": (
            "使用前复权历史行情构造量价特征，分别训练 1、2、3 日分类模型；训练不使用未来数据。"
            + (
                " 标签阈值按每个历史时点的近20日波动率和预测周期自动调整。"
                if normalized_mode == "adaptive"
                else f" 标签使用固定阈值 {threshold:.2%}。"
            )
        ),
    }
