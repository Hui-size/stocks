import pandas as pd


def _format_percent(value: float | None) -> str:
    """把数字格式化为百分比文本，空值时返回暂无数据。"""
    if value is None or pd.isna(value):
        return "暂无数据"
    return f"{value:.2f}%"


def evaluate_trend_regime(df: pd.DataFrame) -> dict:
    """综合价格、均线、波动率和趋势效率判断当前趋势状态。"""
    if df is None or df.empty or len(df) < 20:
        return {"趋势": "震荡", "趋势强度": 0, "趋势置信度": "低", "趋势依据": ["有效样本不足 20 个交易日"]}

    latest = df.iloc[-1]
    close = df["close"]
    returns = close.pct_change()
    daily_volatility = float(returns.tail(20).std()) if len(df) >= 20 else 0.0
    score = 0.0
    reasons = []

    for window, weight in [(5, 0.7), (20, 1.2), (60, 0.8)]:
        if len(df) <= window or daily_volatility <= 0:
            continue
        period_return = float(close.iloc[-1] / close.iloc[-window - 1] - 1)
        normalized_move = period_return / (daily_volatility * (window**0.5))
        contribution = max(-1.5, min(1.5, normalized_move)) * weight
        score += contribution
        if window in (20, 60) and abs(normalized_move) >= 0.55:
            direction = "上涨" if period_return > 0 else "下跌"
            reasons.append(f"近{window}日{direction} {abs(period_return):.2%}，相对波动率方向较明确")

    ma5, ma10, ma20, ma60 = (latest.get(name) for name in ["MA5", "MA10", "MA20", "MA60"])
    if all(pd.notna(value) for value in [ma5, ma10, ma20, ma60]):
        if ma5 > ma10 > ma20 > ma60:
            score += 1.5
            reasons.append("MA5、MA10、MA20、MA60 呈多头排列")
        elif ma5 < ma10 < ma20 < ma60:
            score -= 1.5
            reasons.append("MA5、MA10、MA20、MA60 呈空头排列")
        elif latest["close"] > ma20:
            score += 0.45
        else:
            score -= 0.45

    if len(df) >= 25 and pd.notna(ma20) and pd.notna(df.iloc[-6].get("MA20")) and daily_volatility > 0:
        ma20_slope = float(ma20 / df.iloc[-6]["MA20"] - 1)
        slope_strength = ma20_slope / (daily_volatility * (5**0.5))
        score += max(-1.0, min(1.0, slope_strength))
        if abs(slope_strength) >= 0.45:
            reasons.append("MA20 斜率向上" if slope_strength > 0 else "MA20 斜率向下")

    path_length = float(close.diff().abs().tail(20).sum())
    net_move = float(abs(close.iloc[-1] - close.iloc[-21])) if len(df) >= 21 else 0.0
    trend_efficiency = net_move / path_length if path_length > 0 else 0.0
    if trend_efficiency >= 0.35:
        direction = 1 if close.iloc[-1] >= close.iloc[-21] else -1
        score += direction * min(1.0, trend_efficiency * 1.5)
        reasons.append(f"20日趋势效率 {trend_efficiency:.0%}，走势延续性较高")
    elif trend_efficiency < 0.18:
        score *= 0.82
        reasons.append(f"20日趋势效率仅 {trend_efficiency:.0%}，往返震荡较多")

    if score >= 2.4:
        trend = "上涨"
    elif score <= -2.4:
        trend = "下跌"
    else:
        trend = "震荡"
    strength = int(round(max(-100, min(100, score / 6 * 100))))
    confidence = "高" if abs(score) >= 4 and trend_efficiency >= 0.30 else ("中" if abs(score) >= 2.4 else "低")
    return {
        "趋势": trend,
        "趋势强度": strength,
        "趋势置信度": confidence,
        "趋势效率": trend_efficiency,
        "趋势依据": reasons[:4] or ["多周期价格与均线信号相互抵消"],
    }


def judge_trend(df: pd.DataFrame) -> str:
    """返回多周期、波动率归一化后的当前趋势。"""
    return evaluate_trend_regime(df)["趋势"]


def analyze_ma(df: pd.DataFrame) -> str:
    """分析短中长期均线排列和价格相对均线的位置。"""
    latest = df.iloc[-1]
    close = latest["close"]
    ma5, ma10, ma20, ma60 = latest.get("MA5"), latest.get("MA10"), latest.get("MA20"), latest.get("MA60")
    if any(pd.isna(x) for x in [ma5, ma10, ma20, ma60]):
        return "均线数据仍在形成中，MA60 需要至少 60 个交易日。"
    if ma5 > ma10 > ma20 > ma60 and close > ma5:
        return "MA5、MA10、MA20、MA60 呈多头排列，价格位于短期均线上方，短期走势偏强。"
    if ma5 < ma10 < ma20 < ma60 and close < ma5:
        return "MA5、MA10、MA20、MA60 呈空头排列，价格位于短期均线下方，短期走势偏弱。"
    return "均线排列不完全一致，价格可能处在震荡整理或趋势切换阶段。"


def analyze_volume(df: pd.DataFrame) -> str:
    """比较最近成交量和 20 日均量，判断量能变化。"""
    latest_volume = df.iloc[-1]["volume"]
    avg_volume = df["volume"].tail(20).mean()
    if pd.isna(avg_volume) or avg_volume == 0:
        return "成交量数据不足，暂无法判断量能变化。"
    ratio = latest_volume / avg_volume
    if ratio > 1.5:
        return f"最近成交量约为 20 日均量的 {ratio:.2f} 倍，量能明显放大。"
    if ratio < 0.7:
        return f"最近成交量约为 20 日均量的 {ratio:.2f} 倍，量能偏低。"
    return f"最近成交量约为 20 日均量的 {ratio:.2f} 倍，量能处于常规区间。"


def analyze_macd(df: pd.DataFrame) -> str:
    """根据 DIF、DEA 和 MACD 柱判断 MACD 状态。"""
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) >= 2 else latest
    dif, dea, macd = latest.get("DIF"), latest.get("DEA"), latest.get("MACD")
    prev_macd = previous.get("MACD")
    if any(pd.isna(x) for x in [dif, dea, macd]):
        return "MACD 数据不足，暂无法判断。"
    direction = "扩大" if macd > prev_macd else "收敛"
    if dif > dea and macd > 0:
        return f"DIF 位于 DEA 上方，MACD 柱为正且正在{direction}，动能偏多。"
    if dif < dea and macd < 0:
        return f"DIF 位于 DEA 下方，MACD 柱为负且正在{direction}，动能偏弱。"
    return f"DIF 与 DEA 信号不完全一致，MACD 柱正在{direction}，建议结合价格和成交量观察。"


def analyze_rsi(df: pd.DataFrame) -> str:
    """根据 RSI 数值判断是否超买或超卖。"""
    rsi = df.iloc[-1].get("RSI")
    if pd.isna(rsi):
        return "RSI 数据不足，暂无法判断。"
    if rsi >= 70:
        return f"RSI 为 {rsi:.2f}，处于常见超买区间，短线追高风险上升。"
    if rsi <= 30:
        return f"RSI 为 {rsi:.2f}，处于常见超卖区间，可能存在反弹需求但仍需确认。"
    return f"RSI 为 {rsi:.2f}，处于中性区间。"


def analyze_boll(df: pd.DataFrame) -> str:
    """分析价格相对布林线的位置。"""
    latest = df.iloc[-1]
    close = latest["close"]
    upper = latest.get("BOLL_UPPER")
    middle = latest.get("BOLL_MID")
    lower = latest.get("BOLL_LOWER")
    if any(pd.isna(x) for x in [upper, middle, lower]):
        return "BOLL 数据不足，暂无法判断。"
    if close > upper:
        return "收盘价位于布林线上轨上方，短线波动可能放大。"
    if close < lower:
        return "收盘价位于布林线下轨下方，弱势波动需要关注。"
    if close >= middle:
        return "收盘价位于布林线中轨上方、上轨下方，价格处在相对偏强区间。"
    return "收盘价位于布林线中轨下方、下轨上方，价格处在相对偏弱区间。"


def build_risk_flags(df: pd.DataFrame) -> list[str]:
    """根据 RSI、均线、MACD、成交量和价格关系生成风险提示列表。"""
    risks = []
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) >= 2 else latest
    close = latest["close"]
    open_price = latest["open"]
    rsi = latest.get("RSI")
    ma20 = latest.get("MA20")
    dif = latest.get("DIF")
    dea = latest.get("DEA")
    prev_dif = previous.get("DIF")
    prev_dea = previous.get("DEA")
    latest_volume = latest.get("volume")
    avg_volume = df["volume"].tail(20).mean()

    if not pd.isna(rsi) and rsi > 70:
        risks.append(f"RSI 为 {rsi:.2f}，高于 70，可能存在短期超买风险。")
    if not pd.isna(rsi) and rsi < 30:
        risks.append(f"RSI 为 {rsi:.2f}，低于 30，可能存在短期超卖后的反弹可能。")
    if not pd.isna(ma20) and close < ma20:
        risks.append("收盘价跌破 MA20，短期趋势有转弱迹象。")
    if all(not pd.isna(x) for x in [dif, dea, prev_dif, prev_dea]) and prev_dif >= prev_dea and dif < dea:
        risks.append("MACD 出现死叉，动能有减弱迹象。")
    if (
        not pd.isna(latest_volume)
        and not pd.isna(avg_volume)
        and avg_volume > 0
        and latest_volume > avg_volume * 1.5
        and close < open_price
    ):
        risks.append("成交量异常放大但价格下跌，盘中抛压相对较大。")

    if not risks:
        risks.append("当前技术指标未触发明显极端风险，但仍需关注市场、行业和公司公告变化。")
    return risks


def detect_anomaly_flags(df: pd.DataFrame) -> list[dict]:
    """检测短线异常波动，只提示客观现象，不给交易建议。"""
    if df is None or df.empty or len(df) < 8:
        return []

    data = df.copy()
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) >= 2 else latest
    flags = []

    returns = data["close"].pct_change()
    latest_return = float(returns.iloc[-1]) if pd.notna(returns.iloc[-1]) else 0.0
    avg_abs_return = float(returns.abs().tail(20).mean()) if len(data) >= 20 else float(returns.abs().mean())
    if avg_abs_return > 0 and abs(latest_return) >= max(avg_abs_return * 1.8, 0.025):
        flags.append(
            {
                "类型": "价格波动放大",
                "级别": "关注",
                "说明": f"今日涨跌幅 {latest_return:.2%}，明显高于近期平均绝对波动 {avg_abs_return:.2%}。",
            }
        )

    latest_volume = latest.get("volume")
    avg_volume = data["volume"].tail(20).mean()
    if pd.notna(latest_volume) and pd.notna(avg_volume) and avg_volume > 0:
        volume_ratio = float(latest_volume / avg_volume)
        if volume_ratio >= 2.0:
            flags.append(
                {
                    "类型": "成交量突然放大",
                    "级别": "关注",
                    "说明": f"今日成交量约为 20 日均量的 {volume_ratio:.2f} 倍，量能明显异常。",
                }
            )

    for ma_name in ["MA20", "MA60"]:
        ma_value = latest.get(ma_name)
        prev_ma_value = previous.get(ma_name)
        if all(pd.notna(x) for x in [ma_value, prev_ma_value, latest.get("close"), previous.get("close")]):
            if previous["close"] >= prev_ma_value and latest["close"] < ma_value:
                flags.append(
                    {
                        "类型": f"跌破{ma_name}",
                        "级别": "风险",
                        "说明": f"收盘价由 {ma_name} 上方转到下方，短线趋势可能转弱。",
                    }
                )

    recent3 = data.tail(3).copy()
    if len(recent3) == 3:
        day_returns = recent3["close"].pct_change()
        candle_returns = recent3["close"] / recent3["open"] - 1
        big_up = (candle_returns > 0.025).all() and (day_returns.iloc[1:] > 0).all()
        big_down = (candle_returns < -0.025).all() and (day_returns.iloc[1:] < 0).all()
        if big_up:
            flags.append(
                {
                    "类型": "连续 3 天大阳线",
                    "级别": "关注",
                    "说明": "最近 3 个交易日均为较强阳线，短线波动和分歧可能加大。",
                }
            )
        if big_down:
            flags.append(
                {
                    "类型": "连续 3 天大阴线",
                    "级别": "风险",
                    "说明": "最近 3 个交易日均为较弱阴线，短线承压迹象较明显。",
                }
            )

    return flags


def _nearest_level(levels: list[float], current_price: float, side: str) -> float | None:
    """从候选价位里取最接近当前价的上方或下方位置。"""
    clean = sorted({round(float(level), 2) for level in levels if pd.notna(level) and level > 0})
    if side == "support":
        candidates = [level for level in clean if level <= current_price]
        return max(candidates) if candidates else (min(clean) if clean else None)
    candidates = [level for level in clean if level >= current_price]
    return min(candidates) if candidates else (max(clean) if clean else None)


def _volume_dense_levels(df: pd.DataFrame, bins: int = 12) -> list[float]:
    """用收盘价区间和成交量估算成交密集价位。"""
    data = df.dropna(subset=["close", "volume"]).copy()
    if len(data) < 10:
        return []
    low = float(data["close"].min())
    high = float(data["close"].max())
    if high <= low:
        return [round(float(data["close"].iloc[-1]), 2)]

    bucket_count = min(bins, max(5, len(data) // 4))
    bucket_width = (high - low) / bucket_count
    if bucket_width <= 0:
        return []
    bucket_index = ((data["close"] - low) / bucket_width).clip(0, bucket_count - 1).astype(int)
    volume_by_bucket = data.groupby(bucket_index)["volume"].sum().sort_values(ascending=False)
    dense_levels = []
    for bucket in volume_by_bucket.head(3).index:
        center = low + (int(bucket) + 0.5) * bucket_width
        dense_levels.append(round(center, 2))
    return dense_levels


def estimate_support_resistance(df: pd.DataFrame) -> dict:
    """根据近20/60日高低点和成交密集区估算短线支撑位、压力位。"""
    if df is None or df.empty or len(df) < 20:
        return {}

    data = df.copy()
    latest = data.iloc[-1]
    current_price = float(latest["close"])
    recent20 = data.tail(20)
    recent60 = data.tail(min(60, len(data)))

    dense_levels = _volume_dense_levels(recent60)
    supports = [
        float(recent20["low"].min()),
        float(recent60["low"].min()),
        *[level for level in dense_levels if level <= current_price],
    ]
    resistances = [
        float(recent20["high"].max()),
        float(recent60["high"].max()),
        *[level for level in dense_levels if level >= current_price],
    ]

    support = _nearest_level(supports, current_price, "support")
    resistance = _nearest_level(resistances, current_price, "resistance")
    support_gap = None if support is None else (current_price / support - 1)
    resistance_gap = None if resistance is None else (resistance / current_price - 1)

    explanation = (
        f"近20日低点 {recent20['low'].min():.2f}、近60日低点 {recent60['low'].min():.2f} "
        f"作为支撑候选；近20日高点 {recent20['high'].max():.2f}、近60日高点 {recent60['high'].max():.2f} "
        "作为压力候选。成交密集区由最近60日收盘价区间按成交量粗略分箱估算。"
    )

    return {
        "current_price": current_price,
        "support": support,
        "resistance": resistance,
        "support_gap": support_gap,
        "resistance_gap": resistance_gap,
        "dense_levels": dense_levels,
        "near_20_low": float(recent20["low"].min()),
        "near_20_high": float(recent20["high"].max()),
        "near_60_low": float(recent60["low"].min()),
        "near_60_high": float(recent60["high"].max()),
        "explanation": explanation,
    }


def analyze_risk(df: pd.DataFrame) -> str:
    """结合波动率和布林线位置生成风险提示。"""
    latest = df.iloc[-1]
    recent_volatility = df["close"].pct_change().tail(20).std() * 100
    close = latest["close"]
    upper = latest.get("BOLL_UPPER")
    lower = latest.get("BOLL_LOWER")
    risk_parts = []
    if not pd.isna(recent_volatility) and recent_volatility > 3:
        risk_parts.append(f"近 20 日价格波动率约 {recent_volatility:.2f}%，波动偏高。")
    if not pd.isna(upper) and close > upper:
        risk_parts.append("收盘价突破布林线上轨，短线波动和回落风险需要关注。")
    if not pd.isna(lower) and close < lower:
        risk_parts.append("收盘价跌破布林线下轨，弱势延续风险需要关注。")
    if not risk_parts:
        risk_parts.append("未发现特别极端的技术风险，但仍需关注市场、行业和个股公告变化。")
    return " ".join(risk_parts)


def judge_forward_outlook(df: pd.DataFrame) -> dict:
    """根据技术指标给出后市倾向判断，避免确定性涨跌表述。"""
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) >= 2 else latest
    close = latest["close"]
    open_price = latest["open"]
    ma5 = latest.get("MA5")
    ma10 = latest.get("MA10")
    ma20 = latest.get("MA20")
    ma60 = latest.get("MA60")
    dif = latest.get("DIF")
    dea = latest.get("DEA")
    macd = latest.get("MACD")
    prev_macd = previous.get("MACD")
    rsi = latest.get("RSI")
    volume = latest.get("volume")
    avg_volume = df["volume"].tail(20).mean()

    score = 0
    positives = []
    negatives = []

    if not pd.isna(ma20):
        if close > ma20:
            score += 1
            positives.append("收盘价位于 MA20 上方")
        else:
            score -= 1
            negatives.append("收盘价跌破 MA20")

    if all(not pd.isna(x) for x in [ma5, ma10, ma20, ma60]):
        if ma5 > ma10 > ma20 > ma60:
            score += 1
            positives.append("均线呈多头排列")
        elif ma5 < ma10 < ma20 < ma60:
            score -= 1
            negatives.append("均线呈空头排列")

    if all(not pd.isna(x) for x in [dif, dea, macd]):
        if dif > dea and macd > 0:
            score += 1
            positives.append("MACD 动能偏多")
        elif dif < dea and macd < 0:
            score -= 1
            negatives.append("MACD 动能偏弱")

    if not pd.isna(rsi):
        if 45 <= rsi <= 65:
            score += 0.5
            positives.append("RSI 处于相对健康区间")
        elif rsi > 75:
            score -= 0.5
            negatives.append("RSI 偏高，短线拥挤风险上升")
        elif rsi < 30:
            score -= 0.5
            negatives.append("RSI 偏低，弱势仍需确认")

    if all(not pd.isna(x) for x in [volume, avg_volume]) and avg_volume > 0:
        volume_ratio = volume / avg_volume
        if volume_ratio > 1.3 and close > open_price:
            score += 0.5
            positives.append("放量上涨，资金关注度提升")
        elif volume_ratio > 1.3 and close < open_price:
            score -= 0.5
            negatives.append("放量下跌，抛压偏大")

    if not pd.isna(prev_macd) and not pd.isna(macd):
        if macd > prev_macd:
            positives.append("MACD 柱较前一日改善")
        elif macd < prev_macd:
            negatives.append("MACD 柱较前一日走弱")

    if score >= 2:
        outlook = "偏强"
    elif score <= -2:
        outlook = "偏弱"
    else:
        outlook = "震荡"

    main_reasons = positives[:2] + negatives[:2]
    reason_text = "；".join(main_reasons) if main_reasons else "指标信号不充分，暂以中性观察。"
    return {
        "后市倾向": outlook,
        "后市评分": round(score, 2),
        "判断依据": reason_text,
        "积极信号": positives,
        "压力信号": negatives,
    }


def build_analysis(df: pd.DataFrame, pct_chg: float | None = None) -> dict:
    """生成页面展示所需的结构化文字分析结论。"""
    trend_profile = evaluate_trend_regime(df)
    trend = trend_profile["趋势"]
    ma_text = analyze_ma(df)
    volume_text = analyze_volume(df)
    macd_text = analyze_macd(df)
    rsi_text = analyze_rsi(df)
    boll_text = analyze_boll(df)
    risk_flags = build_risk_flags(df)
    volatility_risk = analyze_risk(df)
    forward_outlook = judge_forward_outlook(df)
    all_risks = risk_flags if volatility_risk in risk_flags else risk_flags + [volatility_risk]
    risk_text = " ".join(all_risks)

    return {
        "趋势判断": trend,
        "趋势强度": trend_profile["趋势强度"],
        "趋势置信度": trend_profile["趋势置信度"],
        "趋势效率": trend_profile.get("趋势效率"),
        "趋势判断依据": "；".join(trend_profile["趋势依据"]),
        "涨跌幅": _format_percent(pct_chg),
        "均线状态": ma_text,
        "成交量变化": volume_text,
        "MACD 状态": macd_text,
        "RSI 状态": rsi_text,
        "BOLL 状态": boll_text,
        "风险提示": risk_text,
        "风险提示列表": all_risks,
        "波动风险": volatility_risk,
        "后市倾向": forward_outlook["后市倾向"],
        "后市评分": forward_outlook["后市评分"],
        "后市判断依据": forward_outlook["判断依据"],
        "积极信号": forward_outlook["积极信号"],
        "压力信号": forward_outlook["压力信号"],
        "简短报告": f"当前趋势判断为{trend}，趋势强度 {trend_profile['趋势强度']:+d}，置信度{trend_profile['趋势置信度']}；后市技术面倾向为{forward_outlook['后市倾向']}。{ma_text} {volume_text} {macd_text} {rsi_text} {boll_text} 风险方面，{risk_text}",
        "综合分析结论": {
            "趋势判断": trend,
            "技术面信号": f"趋势强度 {trend_profile['趋势强度']:+d}（置信度{trend_profile['趋势置信度']}），依据：{'；'.join(trend_profile['趋势依据'])}。{ma_text} {macd_text} {rsi_text} {boll_text} 后市倾向：{forward_outlook['后市倾向']}，依据：{forward_outlook['判断依据']}。",
            "风险点": risk_text,
            "免责声明": "仅供学习参考，不构成投资建议。",
        },
    }
