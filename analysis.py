import pandas as pd


def _format_percent(value: float | None) -> str:
    """把数字格式化为百分比文本，空值时返回暂无数据。"""
    if value is None or pd.isna(value):
        return "暂无数据"
    return f"{value:.2f}%"


def judge_trend(df: pd.DataFrame) -> str:
    """根据最近收盘价和 MA20 判断当前趋势。"""
    latest = df.iloc[-1]
    recent = df.tail(20)
    close = latest["close"]
    ma20 = latest.get("MA20")
    if pd.isna(ma20):
        return "震荡（样本不足，暂以中性处理）"

    recent_return = (recent.iloc[-1]["close"] - recent.iloc[0]["close"]) / recent.iloc[0]["close"] * 100
    if close > ma20 and recent_return > 3:
        return "上涨"
    if close < ma20 and recent_return < -3:
        return "下跌"
    return "震荡"


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
    trend = judge_trend(df)
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
        "简短报告": f"当前趋势判断为{trend}，后市技术面倾向为{forward_outlook['后市倾向']}。{ma_text} {volume_text} {macd_text} {rsi_text} {boll_text} 风险方面，{risk_text}",
        "综合分析结论": {
            "趋势判断": trend,
            "技术面信号": f"{ma_text} {macd_text} {rsi_text} {boll_text} 后市倾向：{forward_outlook['后市倾向']}，依据：{forward_outlook['判断依据']}。",
            "风险点": risk_text,
            "免责声明": "仅供学习参考，不构成投资建议。",
        },
    }
