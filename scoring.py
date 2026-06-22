import pandas as pd

from utils import safe_float


def _clip_score(value: float, max_score: int) -> float:
    """限制单项评分在 0 到满分之间。"""
    return max(0, min(max_score, round(value, 2)))


def score_trend(df: pd.DataFrame) -> tuple[float, list[str]]:
    """计算趋势评分，满分 30 分。"""
    latest = df.iloc[-1]
    close = latest["close"]
    score = 0
    notes = []
    if not pd.isna(latest.get("MA20")) and close > latest["MA20"]:
        score += 10
        notes.append("价格位于 MA20 上方")
    if not pd.isna(latest.get("MA60")) and close > latest["MA60"]:
        score += 8
        notes.append("价格位于 MA60 上方")
    if all(not pd.isna(latest.get(col)) for col in ["MA5", "MA10", "MA20"]) and latest["MA5"] > latest["MA10"] > latest["MA20"]:
        score += 6
        notes.append("MA5、MA10、MA20 呈多头排列")
    if len(df) >= 20 and df.tail(20).iloc[0]["close"]:
        ret20 = (close / df.tail(20).iloc[0]["close"] - 1) * 100
        if ret20 > 5:
            score += 6
            notes.append("近 20 日涨幅较强")
        elif ret20 > 0:
            score += 3
            notes.append("近 20 日小幅上涨")
    return _clip_score(score, 30), notes


def score_momentum(df: pd.DataFrame) -> tuple[float, list[str]]:
    """计算动能评分，满分 25 分。"""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    score = 0
    notes = []
    if latest.get("DIF") > latest.get("DEA"):
        score += 9
        notes.append("MACD 位于金叉状态")
    if latest.get("MACD") > prev.get("MACD"):
        score += 8
        notes.append("MACD 柱状图增强")
    rsi = safe_float(latest.get("RSI"))
    if rsi is not None:
        if 40 <= rsi <= 65:
            score += 8
            notes.append("RSI 处于相对合理区间")
        elif 30 <= rsi < 40 or 65 < rsi <= 75:
            score += 5
            notes.append("RSI 接近边界区间")
    return _clip_score(score, 25), notes


def score_volume(df: pd.DataFrame) -> tuple[float, list[str]]:
    """计算成交量评分，满分 20 分。"""
    latest = df.iloc[-1]
    avg_volume = df["volume"].tail(20).mean()
    if not avg_volume:
        return 8, ["成交量样本不足"]
    ratio = latest["volume"] / avg_volume
    price_up = latest["close"] >= latest["open"]
    score = 10
    notes = []
    if price_up and 1.0 <= ratio <= 1.8:
        score += 7
        notes.append("上涨伴随温和放量")
    elif not price_up and ratio < 1.0:
        score += 5
        notes.append("下跌时量能未明显放大")
    elif not price_up and ratio > 1.5:
        score -= 6
        notes.append("放量下跌，抛压偏大")
    if ratio > 2.5:
        score -= 4
        notes.append("成交量异常放大，波动风险上升")
    return _clip_score(score, 20), notes


def score_risk(df: pd.DataFrame) -> tuple[float, list[str]]:
    """计算风险评分，满分 25 分，分数越高代表风险越可控。"""
    latest = df.iloc[-1]
    score = 25
    notes = []
    rsi = safe_float(latest.get("RSI"))
    if rsi and rsi > 75:
        score -= 6
        notes.append("RSI 过高")
    if not pd.isna(latest.get("MA20")) and latest["close"] < latest["MA20"]:
        score -= 6
        notes.append("价格跌破 MA20")
    if not pd.isna(latest.get("MA60")) and latest["close"] < latest["MA60"]:
        score -= 5
        notes.append("价格跌破 MA60")
    if not pd.isna(latest.get("MA20")) and latest["MA20"] and abs(latest["close"] / latest["MA20"] - 1) > 0.12:
        score -= 4
        notes.append("价格偏离 MA20 较大")
    avg_volume = df["volume"].tail(20).mean()
    if avg_volume and latest["close"] < latest["open"] and latest["volume"] > avg_volume * 1.5:
        score -= 5
        notes.append("放量下跌")
    if not notes:
        notes.append("未触发明显极端风险")
    return _clip_score(score, 25), notes


def build_score(df: pd.DataFrame, flow_df: pd.DataFrame | None = None) -> dict:
    """计算综合评分和等级。"""
    trend_score, trend_notes = score_trend(df)
    momentum_score, momentum_notes = score_momentum(df)
    volume_score, volume_notes = score_volume(df)
    risk_score, risk_notes = score_risk(df)
    total = round(trend_score + momentum_score + volume_score + risk_score, 2)
    if total >= 85:
        grade = "强势"
    elif total >= 70:
        grade = "偏强"
    elif total >= 55:
        grade = "中性"
    elif total >= 40:
        grade = "偏弱"
    else:
        grade = "弱势"
    return {
        "综合分数": total,
        "等级": grade,
        "评分明细": {
            "趋势评分": trend_score,
            "动能评分": momentum_score,
            "成交量评分": volume_score,
            "风险评分": risk_score,
        },
        "评分解释": {
            "趋势评分": trend_notes,
            "动能评分": momentum_notes,
            "成交量评分": volume_notes,
            "风险评分": risk_notes,
        },
        "风险提示": risk_notes,
        "免责声明": "评分仅供学习和研究使用，不构成投资建议。",
    }
