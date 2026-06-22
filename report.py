from datetime import datetime

import pandas as pd


def _format_number(value, digits: int = 2) -> str:
    """格式化数值，空值时返回暂无数据。"""
    if value is None or pd.isna(value):
        return "暂无数据"
    return f"{float(value):.{digits}f}"


def _format_score(score_result: dict | None) -> str:
    """格式化综合评分为 Markdown 文本。"""
    if not score_result:
        return "综合评分暂不可用。"
    detail = score_result.get("评分明细", {})
    explanations = score_result.get("评分解释", {})
    lines = [
        f"- 综合分数：{score_result.get('综合分数', '暂无')}",
        f"- 等级：{score_result.get('等级', '暂无')}",
    ]
    for key, value in detail.items():
        reason = "；".join(explanations.get(key, []))
        lines.append(f"- {key}：{value} 分。{reason}")
    return "\n".join(lines)


def generate_markdown_report(
    stock_info: dict,
    df: pd.DataFrame,
    analysis_result: dict,
    score_result: dict | None = None,
) -> str:
    """根据个股信息、行情数据和评分结果生成 Markdown 分析报告。"""
    latest = df.iloc[-1] if df is not None and not df.empty else {}
    conclusion = analysis_result.get("综合分析结论", {})
    risks = analysis_result.get("风险提示列表", [])
    risk_text = "\n".join(f"- {item}" for item in risks) if risks else f"- {analysis_result.get('风险提示', '暂无风险提示')}"

    return f"""# {stock_info.get("name", stock_info.get("code", ""))}（{stock_info.get("code", "")}）分析报告

## 1. 股票基本信息

- 分析日期：{datetime.now().strftime("%Y-%m-%d %H:%M")}
- 股票代码：{stock_info.get("code", "")}
- 股票名称：{stock_info.get("name", "")}

## 2. 最新行情

- 最近收盘价：{_format_number(latest.get("close") if hasattr(latest, "get") else None)}
- 页面价格：{_format_number(stock_info.get("latest_price"))}
- 涨跌幅：{_format_number(stock_info.get("pct_chg"))}%
- 成交量：{_format_number((latest.get("volume") if hasattr(latest, "get") else None) / 10000 if hasattr(latest, "get") and latest.get("volume") is not None else None)} 万手

## 3. K 线趋势分析

- 趋势判断：{analysis_result.get("趋势判断", "暂无数据")}
- 后市倾向：{analysis_result.get("后市倾向", "暂无数据")}
- 判断依据：{analysis_result.get("后市判断依据", "暂无数据")}

## 4. 均线分析

{analysis_result.get("均线状态", "暂无数据")}

## 5. MACD 分析

{analysis_result.get("MACD 状态", "暂无数据")}

## 6. RSI 分析

{analysis_result.get("RSI 状态", "暂无数据")}

## 7. BOLL 分析

{analysis_result.get("BOLL 状态", "暂无数据")}

## 8. 成交量分析

{analysis_result.get("成交量变化", "暂无数据")}

## 9. 综合评分

{_format_score(score_result)}

## 10. 风险提示

{risk_text}

## 11. 综合结论

- 趋势判断：{conclusion.get("趋势判断", analysis_result.get("趋势判断", "暂无数据"))}
- 技术面信号：{conclusion.get("技术面信号", "暂无数据")}
- 风险点：{conclusion.get("风险点", analysis_result.get("风险提示", "暂无数据"))}

## 12. 免责声明

本报告仅供学习、研究和数据分析参考，不构成任何投资建议。股市有风险，投资需谨慎。
"""
