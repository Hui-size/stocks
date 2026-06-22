import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from prediction import fetch_prediction_history


FEEDBACK_FILE = Path("prediction_feedback.json")


def load_feedback(path: Path = FEEDBACK_FILE) -> list[dict]:
    """读取本地预测复盘记录，文件不存在或损坏时返回空列表。"""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def save_feedback(records: list[dict], path: Path = FEEDBACK_FILE) -> None:
    """把预测复盘记录保存到本地 JSON 文件。"""
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _record_key(record: dict) -> str:
    """生成一条预测记录的唯一键，避免重复写入同一次预测。"""
    return f"{record.get('code')}|{record.get('base_date')}|{record.get('target_date')}|{record.get('horizon')}"


def record_predictions(stock_code: str, prediction_result: dict, path: Path = FEEDBACK_FILE) -> None:
    """记录本次预测结果，后续有实际行情后自动复盘。"""
    records = load_feedback(path)
    existing = {_record_key(item) for item in records}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    changed = False

    for item in prediction_result.get("predictions", []):
        record = {
            "code": stock_code,
            "base_date": prediction_result.get("last_trade_date"),
            "target_date": item.get("预测日期"),
            "horizon": item.get("周期"),
            "predicted": item.get("预测方向"),
            "up_prob": item.get("上涨概率"),
            "flat_prob": item.get("震荡概率"),
            "down_prob": item.get("下跌概率"),
            "confidence": item.get("置信度"),
            "model": item.get("模型"),
            "basis": item.get("预测依据"),
            "status": "pending",
            "created_at": now,
        }
        key = _record_key(record)
        if key not in existing:
            records.append(record)
            existing.add(key)
            changed = True

    if changed:
        save_feedback(records, path)


def _label_from_return(value: float, threshold: float) -> str:
    """根据区间收益率生成上涨、震荡、下跌标签。"""
    if value > threshold:
        return "上涨"
    if value < -threshold:
        return "下跌"
    return "震荡"


def update_prediction_outcomes(stock_code: str, threshold: float, path: Path = FEEDBACK_FILE, hist_df: pd.DataFrame | None = None) -> list[dict]:
    """用最新历史行情回填已经发生的预测结果。"""
    records = load_feedback(path)
    if not records:
        return records

    if hist_df is None:
        try:
            hist_df = fetch_prediction_history(stock_code)
        except Exception:
            return records

    if hist_df.empty or "date" not in hist_df.columns or "close" not in hist_df.columns:
        return records

    price_map = {
        pd.to_datetime(row["date"]).strftime("%Y-%m-%d"): float(row["close"])
        for _, row in hist_df.iterrows()
        if pd.notna(row.get("date")) and pd.notna(row.get("close"))
    }
    changed = False

    for record in records:
        if record.get("code") != stock_code or record.get("status") == "evaluated":
            continue
        base_date = record.get("base_date")
        target_date = record.get("target_date")
        if base_date not in price_map or target_date not in price_map:
            continue
        base_close = price_map[base_date]
        target_close = price_map[target_date]
        if not base_close:
            continue

        actual_return = target_close / base_close - 1
        actual = _label_from_return(actual_return, threshold)
        record.update(
            {
                "actual": actual,
                "actual_return": actual_return,
                "is_correct": actual == record.get("predicted"),
                "status": "evaluated",
                "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        changed = True

    if changed:
        save_feedback(records, path)
    return records


def summarize_feedback(stock_code: str, path: Path = FEEDBACK_FILE) -> dict:
    """统计预测复盘结果、分周期准确率和常见错误类型。"""
    records = [item for item in load_feedback(path) if item.get("code") == stock_code]
    evaluated = [item for item in records if item.get("status") == "evaluated"]
    pending = [item for item in records if item.get("status") != "evaluated"]

    if not evaluated:
        return {
            "total": len(records),
            "evaluated": 0,
            "pending": len(pending),
            "accuracy": None,
            "by_horizon": [],
            "mistakes": [],
            "bias": [],
            "summary": "暂无已完成复盘的预测记录。等后续交易日行情更新后，系统会自动核对预测与实际走势。",
        }

    accuracy = sum(1 for item in evaluated if item.get("is_correct")) / len(evaluated)
    by_horizon = []
    for horizon in sorted({item.get("horizon") for item in evaluated}):
        group = [item for item in evaluated if item.get("horizon") == horizon]
        acc = sum(1 for item in group if item.get("is_correct")) / len(group)
        by_horizon.append({"周期": horizon, "样本数": len(group), "准确率": acc})

    mistakes = [item for item in evaluated if not item.get("is_correct")]
    mistake_pairs: dict[str, int] = {}
    for item in mistakes:
        key = f"{item.get('predicted')} -> {item.get('actual')}"
        mistake_pairs[key] = mistake_pairs.get(key, 0) + 1
    bias = [{"错判类型": key, "次数": value} for key, value in sorted(mistake_pairs.items(), key=lambda x: x[1], reverse=True)]

    if bias:
        top = bias[0]
        summary = (
            f"当前最常见错判是 {top['错判类型']}，出现 {top['次数']} 次。"
            "后续模型调参时应降低该方向的过度置信，并重点观察对应的量价和动能条件。"
        )
    else:
        summary = "已复盘样本暂未出现错误记录，样本量继续增加后判断会更可靠。"

    return {
        "total": len(records),
        "evaluated": len(evaluated),
        "pending": len(pending),
        "accuracy": accuracy,
        "by_horizon": by_horizon,
        "mistakes": mistakes[-10:],
        "bias": bias[:5],
        "summary": summary,
    }


def apply_feedback_calibration(stock_code: str, prediction_result: dict, path: Path = FEEDBACK_FILE) -> dict:
    """根据历史复盘错误对本次预测概率做保守校准。"""
    records = [
        item
        for item in load_feedback(path)
        if item.get("code") == stock_code and item.get("status") == "evaluated" and item.get("predicted")
    ]
    if len(records) < 6:
        return prediction_result

    stats: dict[str, dict] = {}
    for item in records:
        predicted = item.get("predicted")
        actual = item.get("actual")
        bucket = stats.setdefault(predicted, {"total": 0, "wrong": 0, "actual": {}})
        bucket["total"] += 1
        if not item.get("is_correct"):
            bucket["wrong"] += 1
            bucket["actual"][actual] = bucket["actual"].get(actual, 0) + 1

    prob_keys = {"上涨": "上涨概率", "震荡": "震荡概率", "下跌": "下跌概率"}
    adjusted = False
    for item in prediction_result.get("predictions", []):
        direction = item.get("预测方向")
        stat = stats.get(direction)
        if not stat or stat["total"] < 3:
            continue
        error_rate = stat["wrong"] / stat["total"]
        if error_rate < 0.55:
            continue

        penalty = min(0.12, max(0.03, (error_rate - 0.5) * 0.20))
        target = max(stat["actual"], key=stat["actual"].get) if stat["actual"] else "震荡"
        direction_key = prob_keys.get(direction)
        target_key = prob_keys.get(target, "震荡概率")
        if direction_key not in item or target_key not in item:
            continue

        item[direction_key] = max(0.01, item[direction_key] - penalty)
        item[target_key] = item[target_key] + penalty
        total_prob = item["上涨概率"] + item["震荡概率"] + item["下跌概率"]
        for key in ["上涨概率", "震荡概率", "下跌概率"]:
            item[key] = item[key] / total_prob

        direction_after = max(["上涨", "震荡", "下跌"], key=lambda label: item[prob_keys[label]])
        item["预测方向"] = direction_after
        item["置信度"] = "低" if item.get("置信度") == "中" else item.get("置信度", "低")
        item["预测依据"] = (
            f"{item.get('预测依据', '')}；复盘校准：历史上“{direction}”方向错判率较高，"
            f"已保守下调该方向置信并向“{target}”回拨。"
        )
        item["模型"] = f"{item.get('模型', '模型')} + 复盘校准"
        adjusted = True

    if adjusted:
        prediction_result["model_note"] = (
            f"{prediction_result.get('model_note', '')} 系统已结合本地历史复盘记录做保守概率校准；"
            "复盘样本不足时不会启用该校准。"
        )
    return prediction_result
