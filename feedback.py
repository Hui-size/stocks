import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from prediction import fetch_prediction_history


FEEDBACK_FILE = Path("prediction_feedback.json")
HISTORICAL_WEIGHT_HALF_LIFE_DAYS = 180
HISTORICAL_BOOTSTRAP_WEIGHT_CAP = 12.0
HISTORICAL_REAL_SAMPLE_RATIO_CAP = 0.30


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


def _sample_source(record: dict) -> str:
    """兼容旧记录，返回统一的样本来源名称。"""
    return str(record.get("source") or "日常预测")


def _sample_weight(record: dict) -> float:
    """历史推演采用较低权重，避免大量回放样本覆盖真实预测表现。"""
    default = 0.35 if _sample_source(record) == "历史滚动推演" else 1.0
    try:
        return max(0.0, float(record.get("sample_weight", default)))
    except (TypeError, ValueError):
        return default


def _calibration_weight_map(records: list[dict]) -> dict[str, float]:
    """按时间衰减历史样本，并限制其相对真实预测的总影响。"""
    daily_records = [item for item in records if _sample_source(item) != "历史滚动推演"]
    historical_records = [item for item in records if _sample_source(item) == "历史滚动推演"]
    weights = {_record_key(item): _sample_weight(item) for item in daily_records}
    if not historical_records:
        return weights

    parsed_dates = [pd.to_datetime(item.get("base_date"), errors="coerce") for item in historical_records]
    valid_dates = [value for value in parsed_dates if pd.notna(value)]
    latest_date = max(valid_dates) if valid_dates else pd.Timestamp.today().normalize()
    raw_historical_weights = {}
    for item, base_date in zip(historical_records, parsed_dates):
        age_days = max(0, int((latest_date - base_date).days)) if pd.notna(base_date) else HISTORICAL_WEIGHT_HALF_LIFE_DAYS
        recency_weight = 0.5 ** (age_days / HISTORICAL_WEIGHT_HALF_LIFE_DAYS)
        raw_historical_weights[_record_key(item)] = _sample_weight(item) * recency_weight

    daily_total = sum(weights.values())
    historical_cap = max(
        HISTORICAL_BOOTSTRAP_WEIGHT_CAP,
        daily_total * HISTORICAL_REAL_SAMPLE_RATIO_CAP,
    )
    raw_total = sum(raw_historical_weights.values())
    scale = min(1.0, historical_cap / raw_total) if raw_total > 0 else 0.0
    weights.update({key: value * scale for key, value in raw_historical_weights.items()})
    return weights


def _label_version(record: dict) -> str:
    """兼容旧记录；没有版本字段的历史记录按固定阈值第一版处理。"""
    return str(record.get("label_version") or "fixed_v1")


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
            "confidence_reason": item.get("置信度说明"),
            "model": item.get("模型"),
            "basis": item.get("预测依据"),
            "label_threshold": item.get("判断阈值"),
            "threshold_mode": prediction_result.get("threshold_mode", "manual"),
            "label_version": item.get("标签版本", prediction_result.get("label_version", "fixed_v1")),
            "source": "日常预测",
            "sample_weight": 1.0,
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


def get_today_prediction_record_status(stock_code: str, path: Path = FEEDBACK_FILE) -> dict:
    """检查指定股票今天是否已经完整记录过 1-3 日预测。"""
    today = datetime.now().strftime("%Y-%m-%d")
    records = [
        item
        for item in load_feedback(path)
        if item.get("code") == stock_code
        and _sample_source(item) != "历史滚动推演"
        and str(item.get("created_at", "")).startswith(today)
    ]
    horizons = {item.get("horizon") for item in records if item.get("horizon")}
    base_dates = sorted({item.get("base_date") for item in records if item.get("base_date")})
    return {
        "has_full_today_record": {"1日", "2日", "3日"}.issubset(horizons),
        "record_count": len(records),
        "horizons": sorted(horizons),
        "latest_base_date": base_dates[-1] if base_dates else "",
    }


def _label_from_return(value: float, threshold: float) -> str:
    """根据区间收益率生成上涨、震荡、下跌标签。"""
    if value > threshold:
        return "上涨"
    if value < -threshold:
        return "下跌"
    return "震荡"


def _classify_error_reason(record: dict, base_row: pd.Series | None, actual_return: float, threshold: float) -> str:
    """给错判样本做轻量归因，方便后续复盘。"""
    predicted = record.get("predicted")
    actual = record.get("actual")
    reasons = []

    if predicted and actual and predicted != actual:
        if predicted == "震荡" and actual in ["上涨", "下跌"]:
            reasons.append("震荡误判成方向行情")
        elif predicted in ["上涨", "下跌"] and actual == "震荡":
            reasons.append("方向行情未延续")
        else:
            reasons.append("方向反转识别不足")

    if abs(actual_return) <= threshold * 1.25:
        reasons.append("结果接近阈值边界")

    if base_row is not None:
        volume_ratio = base_row.get("volume_ratio_20d")
        if pd.notna(volume_ratio) and float(volume_ratio) >= 1.8:
            reasons.append("基准日放量异动")
        volatility = base_row.get("volatility_20d")
        if pd.notna(volatility) and float(volatility) >= max(threshold, 0.025):
            reasons.append("高波动阶段")
        ma20 = base_row.get("MA20")
        close = base_row.get("close")
        if pd.notna(ma20) and pd.notna(close):
            distance = abs(float(close) / float(ma20) - 1)
            if distance <= 0.01:
                reasons.append("靠近MA20趋势分界")

    return "；".join(dict.fromkeys(reasons)) or "暂未归类"


def record_historical_replay(
    stock_code: str,
    results: pd.DataFrame,
    threshold: float,
    path: Path = FEEDBACK_FILE,
) -> dict:
    """把无未来数据泄漏的滚动回测结果写入复盘系统。"""
    if results is None or results.empty:
        return {"added": 0, "skipped": 0, "total": 0, "effective_added": 0.0, "effective_total": 0.0}

    records = load_feedback(path)
    before_group = [
        item for item in records if item.get("code") == stock_code and item.get("status") == "evaluated"
    ]
    before_weights = _calibration_weight_map(before_group)
    before_historical_weight = sum(
        before_weights.get(_record_key(item), 0.0)
        for item in before_group
        if _sample_source(item) == "历史滚动推演"
    )
    existing = {_record_key(item) for item in records}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added = 0
    skipped = 0

    for _, row in results.iterrows():
        base_date = pd.to_datetime(row.get("date"), errors="coerce")
        target_date = pd.to_datetime(row.get("target_date"), errors="coerce")
        if pd.isna(base_date) or pd.isna(target_date):
            skipped += 1
            continue

        predicted = str(row.get("predicted") or "")
        actual = str(row.get("actual") or "")
        if predicted not in ["上涨", "震荡", "下跌"] or actual not in ["上涨", "震荡", "下跌"]:
            skipped += 1
            continue

        try:
            horizon = int(row.get("horizon"))
            actual_return = float(row.get("actual_return"))
        except (TypeError, ValueError):
            skipped += 1
            continue

        is_correct = predicted == actual
        classified_record = {"predicted": predicted, "actual": actual}
        record = {
            "code": stock_code,
            "base_date": base_date.strftime("%Y-%m-%d"),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "horizon": f"{horizon}日",
            "predicted": predicted,
            "actual": actual,
            "actual_return": actual_return,
            "is_correct": is_correct,
            "error_reason": "" if is_correct else _classify_error_reason(classified_record, None, actual_return, threshold),
            "up_prob": float(row.get("up_prob", 0.0)),
            "flat_prob": float(row.get("flat_prob", 0.0)),
            "down_prob": float(row.get("down_prob", 0.0)),
            "confidence": str(row.get("confidence") or "低"),
            "confidence_reason": "历史滚动推演样本，仅使用该时点之前可获得的数据。",
            "model": str(row.get("model") or "历史滚动模型"),
            "basis": f"历史回放训练样本 {int(row.get('train_samples', 0))} 条；结果已由后续 {horizon} 个交易日验证。",
            "label_threshold": float(row.get("label_threshold", threshold)),
            "threshold_mode": str(row.get("threshold_mode") or "manual"),
            "label_version": str(row.get("label_version") or "fixed_v1"),
            "source": "历史滚动推演",
            "sample_weight": 0.35,
            "status": "evaluated",
            "created_at": now,
            "evaluated_at": now,
        }
        key = _record_key(record)
        if key in existing:
            skipped += 1
            continue
        records.append(record)
        existing.add(key)
        added += 1

    if added:
        save_feedback(records, path)
    after_group = [
        item for item in records if item.get("code") == stock_code and item.get("status") == "evaluated"
    ]
    after_weights = _calibration_weight_map(after_group)
    after_historical_weight = sum(
        after_weights.get(_record_key(item), 0.0)
        for item in after_group
        if _sample_source(item) == "历史滚动推演"
    )
    return {
        "added": added,
        "skipped": skipped,
        "total": len(results),
        "effective_added": max(0.0, after_historical_weight - before_historical_weight),
        "effective_total": after_historical_weight,
    }


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
    row_map = {
        pd.to_datetime(row["date"]).strftime("%Y-%m-%d"): row
        for _, row in hist_df.iterrows()
        if pd.notna(row.get("date"))
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
        try:
            record_threshold = float(record.get("label_threshold", threshold))
        except (TypeError, ValueError):
            record_threshold = threshold
        actual = _label_from_return(actual_return, record_threshold)
        is_correct = actual == record.get("predicted")
        base_row = row_map.get(base_date)
        classified_record = record.copy()
        classified_record["actual"] = actual
        record.update(
            {
                "actual": actual,
                "actual_return": actual_return,
                "is_correct": is_correct,
                "error_reason": "" if is_correct else _classify_error_reason(classified_record, base_row, actual_return, record_threshold),
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
    effective_weights = _calibration_weight_map(evaluated)
    by_source = []
    for source in sorted({_sample_source(item) for item in records}):
        group = [item for item in records if _sample_source(item) == source]
        evaluated_group = [item for item in group if item.get("status") == "evaluated"]
        source_accuracy = None
        if evaluated_group:
            source_accuracy = sum(1 for item in evaluated_group if item.get("is_correct")) / len(evaluated_group)
        by_source.append(
            {
                "样本来源": source,
                "样本数": len(group),
                "已复盘": len(evaluated_group),
                "准确率": source_accuracy,
                "校准权重": _sample_weight(group[0]) if group else 0.0,
                "有效权重": sum(effective_weights.get(_record_key(item), 0.0) for item in evaluated_group),
            }
        )

    by_label_version = []
    for version in sorted({_label_version(item) for item in records}):
        group = [item for item in records if _label_version(item) == version]
        evaluated_group = [item for item in group if item.get("status") == "evaluated"]
        version_accuracy = None
        if evaluated_group:
            version_accuracy = sum(1 for item in evaluated_group if item.get("is_correct")) / len(evaluated_group)
        by_label_version.append(
            {
                "标签版本": version,
                "样本数": len(group),
                "已复盘": len(evaluated_group),
                "准确率": version_accuracy,
            }
        )

    if not evaluated:
        return {
            "total": len(records),
            "evaluated": 0,
            "pending": len(pending),
            "accuracy": None,
            "by_horizon": [],
            "mistakes": [],
            "bias": [],
            "error_reasons": [],
            "by_source": by_source,
            "by_label_version": by_label_version,
            "summary": "暂无已完成复盘的预测记录。等后续交易日行情更新后，系统会自动核对预测与实际走势。",
        }

    total_effective_weight = sum(effective_weights.values())
    accuracy = (
        sum(effective_weights.get(_record_key(item), 0.0) for item in evaluated if item.get("is_correct"))
        / total_effective_weight
        if total_effective_weight > 0
        else None
    )
    by_horizon = []
    for horizon in sorted({item.get("horizon") for item in evaluated}):
        group = [item for item in evaluated if item.get("horizon") == horizon]
        group_weight = sum(effective_weights.get(_record_key(item), 0.0) for item in group)
        acc = (
            sum(effective_weights.get(_record_key(item), 0.0) for item in group if item.get("is_correct"))
            / group_weight
            if group_weight > 0
            else None
        )
        by_horizon.append({"周期": horizon, "样本数": len(group), "有效权重": group_weight, "准确率": acc})

    mistakes = [item for item in evaluated if not item.get("is_correct")]
    mistake_pairs: dict[str, int] = {}
    for item in mistakes:
        key = f"{item.get('predicted')} -> {item.get('actual')}"
        mistake_pairs[key] = mistake_pairs.get(key, 0) + 1
    bias = [{"错判类型": key, "次数": value} for key, value in sorted(mistake_pairs.items(), key=lambda x: x[1], reverse=True)]
    reason_counts: dict[str, int] = {}
    for item in mistakes:
        reasons = str(item.get("error_reason") or "暂未归类").split("；")
        for reason in reasons:
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    error_reasons = [
        {"错误原因": key, "次数": value}
        for key, value in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    ]

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
        "error_reasons": error_reasons[:5],
        "by_source": by_source,
        "by_label_version": by_label_version,
        "summary": summary,
    }


def build_feedback_export_table(stock_code: str, path: Path = FEEDBACK_FILE) -> pd.DataFrame:
    """构建指定股票的复盘记录导出表。"""
    records = [item for item in load_feedback(path) if item.get("code") == stock_code]
    rows = []
    for item in records:
        actual_return = item.get("actual_return")
        rows.append(
            {
                "股票代码": item.get("code", ""),
                "样本来源": _sample_source(item),
                "校准权重": _sample_weight(item),
                "预测基准日": item.get("base_date", ""),
                "目标日期": item.get("target_date", ""),
                "周期": item.get("horizon", ""),
                "预测方向": item.get("predicted", ""),
                "实际方向": item.get("actual", ""),
                "是否正确": item.get("is_correct", ""),
                "错误原因": item.get("error_reason", ""),
                "实际涨跌幅": "" if actual_return is None else actual_return,
                "上涨概率": item.get("up_prob", ""),
                "震荡概率": item.get("flat_prob", ""),
                "下跌概率": item.get("down_prob", ""),
                "置信度": item.get("confidence", ""),
                "置信度说明": item.get("confidence_reason", ""),
                "模型": item.get("model", ""),
                "预测依据": item.get("basis", ""),
                "状态": item.get("status", ""),
                "判断阈值": item.get("label_threshold", ""),
                "阈值模式": item.get("threshold_mode", "manual"),
                "标签版本": item.get("label_version", "fixed_v1"),
                "创建时间": item.get("created_at", ""),
                "复盘时间": item.get("evaluated_at", ""),
            }
        )
    return pd.DataFrame(rows)


def apply_feedback_calibration(stock_code: str, prediction_result: dict, path: Path = FEEDBACK_FILE) -> dict:
    """根据历史复盘错误对本次预测概率做保守校准。"""
    target_label_version = str(prediction_result.get("label_version") or "fixed_v1")
    records = [
        item
        for item in load_feedback(path)
        if item.get("code") == stock_code
        and item.get("status") == "evaluated"
        and item.get("predicted")
        and _label_version(item) == target_label_version
    ]
    calibration_weights = _calibration_weight_map(records)
    if sum(calibration_weights.values()) < 6:
        return prediction_result

    stats: dict[str, dict] = {}
    for item in records:
        predicted = item.get("predicted")
        actual = item.get("actual")
        weight = calibration_weights.get(_record_key(item), 0.0)
        bucket = stats.setdefault(predicted, {"total": 0, "wrong": 0, "actual": {}})
        bucket["total"] += weight
        if not item.get("is_correct"):
            bucket["wrong"] += weight
            bucket["actual"][actual] = bucket["actual"].get(actual, 0) + weight

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
        item["置信度说明"] = item.get("置信度说明", "已启用复盘校准。").replace("未启用复盘校准", "已启用复盘校准")
        adjusted = True

    if adjusted:
        prediction_result["model_note"] = (
            f"{prediction_result.get('model_note', '')} 系统已结合本地历史复盘记录做保守概率校准；"
            "历史滚动推演样本按 0.35 权重参与，复盘样本不足时不会启用该校准。"
        )
    return prediction_result
