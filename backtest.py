import pandas as pd
import plotly.graph_objects as go

from prediction import (
    ADAPTIVE_THRESHOLD_VERSION,
    FEATURE_COLUMNS,
    LABELS,
    add_prediction_labels,
    build_features,
    fetch_prediction_history,
    _blend_probabilities,
    _confidence_from_probs,
    _extract_ensemble_probs,
    _extract_ensemble_probability_frame,
    _fit_model_ensemble,
    _refit_model_ensemble,
    _recent_label_probs,
    _rule_probabilities,
    _select_prediction_direction,
)


def _safe_accuracy(df: pd.DataFrame) -> float | None:
    """计算准确率，空数据返回 None。"""
    if df.empty:
        return None
    return float((df["actual"] == df["predicted"]).mean())


def _safe_balanced_accuracy(df: pd.DataFrame) -> float | None:
    """计算各类别召回率的平均值，避免多数类准确率掩盖方向失灵。"""
    if df.empty:
        return None
    recalls = []
    for label in LABELS:
        actual_label = df[df["actual"] == label]
        if not actual_label.empty:
            recalls.append(float((actual_label["predicted"] == label).mean()))
    return float(sum(recalls) / len(recalls)) if recalls else None


def _direction_distribution(df: pd.DataFrame, column: str) -> dict:
    """返回固定顺序的三分类占比。"""
    if df.empty or column not in df.columns:
        return {label: 0.0 for label in LABELS}
    counts = df[column].value_counts(normalize=True)
    return {label: float(counts.get(label, 0.0)) for label in LABELS}


def _fit_predict_window(
    train_df: pd.DataFrame,
    test_row: pd.Series,
    label_col: str,
    horizon: int,
    fitted_model=None,
    ensemble_template=None,
) -> tuple[dict, object | None]:
    """按需训练与线上一致的时间隔离集成，并返回单个历史预测。"""
    if fitted_model is None:
        train_df = train_df.dropna(subset=FEATURE_COLUMNS + [label_col])
    if fitted_model is None and (len(train_df) < 160 or train_df[label_col].nunique() < 2):
        return {
            "predicted": "震荡",
            "up_prob": 0.2,
            "flat_prob": 0.6,
            "down_prob": 0.2,
            "confidence": "低",
            "model": "历史回放规则回退",
        }, None
    model = fitted_model
    if model is None:
        if ensemble_template is None:
            model = _fit_model_ensemble(train_df[FEATURE_COLUMNS], train_df[label_col], horizon)
        else:
            model = _refit_model_ensemble(ensemble_template, train_df[FEATURE_COLUMNS], train_df[label_col])
    if model is None:
        return {
            "predicted": "震荡",
            "up_prob": 0.2,
            "flat_prob": 0.6,
            "down_prob": 0.2,
            "confidence": "低",
            "model": "历史回放规则回退",
        }, None
    model_probs = _extract_ensemble_probs(model, test_row)
    recent_probs = _recent_label_probs(train_df, label_col)
    rule_probs, _, _, _ = _rule_probabilities(test_row, horizon)
    probabilities = _blend_probabilities(model_probs, recent_probs, rule_probs, model["cv_score"])
    predicted, decision_reason = _select_prediction_direction(
        probabilities,
        model_probs=model_probs,
        rule_probs=rule_probs,
        model_score=model["cv_score"],
    )
    return {
        "predicted": predicted,
        "up_prob": probabilities["上涨"],
        "flat_prob": probabilities["震荡"],
        "down_prob": probabilities["下跌"],
        "confidence": _confidence_from_probs(probabilities, model["cv_score"], low_reliability=False),
        "model": f"历史滚动时间隔离集成（{model['name']}）",
        "validation_balanced_accuracy": model["cv_score"],
        "validation_log_loss": model["cv_log_loss"],
        "decision_reason": decision_reason,
    }, model


def rolling_backtest(
    stock_code: str,
    threshold: float = 0.01,
    test_days: int = 120,
    train_window: int = 500,
    retrain_interval: int = 20,
    threshold_mode: str = "manual",
) -> dict:
    """使用历史滚动方式回测 1、2、3 日预测效果，避免数据泄漏。"""
    hist_df = fetch_prediction_history(stock_code)
    normalized_mode = "adaptive" if threshold_mode == "adaptive" else "manual"
    feature_df = add_prediction_labels(
        build_features(hist_df),
        threshold=threshold,
        threshold_mode=normalized_mode,
    )
    for horizon in [1, 2, 3]:
        feature_df[f"target_date_{horizon}d"] = feature_df["date"].shift(-horizon)
    data = feature_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    results = {}
    for horizon in [1, 2, 3]:
        label_col = f"label_{horizon}d"
        target_date_col = f"target_date_{horizon}d"
        return_col = f"future_return_{horizon}d"
        usable = data.dropna(subset=[label_col, target_date_col, return_col]).reset_index(drop=True)
        rows = []
        minimum_history = 160 + horizon - 1
        start = max(minimum_history, len(usable) - test_days)
        fitted_model = None
        ensemble_template = None
        model_train_samples = 0
        retrain_count = 0
        idx = start
        while idx < len(usable):
            # 在测试日只能使用目标结果已经发生的训练标签。
            train_end = idx - horizon + 1
            train_start = max(0, train_end - train_window)
            training_slice = usable.iloc[train_start:train_end].dropna(subset=FEATURE_COLUMNS + [label_col])
            model_train_samples = len(training_slice)
            retrain_count += 1
            if retrain_count == 1 or (retrain_count - 1) % 3 == 0:
                ensemble_template = None
            if len(training_slice) < 160 or training_slice[label_col].nunique() < 2:
                fitted_model = None
            elif ensemble_template is None:
                fitted_model = _fit_model_ensemble(
                    training_slice[FEATURE_COLUMNS],
                    training_slice[label_col],
                    horizon,
                )
            else:
                fitted_model = _refit_model_ensemble(
                    ensemble_template,
                    training_slice[FEATURE_COLUMNS],
                    training_slice[label_col],
                )
            if fitted_model is not None:
                ensemble_template = fitted_model
            segment_end = min(len(usable), idx + max(1, retrain_interval))
            segment = usable.iloc[idx:segment_end]
            probability_frame = (
                _extract_ensemble_probability_frame(fitted_model, segment[FEATURE_COLUMNS])
                if fitted_model is not None
                else None
            )
            for test_idx in range(idx, segment_end):
                test_row = usable.iloc[test_idx]
                current_train_end = test_idx - horizon + 1
                current_train_start = max(0, current_train_end - train_window)
                current_train = usable.iloc[current_train_start:current_train_end]
                if fitted_model is None or probability_frame is None:
                    probabilities = {"上涨": 0.2, "震荡": 0.6, "下跌": 0.2}
                    model_probs = probabilities.copy()
                    recent_probs = probabilities.copy()
                    rule_probs = probabilities.copy()
                    predicted = "震荡"
                    confidence = "低"
                    model_name = "历史回放规则回退"
                    validation_accuracy = None
                    validation_loss = None
                else:
                    model_probs = probability_frame.loc[test_row.name].to_dict()
                    recent_probs = _recent_label_probs(current_train, label_col)
                    rule_probs, _, _, _ = _rule_probabilities(test_row, horizon)
                    probabilities = _blend_probabilities(
                        model_probs,
                        recent_probs,
                        rule_probs,
                        fitted_model["cv_score"],
                    )
                    predicted, decision_reason = _select_prediction_direction(
                        probabilities,
                        model_probs=model_probs,
                        rule_probs=rule_probs,
                        model_score=fitted_model["cv_score"],
                    )
                    confidence = _confidence_from_probs(
                        probabilities,
                        fitted_model["cv_score"],
                        low_reliability=False,
                    )
                    model_name = f"历史滚动时间隔离集成（{fitted_model['name']}）"
                    validation_accuracy = fitted_model["cv_score"]
                    validation_loss = fitted_model["cv_log_loss"]
                if fitted_model is None or probability_frame is None:
                    decision_reason = "历史训练样本不足，使用保守震荡回退"
                rows.append(
                    {
                        "date": test_row["date"],
                        "target_date": test_row[target_date_col],
                        "predicted": predicted,
                        "actual": str(test_row[label_col]),
                        "actual_return": float(test_row[return_col]),
                        "horizon": horizon,
                        "up_prob": probabilities["上涨"],
                        "flat_prob": probabilities["震荡"],
                        "down_prob": probabilities["下跌"],
                        "model_up_prob": model_probs["上涨"],
                        "model_flat_prob": model_probs["震荡"],
                        "model_down_prob": model_probs["下跌"],
                        "recent_up_prob": recent_probs["上涨"],
                        "recent_flat_prob": recent_probs["震荡"],
                        "recent_down_prob": recent_probs["下跌"],
                        "rule_up_prob": rule_probs["上涨"],
                        "rule_flat_prob": rule_probs["震荡"],
                        "rule_down_prob": rule_probs["下跌"],
                        "confidence": confidence,
                        "model": model_name,
                        "validation_balanced_accuracy": validation_accuracy,
                        "validation_log_loss": validation_loss,
                        "decision_reason": decision_reason,
                        "train_samples": model_train_samples,
                        "label_threshold": float(test_row.get(f"label_threshold_{horizon}d", threshold)),
                        "threshold_mode": normalized_mode,
                        "label_version": ADAPTIVE_THRESHOLD_VERSION if normalized_mode == "adaptive" else "fixed_v1",
                    }
                )
            idx = segment_end
        result_df = pd.DataFrame(rows)
        results[horizon] = result_df

    combined = pd.concat(results.values(), ignore_index=True) if results else pd.DataFrame()
    recent_60 = combined.groupby("horizon").tail(60) if not combined.empty else pd.DataFrame()
    recent_120 = combined.groupby("horizon").tail(120) if not combined.empty else pd.DataFrame()
    accuracy_by_horizon = {f"{h}日预测准确率": _safe_accuracy(results[h]) for h in [1, 2, 3]}
    confusion = pd.crosstab(combined["actual"], combined["predicted"], rownames=["实际走势"], colnames=["预测走势"]).reindex(index=LABELS, columns=LABELS, fill_value=0) if not combined.empty else pd.DataFrame(index=LABELS, columns=LABELS).fillna(0)
    prediction_distribution = _direction_distribution(combined, "predicted")
    actual_distribution = _direction_distribution(combined, "actual")
    flat_prediction_ratio = prediction_distribution["震荡"]
    return {
        "rows": len(hist_df),
        "accuracy_60": _safe_accuracy(recent_60),
        "accuracy_120": _safe_accuracy(recent_120),
        "accuracy_by_horizon": accuracy_by_horizon,
        "balanced_accuracy": _safe_balanced_accuracy(combined),
        "prediction_distribution": prediction_distribution,
        "actual_distribution": actual_distribution,
        "collapse_warning": flat_prediction_ratio >= 0.85,
        "confusion_matrix": confusion,
        "results": combined,
        "threshold_mode": normalized_mode,
        "label_version": ADAPTIVE_THRESHOLD_VERSION if normalized_mode == "adaptive" else "fixed_v1",
        "note": (
            "回测使用滚动训练方式，每个测试点只使用当时已经发生并可获得结果的历史标签训练，"
            f"2-3 日标签也会等待对应周期结束后才进入训练；模型每 {max(1, retrain_interval)} 个交易日重新训练一次以控制耗时。"
            "预测与线上一致，使用带标签周期隔离区的时间序列验证、近期状态加权和双模型集成。"
            "回测中每 3 次重训重新选择一次模型组合，其余重训沿用最近一次组合以控制计算耗时。"
            "历史表现不代表未来一定准确。"
        ),
    }


def build_backtest_figure(results: pd.DataFrame) -> go.Figure:
    """构建实际走势 vs 预测走势对比图。"""
    fig = go.Figure()
    if results is None or results.empty:
        return fig
    score_map = {"上涨": 1, "震荡": 0, "下跌": -1}
    chart_df = results[results["horizon"] == 1].tail(120).copy()
    chart_df["date_label"] = pd.to_datetime(chart_df["date"]).dt.strftime("%Y-%m-%d")
    chart_df["actual_score"] = chart_df["actual"].map(score_map)
    chart_df["predicted_score"] = chart_df["predicted"].map(score_map)
    fig.add_trace(go.Scatter(x=chart_df["date_label"], y=chart_df["actual_score"], mode="lines+markers", name="实际走势"))
    fig.add_trace(go.Scatter(x=chart_df["date_label"], y=chart_df["predicted_score"], mode="lines+markers", name="预测走势"))
    fig.update_layout(height=420, yaxis=dict(tickmode="array", tickvals=[-1, 0, 1], ticktext=["下跌", "震荡", "上涨"]), hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20))
    fig.update_xaxes(type="category", nticks=12)
    return fig
