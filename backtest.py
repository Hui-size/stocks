import pandas as pd
import plotly.graph_objects as go

from prediction import (
    ADAPTIVE_THRESHOLD_VERSION,
    FEATURE_COLUMNS,
    LABELS,
    add_prediction_labels,
    build_features,
    fetch_prediction_history,
)
from sklearn.ensemble import RandomForestClassifier


def _safe_accuracy(df: pd.DataFrame) -> float | None:
    """计算准确率，空数据返回 None。"""
    if df.empty:
        return None
    return float((df["actual"] == df["predicted"]).mean())


def _fit_predict_window(
    train_df: pd.DataFrame,
    test_row: pd.Series,
    label_col: str,
    fitted_model=None,
) -> tuple[dict, object | None]:
    """按需训练历史窗口模型，并返回单个样本的方向、概率和可复用模型。"""
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
        model = RandomForestClassifier(
            n_estimators=40,
            max_depth=5,
            min_samples_leaf=8,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(train_df[FEATURE_COLUMNS], train_df[label_col])
    probabilities = {label: 0.0 for label in LABELS}
    for label, value in zip(model.classes_, model.predict_proba(test_row[FEATURE_COLUMNS].to_frame().T)[0]):
        probabilities[str(label)] = float(value)
    predicted = max(probabilities, key=probabilities.get)
    top_probability = probabilities[predicted]
    confidence = "高" if top_probability >= 0.58 else ("中" if top_probability >= 0.43 else "低")
    return {
        "predicted": predicted,
        "up_prob": probabilities["上涨"],
        "flat_prob": probabilities["震荡"],
        "down_prob": probabilities["下跌"],
        "confidence": confidence,
        "model": "历史滚动随机森林",
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
        model_train_samples = 0
        for idx in range(start, len(usable)):
            # 在测试日只能使用目标结果已经发生的训练标签。
            train_end = idx - horizon + 1
            train_start = max(0, train_end - train_window)
            train_df = usable.iloc[train_start:train_end]
            test_row = usable.iloc[idx]
            should_retrain = fitted_model is None or (idx - start) % max(1, retrain_interval) == 0
            if should_retrain:
                fitted_model = None
                model_train_samples = len(train_df)
            prediction, fitted_model = _fit_predict_window(
                train_df,
                test_row,
                label_col,
                fitted_model=fitted_model,
            )
            rows.append(
                {
                    "date": test_row["date"],
                    "target_date": test_row[target_date_col],
                    "predicted": prediction["predicted"],
                    "actual": str(test_row[label_col]),
                    "actual_return": float(test_row[return_col]),
                    "horizon": horizon,
                    "up_prob": prediction["up_prob"],
                    "flat_prob": prediction["flat_prob"],
                    "down_prob": prediction["down_prob"],
                    "confidence": prediction["confidence"],
                    "model": prediction["model"],
                    "train_samples": model_train_samples,
                    "label_threshold": float(test_row.get(f"label_threshold_{horizon}d", threshold)),
                    "threshold_mode": normalized_mode,
                    "label_version": ADAPTIVE_THRESHOLD_VERSION if normalized_mode == "adaptive" else "fixed_v1",
                }
            )
        result_df = pd.DataFrame(rows)
        results[horizon] = result_df

    combined = pd.concat(results.values(), ignore_index=True) if results else pd.DataFrame()
    recent_60 = combined.groupby("horizon").tail(60) if not combined.empty else pd.DataFrame()
    recent_120 = combined.groupby("horizon").tail(120) if not combined.empty else pd.DataFrame()
    accuracy_by_horizon = {f"{h}日预测准确率": _safe_accuracy(results[h]) for h in [1, 2, 3]}
    confusion = pd.crosstab(combined["actual"], combined["predicted"], rownames=["实际走势"], colnames=["预测走势"]).reindex(index=LABELS, columns=LABELS, fill_value=0) if not combined.empty else pd.DataFrame(index=LABELS, columns=LABELS).fillna(0)
    return {
        "rows": len(hist_df),
        "accuracy_60": _safe_accuracy(recent_60),
        "accuracy_120": _safe_accuracy(recent_120),
        "accuracy_by_horizon": accuracy_by_horizon,
        "confusion_matrix": confusion,
        "results": combined,
        "threshold_mode": normalized_mode,
        "label_version": ADAPTIVE_THRESHOLD_VERSION if normalized_mode == "adaptive" else "fixed_v1",
        "note": (
            "回测使用滚动训练方式，每个测试点只使用当时已经发生并可获得结果的历史标签训练，"
            f"2-3 日标签也会等待对应周期结束后才进入训练；模型每 {max(1, retrain_interval)} 个交易日重新训练一次以控制耗时。"
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
