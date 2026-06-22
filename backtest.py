import pandas as pd
import plotly.graph_objects as go

from prediction import FEATURE_COLUMNS, LABELS, add_prediction_labels, build_features, fetch_prediction_history
from sklearn.ensemble import RandomForestClassifier


def _safe_accuracy(df: pd.DataFrame) -> float | None:
    """计算准确率，空数据返回 None。"""
    if df.empty:
        return None
    return float((df["actual"] == df["predicted"]).mean())


def _fit_predict_window(train_df: pd.DataFrame, test_row: pd.Series, label_col: str) -> str:
    """用历史窗口训练模型并预测单个测试样本。"""
    train_df = train_df.dropna(subset=FEATURE_COLUMNS + [label_col])
    if len(train_df) < 160 or train_df[label_col].nunique() < 2:
        return "震荡"
    model = RandomForestClassifier(n_estimators=40, max_depth=5, min_samples_leaf=8, random_state=42, class_weight="balanced")
    model.fit(train_df[FEATURE_COLUMNS], train_df[label_col])
    return str(model.predict(test_row[FEATURE_COLUMNS].to_frame().T)[0])


def rolling_backtest(stock_code: str, threshold: float = 0.01, test_days: int = 120, train_window: int = 500) -> dict:
    """使用历史滚动方式回测 1、2、3 日预测效果，避免数据泄漏。"""
    hist_df = fetch_prediction_history(stock_code)
    data = add_prediction_labels(build_features(hist_df), threshold=threshold).dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    results = {}
    for horizon in [1, 2, 3]:
        label_col = f"label_{horizon}d"
        usable = data.dropna(subset=[label_col]).reset_index(drop=True)
        rows = []
        start = max(train_window, len(usable) - test_days)
        for idx in range(start, len(usable)):
            train_start = max(0, idx - train_window)
            train_df = usable.iloc[train_start:idx]
            test_row = usable.iloc[idx]
            predicted = _fit_predict_window(train_df, test_row, label_col)
            rows.append({"date": test_row["date"], "predicted": predicted, "actual": test_row[label_col], "horizon": horizon})
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
        "note": "回测使用滚动训练方式，每个测试点只使用其之前的数据训练，历史表现不代表未来一定准确。",
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
