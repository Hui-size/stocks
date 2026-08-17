import pandas as pd

from app import build_prediction_trend_figure, get_plotly_config, prediction_session_key, select_watchlist_stock


def test_prediction_trend_chart_separates_history_and_forecast():
    history = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-07-01", periods=20),
            "close": [100 + index * 0.5 for index in range(20)],
        }
    )
    result = {
        "data": history,
        "predictions": [
            {"预测日期": "2026-07-29", "周期": "1日", "预测方向": "上涨", "预测价位区间": (109.5, 111.5)},
            {"预测日期": "2026-07-30", "周期": "2日", "预测方向": "震荡", "预测价位区间": (109.0, 112.0)},
            {"预测日期": "2026-07-31", "周期": "3日", "预测方向": "下跌", "预测价位区间": (108.0, 111.0)},
        ],
    }

    figure = build_prediction_trend_figure(result)

    assert figure is not None
    traces = {trace.name: trace for trace in figure.data if trace.name}
    assert traces["历史收盘价"].line.dash == "solid"
    assert traces["预测情景中枢"].line.dash == "dash"
    assert traces["预测价位区间"].fill == "tonexty"
    assert len(traces["预测情景中枢"].x) == 4
    assert figure.layout.legend.font.color


def test_prediction_chart_supports_deleting_one_drawn_line_at_a_time():
    config = get_plotly_config(include_shape_eraser=True)

    assert "drawline" in config["modeBarButtonsToAdd"]
    assert "eraseshape" in config["modeBarButtonsToAdd"]


def test_watchlist_selection_callback_updates_both_stock_states():
    state = {"selected_stock": "600519", "stock_code_input": "600519"}

    select_watchlist_stock("002463", state=state)

    assert state["selected_stock"] == "002463"
    assert state["stock_code_input"] == "002463"


def test_app_prediction_session_key_switches_at_shanghai_close():
    assert prediction_session_key("2026-08-17 14:59:59+08:00") == "2026-08-17|before_close"
    assert prediction_session_key("2026-08-17 15:00:00+08:00") == "2026-08-17|after_close"
