import pandas as pd

from feedback import (
    HISTORICAL_BOOTSTRAP_WEIGHT_CAP,
    _calibration_weight_map,
    _record_key,
    record_historical_replay,
)


def make_record(index: int, source: str, base_date: pd.Timestamp) -> dict:
    return {
        "code": "600519",
        "base_date": base_date.strftime("%Y-%m-%d"),
        "target_date": (base_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "horizon": f"{index % 3 + 1}日",
        "source": source,
        "sample_weight": 0.35 if source == "历史滚动推演" else 1.0,
        "status": "evaluated",
    }


def test_historical_weights_decay_and_are_capped():
    latest = pd.Timestamp("2026-08-14")
    daily = [make_record(index, "日常预测", latest - pd.Timedelta(days=index)) for index in range(10)]
    historical = [
        make_record(100 + index, "历史滚动推演", latest - pd.Timedelta(days=100 + index * 8))
        for index in range(80)
    ]
    records = daily + historical
    weights = _calibration_weight_map(records)
    historical_total = sum(weights[_record_key(item)] for item in historical)

    assert historical_total <= HISTORICAL_BOOTSTRAP_WEIGHT_CAP + 1e-9
    assert weights[_record_key(historical[0])] > weights[_record_key(historical[-1])]
    assert all(weights[_record_key(item)] == 1.0 for item in daily)


def test_historical_replay_reports_capped_effective_weight(tmp_path):
    dates = pd.bdate_range("2026-01-05", periods=60)
    results = pd.DataFrame(
        {
            "date": dates,
            "target_date": dates + pd.offsets.BDay(1),
            "horizon": [index % 3 + 1 for index in range(60)],
            "predicted": ["震荡"] * 60,
            "actual": ["震荡"] * 60,
            "actual_return": [0.0] * 60,
            "label_threshold": [0.01] * 60,
            "threshold_mode": ["adaptive"] * 60,
            "label_version": ["volatility_20d_v1"] * 60,
        }
    )

    outcome = record_historical_replay("600519", results, 0.01, path=tmp_path / "feedback.json")

    assert outcome["added"] == 60
    assert outcome["effective_total"] <= HISTORICAL_BOOTSTRAP_WEIGHT_CAP + 1e-9
