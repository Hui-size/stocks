import realtime


class FakeResponse:
    content = (
        'v_sh600519="1~贵州茅台~600519~1341.99~1355.27~1355.00~29853~12760~17093~'
        '1341.98~1~1341.90~2~1341.69~1~1341.68~1~1341.62~3~1341.99~283~1342.00~23~'
        '1342.01~1~1342.02~2~1342.06~3~~20260814161443~-13.28~-0.98~1356.80~1336.20~'
        '1341.99/29900/4024000000~29900~402400~"'
    ).encode("gbk")

    def raise_for_status(self):
        return None


def test_realtime_quote_uses_single_stock_endpoint(monkeypatch):
    captured = {}

    def fake_get(url, timeout):
        captured.update({"url": url, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(realtime.requests, "get", fake_get)
    quote = realtime.fetch_realtime_quote("600519")

    assert captured["url"].endswith("q=sh600519")
    assert captured["timeout"] == 4
    assert quote["name"] == "贵州茅台"
    assert quote["latest_price"] == 1341.99
    assert quote["pct_chg"] == -0.98
    assert quote["amount"] == 4024000000
    assert quote["source"] == "腾讯单股票实时行情"


def test_realtime_quote_rejects_empty_payload(monkeypatch):
    monkeypatch.setattr(
        realtime.requests,
        "get",
        lambda *_args, **_kwargs: type(
            "R",
            (),
            {"content": b'v_sz000001=""', "raise_for_status": lambda self: None},
        )(),
    )

    try:
        realtime.fetch_realtime_quote("000001")
    except realtime.RealtimeQuoteError as exc:
        assert "返回为空" in str(exc)
    else:
        raise AssertionError("空实时行情应抛出 RealtimeQuoteError")
