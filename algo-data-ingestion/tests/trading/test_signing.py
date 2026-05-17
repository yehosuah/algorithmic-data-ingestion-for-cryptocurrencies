from app.trading.signing import sign_decision_payload, verify_decision_payload


def test_decision_hmac_signing_round_trip():
    message = {
        "model": "xgb_primary",
        "symbol": "ETH/USDT",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "probability": 0.9,
    }
    sign_decision_payload(message, "secret")
    assert verify_decision_payload(message, "secret") is True

    message["probability"] = 0.1
    assert verify_decision_payload(message, "secret") is False


def test_decision_hmac_requires_signature_and_secret():
    assert verify_decision_payload({"model": "xgb_primary"}, "secret") is False
    assert verify_decision_payload({"model": "xgb_primary", "signature": "abc"}, "") is False
