from app.decision_auth import sign_decision_message, verify_decision_message


def test_decision_auth_signs_and_detects_tampering():
    message = {
        "model": "xgb_primary",
        "symbol": "BTC/USDT",
        "timestamp": "2025-10-01T00:00:00+00:00",
        "probability": 0.9,
        "gate_pass": True,
    }
    signed = sign_decision_message(message, "test-secret")

    assert signed["decision_signature_algorithm"] == "HMAC-SHA256"
    assert verify_decision_message(signed, "test-secret") == (True, "ok")

    tampered = dict(signed)
    tampered["probability"] = 0.1
    ok, reason = verify_decision_message(tampered, "test-secret")
    assert ok is False
    assert reason == "invalid_decision_signature"
