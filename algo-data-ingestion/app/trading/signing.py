from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Mapping, MutableMapping

SIGNATURE_FIELD = "signature"

def canonical_decision_payload(message: Mapping[str, Any]) -> bytes:
    """Return deterministic JSON bytes for decision HMAC signing."""
    unsigned = {k: v for k, v in message.items() if k != SIGNATURE_FIELD}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sign_decision_payload(message: MutableMapping[str, Any], secret: str) -> MutableMapping[str, Any]:
    """Attach a hex HMAC-SHA256 signature to a decision payload."""
    if not secret:
        raise ValueError("decision signing secret is required")
    digest = hmac.new(secret.encode("utf-8"), canonical_decision_payload(message), hashlib.sha256).hexdigest()
    message[SIGNATURE_FIELD] = digest
    return message


def verify_decision_payload(message: Mapping[str, Any], secret: str) -> bool:
    """Verify the HMAC-SHA256 signature on a decision payload."""
    if not secret:
        return False
    actual = message.get(SIGNATURE_FIELD)
    if not isinstance(actual, str) or not actual:
        return False
    expected = hmac.new(secret.encode("utf-8"), canonical_decision_payload(message), hashlib.sha256).hexdigest()
    return hmac.compare_digest(actual, expected)
