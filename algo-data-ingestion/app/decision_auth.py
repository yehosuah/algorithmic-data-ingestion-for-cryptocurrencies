from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Dict, Mapping, Optional, Tuple

SIGNATURE_ALGORITHM = "HMAC-SHA256"
SIGNATURE_FIELD = "decision_signature"
SIGNATURE_ALGORITHM_FIELD = "decision_signature_algorithm"
_SIGNED_PAYLOAD_FIELDS = (SIGNATURE_FIELD, SIGNATURE_ALGORITHM_FIELD)


def _clean_secret(secret: Optional[str]) -> Optional[str]:
    if secret is None:
        return None
    value = str(secret).strip()
    return value or None


def _unsigned_payload(message: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in message.items() if key not in _SIGNED_PAYLOAD_FIELDS}


def canonical_decision_payload(message: Mapping[str, Any]) -> bytes:
    """
    Serialize a decision message deterministically for HMAC signing.

    The signature metadata fields are excluded so the same helper can be used for
    both initial signing and later verification.
    """
    return json.dumps(
        _unsigned_payload(message),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def sign_decision_message(message: Mapping[str, Any], secret: Optional[str]) -> Dict[str, Any]:
    """Return a signed copy of *message* when a decision HMAC secret is configured."""
    signed = dict(message)
    secret_value = _clean_secret(secret)
    if not secret_value:
        return signed
    digest = hmac.new(
        secret_value.encode("utf-8"),
        canonical_decision_payload(signed),
        hashlib.sha256,
    ).hexdigest()
    signed[SIGNATURE_ALGORITHM_FIELD] = SIGNATURE_ALGORITHM
    signed[SIGNATURE_FIELD] = digest
    return signed


def verify_decision_message(message: Mapping[str, Any], secret: Optional[str]) -> Tuple[bool, str]:
    """Verify a signed decision message.

    Returns (True, "ok") when verification succeeds. If no secret is configured,
    messages are accepted for dry-run/demo compatibility and the caller is
    expected to prohibit live trading without a secret.
    """
    secret_value = _clean_secret(secret)
    if not secret_value:
        return True, "decision_hmac_not_configured"
    if message.get(SIGNATURE_ALGORITHM_FIELD) != SIGNATURE_ALGORITHM:
        return False, "missing_or_unsupported_decision_signature_algorithm"
    supplied = message.get(SIGNATURE_FIELD)
    if not isinstance(supplied, str) or not supplied.strip():
        return False, "missing_decision_signature"
    expected = hmac.new(
        secret_value.encode("utf-8"),
        canonical_decision_payload(message),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(supplied.strip(), expected):
        return False, "invalid_decision_signature"
    return True, "ok"
