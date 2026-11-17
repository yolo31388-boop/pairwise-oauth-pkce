"""OAuth 2.0 / PKCE request-binding helpers.

Security properties enforced here:

* ``code_challenge`` is actually verified, and the ``plain`` and ``S256``
  transformations are kept distinct (an unknown ``code_challenge_method``
  is rejected).
* ``redirect_uri`` is matched exactly -- no prefix matching, so a trailing
  slash or a suffix such as ``.evil`` cannot redirect elsewhere.
* ``state`` and ``nonce`` must both be present on both sides and match.
* An authorization code can be redeemed at most once, atomically, so two
  concurrent requests cannot both win.
* Nothing here requires (or stores) a ``client_secret``; public clients are
  authenticated by PKCE and the client identifier only.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import threading
import time
from collections.abc import Mapping, MutableMapping
from typing import Any, Optional


# A single process-wide lock makes the "read used flag -> mark used" step
# atomic with respect to every code/store, which is what prevents concurrent
# redemptions of the same authorization code from both succeeding.
_REDEEM_LOCK = threading.RLock()


def _non_empty(value: Any) -> Optional[str]:
    """Return *value* when it is a non-empty string, else ``None``."""
    if isinstance(value, str) and value:
        return value
    return None


def _constant_time_equals(left: Any, right: Any) -> bool:
    """Compare two strings in constant time, rejecting empty/non-str input."""
    left = _non_empty(left)
    right = _non_empty(right)
    if left is None or right is None:
        return False
    try:
        left_bytes = left.encode("ascii")
        right_bytes = right.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(left_bytes, right_bytes)


def verify_challenge(code_verifier: str, code_challenge: str, method: str) -> bool:
    """Verify a PKCE ``code_verifier`` against the stored challenge.

    Only the two methods defined by RFC 7636 are accepted.  ``S256`` hashes
    the verifier with SHA-256 and base64url-encodes it without padding;
    ``plain`` compares the verifier directly.  Any other method (including
    missing values) fails closed.
    """
    verifier = _non_empty(code_verifier)
    challenge = _non_empty(code_challenge)
    if verifier is None or challenge is None or not isinstance(method, str):
        return False

    try:
        verifier_bytes = verifier.encode("ascii")
    except UnicodeEncodeError:
        return False

    if method == "S256":
        digest = hashlib.sha256(verifier_bytes).digest()
        computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return _constant_time_equals(computed, challenge)

    if method == "plain":
        return _constant_time_equals(verifier, challenge)

    return False


def redirect_uri_ok(registered_redirect_uri: str, requested_redirect_uri: str) -> bool:
    """Return ``True`` only when the requested redirect URI matches exactly.

    The comparison is byte-for-byte after rejecting empty values: scheme,
    host, port, path, query and trailing slash are all significant.  This is
    deliberately not a prefix check.
    """
    return _constant_time_equals(registered_redirect_uri, requested_redirect_uri)


def bind_state(session_state: str, returned_state: str) -> bool:
    """Check that the ``state`` returned on the callback matches the session.

    A missing value on either side fails closed, so a callback without a
    session-bound ``state`` cannot be accepted.
    """
    return _constant_time_equals(session_state, returned_state)


def nonce_ok(expected_nonce: str, received_nonce: str) -> bool:
    """Check the ``nonce`` claim in an ID Token against the stored nonce.

    Both values must be present and identical; a token without a nonce, or
    a server that never stored one, must be rejected.
    """
    return _constant_time_equals(expected_nonce, received_nonce)


def redeem_code(store: MutableMapping, code: str, client_id: str) -> bool:
    """Redeem an authorization code exactly once for the bound client.

    ``store`` maps authorization codes to records like::

        {"client": "web", "used": False, "expires_at": 1700000000}

    The client binding is checked before the code is consumed, and the
    record is marked used while holding a lock, so racing requests can
    never both receive tokens.  An optional ``expires_at`` (epoch seconds)
    is honoured when present.
    """
    code_value = _non_empty(code)
    client_value = _non_empty(client_id)
    if code_value is None or client_value is None:
        return False
    if store is None or not hasattr(store, "get"):
        return False

    with _REDEEM_LOCK:
        record = store.get(code_value)
        if not isinstance(record, Mapping):
            return False

        stored_client = record.get("client")
        if stored_client is None:
            stored_client = record.get("client_id")
        if not _constant_time_equals(stored_client, client_value):
            return False

        if record.get("used"):
            return False

        expires_at = record.get("expires_at")
        if expires_at is not None:
            try:
                if float(expires_at) <= time.time():
                    return False
            except (TypeError, ValueError):
                return False

        # Consume before returning: by the time a token is issued the code
        # is already unusable for any other request.
        if isinstance(record, MutableMapping):
            record["used"] = True
        else:
            return False
        return True
