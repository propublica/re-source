"""Okta bearer-token authentication for the search API."""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass

import requests
from fastapi import Depends, HTTPException, Request, status
from joserfc import jwt
from joserfc.errors import JoseError
from joserfc.jwk import KeySet

logger = logging.getLogger(__name__)

OKTA_ISSUER = os.getenv("OKTA_ISSUER", "").rstrip("/")
OKTA_AUDIENCE = os.getenv("OKTA_AUDIENCE", "api://default")
API_REQUIRED_SCOPE = os.getenv("API_REQUIRED_SCOPE", "re-source:search").strip()

# Fail closed: only an exact "false" disables auth, so REQUIRE_API_AUTH=0 or a
# stray typo leaves the API protected rather than wide open. (The Streamlit
# REQUIRE_LOGIN flag is looser; this is the internet-facing surface.)
REQUIRE_API_AUTH = os.getenv("REQUIRE_API_AUTH", "true").strip().lower() != "false"

# Okta rotates signing keys, so the JWKS cache has to expire. An unknown `kid`
# also forces an immediate refetch (see _signing_keys), which is what actually
# covers rotation; this bound just stops a stale set living forever.
_JWKS_TTL_SECONDS = int(os.getenv("OKTA_JWKS_TTL", "3600"))

# Only asymmetric algorithms. Never include "none" or an HMAC family here — with
# a public JWKS, accepting HS256 would let a caller sign their own token using
# the public key as the shared secret.
_ALLOWED_ALGORITHMS = ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]

_keyset: KeySet | None = None
_keyset_fetched_at: float = 0.0
_keyset_lock = threading.Lock()


@dataclass(frozen=True)
class Principal:
    """The authenticated caller, as read from the access token."""

    subject: str
    email: str | None = None
    scopes: tuple[str, ...] = ()

    @property
    def user_id(self) -> str:
        """Identity for trace attribution — email where Okta provides it, so
        API traces group with the UI's (which key on st.user email)."""
        return self.email or self.subject


def _jwks_url() -> str:
    if not OKTA_ISSUER:
        raise RuntimeError(
            "REQUIRE_API_AUTH is on but OKTA_ISSUER is not set. Set it to your "
            "Okta issuer (e.g. https://your-org.okta.com/oauth2/default), or "
            "set REQUIRE_API_AUTH=false for local development."
        )
    return f"{OKTA_ISSUER}/v1/keys"


def _fetch_keyset() -> KeySet:
    response = requests.get(_jwks_url(), timeout=10)
    response.raise_for_status()
    return KeySet.import_key_set(response.json())


def _signing_keys(force_refresh: bool = False) -> KeySet:
    """The cached JWKS, refetched when stale or when `force_refresh` is set."""
    global _keyset, _keyset_fetched_at
    with _keyset_lock:
        stale = time.monotonic() - _keyset_fetched_at > _JWKS_TTL_SECONDS
        if _keyset is None or stale or force_refresh:
            _keyset = _fetch_keyset()
            _keyset_fetched_at = time.monotonic()
        return _keyset


def _token_kid(token: str) -> str | None:
    """The `kid` from a token's header, or None if there isn't a readable one."""
    try:
        segment = token.split(".", 1)[0]
        padded = segment + "=" * (-len(segment) % 4)
        header = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except Exception:
        return None
    kid = header.get("kid") if isinstance(header, dict) else None
    return str(kid) if kid else None


def _has_unknown_kid(token: str) -> bool:
    """Whether this token names a signing key the cached JWKS doesn't hold."""
    kid = _token_kid(token)
    if not kid:
        return False
    try:
        keys = _signing_keys()
    except requests.RequestException:
        # The caller's own handler turns this into a 503; swallowing it here
        # would downgrade an outage to "invalid token".
        raise
    except Exception:
        return False
    try:
        keys.get_by_kid(kid)
    except Exception:
        return True
    return False


def _claims_registry() -> jwt.JWTClaimsRegistry:
    """Require and pin the claims that make the token ours: `iss` must be our
    Okta org, `aud` our API, and `exp` must be present (a token with no `exp`
    would otherwise validate forever)."""
    return jwt.JWTClaimsRegistry(
        iss={"essential": True, "value": OKTA_ISSUER},
        aud={"essential": True, "value": OKTA_AUDIENCE},
        exp={"essential": True},
    )


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. "
            "Expected: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token.strip()


def _decode(token: str, *, force_refresh: bool = False) -> dict:
    decoded = jwt.decode(
        token, _signing_keys(force_refresh=force_refresh),
        algorithms=_ALLOWED_ALGORITHMS,
    )
    _claims_registry().validate(decoded.claims)
    return decoded.claims


def _unauthorized(exc: Exception | None = None) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _unavailable(exc: Exception) -> HTTPException:
    # Okta unreachable — fail closed, but say so distinctly: this is our
    # outage, not the caller's bad token.
    logger.error(f"could not reach Okta JWKS: {exc}")
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Cannot verify credentials right now.",
    )


def verify_token(token: str) -> Principal:
    """Validate an Okta access token and return its Principal."""
    try:
        claims = _decode(token)
    except requests.RequestException as exc:
        raise _unavailable(exc) from exc
    except JoseError as first_error:
        # A rotated signing key is the one failure a refetch can fix, and it
        # has a signature: the token names a `kid` the cached set doesn't hold.
        # Retrying on every JoseError meant any malformed or self-signed token
        # forced a blocking JWKS fetch under the module lock, so an
        # unauthenticated caller could stall every in-flight request and hammer
        # Okta from this server's IP just by looping bad tokens.
        if not _has_unknown_kid(token):
            logger.warning(f"rejected access token: {first_error}")
            raise _unauthorized() from first_error
        try:
            claims = _decode(token, force_refresh=True)
        except requests.RequestException as exc:
            # Raised from inside this handler, so the sibling clause above
            # cannot catch it — Python does not route an exception raised in an
            # except block to the other clauses of the same try.
            raise _unavailable(exc) from exc
        except JoseError:
            logger.warning(f"rejected access token: {first_error}")
            raise _unauthorized() from first_error

    subject = claims.get("sub")
    if not subject:
        logger.warning("rejected access token: no `sub` claim")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_scopes = claims.get("scp") or claims.get("scope") or ()
    if isinstance(raw_scopes, str):
        raw_scopes = raw_scopes.split()

    return Principal(
        subject=str(subject),
        email=claims.get("email") or claims.get("preferred_username"),
        scopes=tuple(str(s) for s in raw_scopes),
    )


def require_user(request: Request) -> Principal:
    """FastAPI dependency: the authenticated caller."""
    if not REQUIRE_API_AUTH:
        return Principal(subject="anonymous", email="anonymous")
    return verify_token(_bearer_token(request))


def require_search_user(request: Request) -> Principal:
    """Require an authenticated principal with search permission."""
    principal = require_user(request)
    if REQUIRE_API_AUTH and API_REQUIRED_SCOPE not in principal.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient scope.",
        )
    return principal


CurrentUser = Depends(require_user)
