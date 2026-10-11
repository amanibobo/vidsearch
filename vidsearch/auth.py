"""Auth0 access tokens -> user ids, as FastAPI dependencies.

The browser sends `Authorization: Bearer <access token>` issued by Auth0 for our API audience.
We verify it as an RS256 JWT against the tenant's JWKS; the user id is the token's `sub`.
A token that is present but invalid is a 401 everywhere: it never degrades to anonymous.

For local end-to-end testing without a tenant, VS_DEV_AUTH=1 with an empty VS_AUTH0_DOMAIN accepts
`Bearer dev:<sub>` as user <sub>. A configured domain always selects the Auth0 verifier, so the dev
path can't be reached in a deployment that has sign-in.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Protocol

import jwt
from fastapi import HTTPException, Request

from .config import Settings


log = logging.getLogger("vidsearch.auth")


class TokenVerifier(Protocol):
    def user_id(self, token: str) -> str:
        """The token's user id. Raises jwt.PyJWTError when the token is not valid for this API."""


class Verifier:
    def __init__(self, domain: str, audience: str):
        self.issuer = f"https://{domain}/"
        self.audience = audience
        self.jwks = jwt.PyJWKClient(f"https://{domain}/.well-known/jwks.json", cache_keys=True)

    def user_id(self, token: str) -> str:
        """The token's `sub`. Raises jwt.PyJWTError when the token is not valid for this API."""
        key = self.jwks.get_signing_key_from_jwt(token)
        claims = jwt.decode(token, key.key, algorithms=["RS256"], audience=self.audience, issuer=self.issuer,
                            options={"require": ["sub", "exp", "iss", "aud"]})
        return claims["sub"]


class DevVerifier:
    """`dev:<sub>` -> <sub>. Anything else is rejected, so a real Auth0 token is a 401 here too."""

    TOKEN = re.compile(r"dev:([A-Za-z0-9_.|@-]{1,64})")

    def user_id(self, token: str) -> str:
        m = self.TOKEN.fullmatch(token)
        if not m:
            raise jwt.InvalidTokenError("dev auth expects 'dev:<user>'")
        return m.group(1)


def make_verifier(s: Settings) -> TokenVerifier | None:
    if s.auth0_domain:
        return Verifier(s.auth0_domain, s.auth0_audience)
    if s.dev_auth:
        log.warning("\n" + "!" * 72 + "\n!!! VS_DEV_AUTH=1: 'Bearer dev:<user>' signs anyone in as <user>. "
                    "Local testing only.\n" + "!" * 72)
        return DevVerifier()
    return None


def user_dependencies(s: Settings) -> tuple[Callable[[Request], str | None], Callable[[Request], str]]:
    """(optional_user, required_user) for Depends(). Empty VS_AUTH0_DOMAIN (without dev auth) turns auth off."""
    verifier = make_verifier(s)

    def optional_user(request: Request) -> str | None:
        header = request.headers.get("authorization")
        if verifier is None or not header:
            return None
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "expected 'Authorization: Bearer <token>'")
        try:
            return verifier.user_id(token)
        except jwt.PyJWTError as e:
            raise HTTPException(401, f"invalid token: {e}") from None

    def required_user(request: Request) -> str:
        if verifier is None:
            raise HTTPException(503, "uploads not configured")
        user = optional_user(request)
        if user is None:
            raise HTTPException(401, "sign in required")
        return user

    return optional_user, required_user
