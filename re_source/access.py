"""Authorization checks for the Streamlit application."""

from __future__ import annotations

from collections.abc import Mapping

from . import config


def _claim_values(user: Mapping, name: str) -> set[str]:
    value = user.get(name)
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = ()
    return {str(item).strip().casefold() for item in values if str(item).strip()}


def is_admin(user: Mapping) -> bool:
    """Return whether the current user may mutate application data."""
    if not config.REQUIRE_LOGIN:
        return True
    email = str(user.get("email") or "").strip().casefold()
    groups = _claim_values(user, "groups")
    return bool(
        (email and email in config.ADMIN_EMAILS)
        or groups.intersection(config.ADMIN_GROUPS)
    )
