from re_source import access, config


def test_anonymous_development_mode_allows_admin(monkeypatch):
    monkeypatch.setattr(config, "REQUIRE_LOGIN", False)
    assert access.is_admin({}) is True


def test_logged_in_user_needs_an_admin_claim(monkeypatch):
    monkeypatch.setattr(config, "REQUIRE_LOGIN", True)
    monkeypatch.setattr(config, "ADMIN_EMAILS", frozenset())
    monkeypatch.setattr(config, "ADMIN_GROUPS", frozenset())
    assert access.is_admin({"email": "user@example.com"}) is False


def test_admin_email_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(config, "REQUIRE_LOGIN", True)
    monkeypatch.setattr(config, "ADMIN_EMAILS", frozenset({"admin@example.com"}))
    monkeypatch.setattr(config, "ADMIN_GROUPS", frozenset())
    assert access.is_admin({"email": "Admin@Example.com"}) is True


def test_admin_group_is_accepted(monkeypatch):
    monkeypatch.setattr(config, "REQUIRE_LOGIN", True)
    monkeypatch.setattr(config, "ADMIN_EMAILS", frozenset())
    monkeypatch.setattr(config, "ADMIN_GROUPS", frozenset({"data-admins"}))
    assert access.is_admin({"groups": ["Data-Admins"]}) is True
