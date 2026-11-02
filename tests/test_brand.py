"""The design system builds without a Streamlit runtime, and without the fonts."""

import re

from re_source import brand


def test_every_color_becomes_a_token():
    css = brand.brand_css()
    for name, value in brand.COLORS.items():
        assert f"--{name.replace('_', '-')}: {value};" in css


def test_tokens_are_hyphenated_not_underscored():
    assert "--warm-black:" in brand.brand_css()
    assert "--warm_black:" not in brand.brand_css()


def test_no_raw_hex_outside_the_token_block():
    css = brand.brand_css()
    body = css.split(":root {", 1)[1].split("}", 1)[1]
    assert not re.findall(r"#[0-9A-Fa-f]{3,8}\b", body)


def test_stylesheet_is_a_single_style_block():
    css = brand.brand_css()
    assert css.count("<style>") == 1
    assert css.count("</style>") == 1


def test_every_typography_role_uses_the_system_font_stack():
    css = brand.brand_css()
    assert set(brand.STACKS.values()) == {brand.SYSTEM_FONT_STACK}
    assert "system-ui" in brand.SYSTEM_FONT_STACK
    assert '"Segoe UI"' in brand.SYSTEM_FONT_STACK
    assert brand.SYSTEM_FONT_STACK in css
    assert "@font-face" not in css


def _css_section(css: str, heading: str, next_heading: str) -> str:
    return css.split(heading, 1)[1].split(next_heading, 1)[0]


def test_buttons_keep_streamlit_geometry_and_only_override_brand_colors():
    section = _css_section(
        brand.brand_css(),
        "/* ===== BUTTONS",
        "/* ===== INPUTS",
    )
    for declaration in (
        "font-weight:",
        "letter-spacing:",
        "border-radius:",
        "padding:",
        "transition:",
        "box-shadow:",
        "outline:",
        "outline-offset:",
    ):
        assert declaration not in section
    assert "var(--blue-03)" in section
    assert "var(--blue-04)" in section


def test_checkboxes_keep_streamlit_geometry_and_only_override_brand_colors():
    section = _css_section(
        brand.brand_css(),
        "/* ===== CHECKBOXES",
        "/* ===== SIDEBAR",
    )
    for declaration in (
        "width:",
        "height:",
        "min-width:",
        "border-radius:",
        "border:",
    ):
        assert declaration not in section
    assert "var(--blue-03)" in section


def test_the_ui_carries_no_palette_of_its_own():
    from pathlib import Path

    # Resolved against the package, not the working directory: read from the
    # repo root this passed by accident, and from anywhere else it read a
    # file that did not exist and never saw a hex code at all.
    package = Path(brand.__file__).parent
    for name in ("search_ui.py", "admin_ui.py"):
        source = (package / name).read_text()
        assert not re.findall(r":\s*#[0-9A-Fa-f]{3,8}\b", source), name
