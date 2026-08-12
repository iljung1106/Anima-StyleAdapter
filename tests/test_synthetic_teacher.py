from anima_style_data.synthetic_teacher import (
    artist_tag,
    comfy_literal_artist_tag,
    normalize_artist_name,
)


def test_artist_tag_preserves_literal_parenthesized_name():
    assert normalize_artist_name("foo_(bar)") == "foo (bar)"
    assert artist_tag("foo_(bar)") == "@foo (bar)"
    assert comfy_literal_artist_tag("foo_(bar)") == r"@foo \(bar\)"
