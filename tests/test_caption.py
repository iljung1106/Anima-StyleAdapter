from anima_style_data.caption import build_anima_caption


def test_anima_caption_has_expected_sections_and_deduplicates():
    row = {
        "id": 1,
        "artist": "artist",
        "local_path": "image.png",
        "rating": "general",
        "general_tags": ["blue hair", "1girl", "solo", "blue hair", "maid"],
        "character_tags": ["ouro kronii", "ouro kronii (maid)"],
        "tagger_revision": "revision",
        "tagger_threshold": 0.6025,
    }
    cfg = {
        "rating_map": {"general": "safe"},
        "include_rating": True,
        "include_characters": True,
        "include_general": True,
    }
    result = build_anima_caption(row, cfg)
    assert result["anima_caption"] == (
        "safe, 1girl, ouro kronii, ouro kronii (maid), blue hair, solo, maid"
    )
    assert result["content_caption"] == (
        "1girl, ouro kronii, ouro kronii (maid), blue hair, solo, maid"
    )
    assert result["count_tags"] == ["1girl"]
