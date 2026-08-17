from anima_style_data.synthetic_teacher import (
    artist_tag,
    build_synthetic_teacher_plan,
    comfy_literal_artist_tag,
    normalize_artist_name,
)
from anima_style_data.io import write_records


def test_artist_tag_preserves_literal_parenthesized_name():
    assert normalize_artist_name("foo_(bar)") == "foo (bar)"
    assert artist_tag("foo_(bar)") == "@foo (bar)"
    assert comfy_literal_artist_tag("foo_(bar)") == r"@foo \(bar\)"


def test_synthetic_plan_uses_raw_artist_names_and_keeps_style_ids(tmp_path):
    rows = []
    for artist_index in range(8):
        artist = f"artist_{artist_index}"
        for image_index in range(2):
            female = (artist_index + image_index) % 2 == 0
            rows.append(
                {
                    "id": artist_index * 10 + image_index,
                    "artist": artist,
                    "style_id": f"human:{artist}",
                    "split": "train",
                    "rating_anima": "safe",
                    "count_tags": ["1girl"] if female else ["solo"],
                    "character_tags": [],
                    "general_tags": ["1girl", "smile"] if female else ["landscape"],
                }
            )
    captions = tmp_path / "captions"
    captions.mkdir()
    write_records(captions / "part-00000.parquet", rows)
    config = {
        "synthetic_teacher": {
            "output_directory": "synthetic",
            "seed": 7,
            "artist_count": 4,
            "contents_per_artist": 2,
            "female_contents": 1,
            "seeds_per_content": 1,
            "bootstrap": {
                "split_seed": 11,
                "validation_artists": 1,
                "meta_test_artists": 1,
            },
        }
    }

    plan, prompts = build_synthetic_teacher_plan(config, tmp_path)
    artist_rows = [row for row in plan if row["kind"] == "artist"]

    assert len({row["artist"] for row in artist_rows}) == 4
    assert all(row["style_id"] == f"human:{row['artist']}" for row in artist_rows)
    assert all("@human:" not in row["artist_tag"] for row in artist_rows)
    assert {row["artist_split"] for row in artist_rows} == {
        "train",
        "validation",
        "meta_test",
    }
    assert all("@human:" not in str(row.get("prompt", "")) for row in prompts)
