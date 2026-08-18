import torch

from anima_style_data.synthetic_teacher import (
    _dct2,
    _dct_downscale,
    _idct2,
    artist_tag,
    build_synthetic_teacher_plan,
    comfy_literal_artist_tag,
    normalize_artist_name,
    synthetic_artist_split_map,
)
from anima_style_data.io import write_records


def test_artist_tag_preserves_literal_parenthesized_name():
    assert normalize_artist_name("foo_(bar)") == "foo (bar)"
    assert artist_tag("foo_(bar)") == "@foo (bar)"
    assert comfy_literal_artist_tag("foo_(bar)") == r"@foo \(bar\)"


def test_gpu_compatible_dct_roundtrip_and_low_frequency_crop():
    value = torch.randn(2, 3, 8, 8)
    coefficients = _dct2(value)

    assert torch.allclose(_idct2(coefficients), value, atol=1e-5, rtol=1e-5)
    assert _dct_downscale(value, 0.5).shape == (2, 3, 4, 4)
    constant = _dct2(torch.ones(1, 1, 8, 8))
    assert torch.allclose(constant[..., 0, 0], torch.tensor([[8.0]]), atol=1e-5)
    assert constant[..., 1:, :].abs().max() < 1e-5
    assert constant[..., :, 1:].abs().max() < 1e-5


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
    previous = tmp_path / "synthetic_previous"
    previous.mkdir()
    write_records(
        previous / "plan.parquet",
        [
            {"kind": "artist", "artist": "artist_0"},
            {"kind": "artist", "artist": "artist_1"},
        ],
    )
    config = {
        "synthetic_teacher": {
            "output_directory": "synthetic",
            "seed": 7,
            "artist_count": 4,
            "contents_per_artist": 2,
            "female_contents": 1,
            "seeds_per_content": 1,
            "exclude_artist_plan_manifests": [
                "synthetic_previous/plan.parquet"
            ],
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
    assert not {"artist_0", "artist_1"} & {
        row["artist"] for row in artist_rows
    }
    assert all(row["style_id"] == f"human:{row['artist']}" for row in artist_rows)
    assert all("@human:" not in row["artist_tag"] for row in artist_rows)
    assert {row["artist_split"] for row in artist_rows} == {
        "train",
        "validation",
        "meta_test",
    }
    assert all("@human:" not in str(row.get("prompt", "")) for row in prompts)


def test_synthetic_split_is_recovered_from_legacy_rows_without_field(tmp_path):
    config = {
        "synthetic_teacher": {
            "seed": 7,
            "bootstrap": {
                "split_seed": 11,
                "validation_artists": 1,
                "meta_test_artists": 1,
            },
        }
    }
    rows = [
        {"kind": "artist", "artist": f"artist_{index}"}
        for index in range(4)
    ]

    recovered = synthetic_artist_split_map(config, rows)

    assert set(recovered) == {row["artist"] for row in rows}
    assert list(recovered.values()).count("train") == 2
    assert list(recovered.values()).count("validation") == 1
    assert list(recovered.values()).count("meta_test") == 1
