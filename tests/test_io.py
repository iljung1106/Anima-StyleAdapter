from anima_style_data.io import read_records, write_records


def test_write_records_preserves_fields_introduced_after_first_row(tmp_path):
    path = tmp_path / "mixed.parquet"

    write_records(
        path,
        [
            {"kind": "control", "artist": None},
            {"kind": "artist", "artist": "example", "artist_split": "train"},
        ],
    )

    assert read_records(path) == [
        {"kind": "control", "artist": None, "artist_split": None},
        {"kind": "artist", "artist": "example", "artist_split": "train"},
    ]
