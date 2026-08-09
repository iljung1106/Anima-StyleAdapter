from anima_style_data.metadata import _select_artist_candidates, _valid_row


def _row(image_id: int, day: int) -> dict:
    return {
        "id": image_id,
        "created_at": f"2026-01-{day:02d}T00:00:00+00:00",
        "created_ordinal": 739616 + day,
        "md5": f"{image_id:032x}",
        "file_ext": "jpg",
        "file_size": 100,
        "image_width": 1024,
        "image_height": 1024,
        "source": None,
        "pixiv_id": None,
        "parent_id": None,
        "tag_string_meta": "",
        "file_url": f"https://example.invalid/{image_id}.jpg",
        "large_file_url": None,
        "original_url": None,
    }


def test_candidate_selection_is_deterministic_and_respects_window():
    cfg = {
        "seed": 7,
        "images_per_artist": 2,
        "candidate_multiplier": 2,
        "date_windows_days": [4, 20],
        "image_recency_tau_days": 30,
    }
    rows = [_row(image_id, day) for image_id, day in enumerate([1, 2, 20, 21, 22, 23], 1)]
    first, window = _select_artist_candidates("artist", rows, cfg)
    second, second_window = _select_artist_candidates("artist", list(reversed(rows)), cfg)
    assert window == second_window == 4
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 4


def test_exact_md5_is_removed_before_candidate_count():
    cfg = {
        "seed": 7,
        "images_per_artist": 2,
        "candidate_multiplier": 2,
        "date_windows_days": [100],
        "image_recency_tau_days": 30,
    }
    rows = [_row(index, index) for index in range(1, 5)]
    rows[-1]["md5"] = rows[-2]["md5"]
    assert _select_artist_candidates("artist", rows, cfg) is None


def test_mirror_selection_does_not_require_source_url():
    row = _row(1, 1)
    row.update(
        {
            "file_url": None,
            "tag_string_artist": "artist",
            "tag_count_artist": 1,
            "is_pending": False,
            "is_flagged": False,
            "is_deleted": False,
            "is_banned": False,
        }
    )
    cfg = {
        "require_download_url": False,
        "require_single_artist": True,
        "allowed_extensions": ["jpg"],
    }

    assert _valid_row(row, cfg)
