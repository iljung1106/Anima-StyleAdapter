from __future__ import annotations

import torch

from anima_style_data.io import write_records
from anima_style_data.native_centered_teacher import (
    NativeCenteredTeacherBank,
    _teacher_spec,
)


def test_train_population_offset_is_persisted_and_reused(tmp_path):
    source = torch.tensor([1.0, 3.0, 9.0], dtype=torch.float16).reshape(3, 1, 1)
    summary = {"train_style_ids": ["artist:0", "artist:1"]}
    artist_to_index = {"artist:0": 0, "artist:1": 1, "artist:2": 2}
    bank = NativeCenteredTeacherBank(
        tensors={"centered_teacher": source},
        summary=summary,
        artist_to_index=artist_to_index,
        root=tmp_path,
    )

    assert torch.equal(
        bank.train_population_offset(),
        torch.tensor([[2.0]], dtype=torch.float16),
    )
    assert (tmp_path / "train_population_offset.safetensors").exists()

    replacement = NativeCenteredTeacherBank(
        tensors={"centered_teacher": source + 100},
        summary=summary,
        artist_to_index=artist_to_index,
        root=tmp_path,
    )
    assert torch.equal(
        replacement.train_population_offset(),
        torch.tensor([[2.0]], dtype=torch.float16),
    )


def test_teacher_spec_can_pair_full_artist_manifest_with_probe_controls(tmp_path):
    artist_rows = [
        {
            "artist": f"artist_{index}",
            "style_id": f"human:artist_{index}",
            "split": "train" if index < 2 else "validation",
        }
        for index in range(3)
    ]
    probe_rows = [
        {
            "id": 100 + index,
            "kind": "content_control",
            "content_index": index,
            "seed_index": 0,
        }
        for index in range(2)
    ]
    write_records(tmp_path / "artists.parquet", artist_rows)
    write_records(tmp_path / "probes.parquet", probe_rows)

    artists, style_ids, splits, probe_ids, _ = _teacher_spec(
        {
            "artist_manifest": "artists.parquet",
            "probe_manifest": "probes.parquet",
            "artist_count": 3,
            "content_count": 2,
        },
        tmp_path,
        {},
    )

    assert artists == ["artist_0", "artist_1", "artist_2"]
    assert style_ids == [
        "human:artist_0",
        "human:artist_1",
        "human:artist_2",
    ]
    assert splits == ["train", "train", "validation"]
    assert probe_ids == [100, 101]
