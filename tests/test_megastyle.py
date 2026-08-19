from __future__ import annotations

import io
import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from anima_style_data.io import read_records
from anima_style_data.megastyle import prepare_megastyle_subset


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (120, 80, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_prepare_megastyle_subset_keeps_whole_styles_and_content_overlap(tmp_path: Path):
    destination = tmp_path / "main"
    source = tmp_path / "mega" / "source"
    destination.mkdir()
    source.mkdir(parents=True)
    image = _jpeg()
    rows = []
    style_indices = {}
    for style in range(6):
        indices = []
        for content in range(8):
            indices.append(len(rows))
            rows.append({
                "id": f"s{style}_c{content}",
                "image": {"bytes": image, "path": f"s{style}_c{content}.jpg"},
                "content": f"girl pose content {content}",
                "style": f"private style description {style}",
            })
        style_indices[f"private style description {style}"] = indices
    pq.write_table(
        pa.Table.from_pylist(rows), source / "train-00000.parquet", row_group_size=5
    )
    with (source / "style_indices.pkl").open("wb") as handle:
        pickle.dump(style_indices, handle)
    config = {
        "megastyle": {
            "source_directory": "../mega/source",
            "subset": {
                "output_directory": "../mega",
                "style_count": 4,
                "images_per_style": 8,
                "validation_styles": 1,
                "seed": 7,
                "seed_styles": 1,
                "image_workers": 2,
                "pending_writes": 4,
            },
        }
    }

    summary = prepare_megastyle_subset(config, destination)
    manifest = read_records(tmp_path / "mega" / "final_manifest.parquet")
    captions = read_records(tmp_path / "mega" / "captions" / "part-00000.parquet")

    assert summary["styles"] == 4
    assert summary["images"] == 32
    assert summary["validation_content_overlap_fraction"] == 1.0
    assert {row["split"] for row in manifest} == {"train", "validation"}
    assert sorted({row["style_id"] for row in manifest}) == sorted(
        {row["style_id"] for row in captions}
    )
    assert all(Path(row["local_path"]).is_file() for row in manifest)
    assert all("private style description" not in row["anima_caption"] for row in captions)
    assert prepare_megastyle_subset(config, destination)["selection_signature"] == summary[
        "selection_signature"
    ]
