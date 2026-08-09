import hashlib
import io
import json
import tarfile

from PIL import Image

from anima_style_data.anima500k import _extract_shard, extract_anima500k_human
from anima_style_data.io import read_records


def _webp(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 48), color).save(output, format="WEBP", lossless=True)
    return output.getvalue()


def _add(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def test_human_extractor_uses_full_images_only(tmp_path):
    destination = tmp_path / "data"
    shard_dir = destination / "source" / "human"
    shard_dir.mkdir(parents=True)
    full = _webp((20, 40, 60))
    face = _webp((200, 100, 50))
    record = {
        "artist": "artist_a",
        "id": 123,
        "created_at": "2026-01-01T00:00:00Z",
        "gender": "female",
        "rating": "s",
        "md5": "original-md5",
        "source": "human",
        "split": "train",
        "artist_image_index": 0,
        "style_id": "human:artist_a",
        "danbooru_post_id": 123,
        "record_id": "human_000000123",
        "tag_string_general": "1girl solo",
        "tag_string_character": "",
        "tag_string_copyright": "",
        "tag_string_meta": "",
        "full_image": {
            "filename": "human_000000123.full.webp",
            "width": 32,
            "height": 48,
            "sha256": hashlib.sha256(full).hexdigest(),
        },
        "face_image": {"filename": "human_000000123.face.webp"},
    }
    with tarfile.open(shard_dir / "human-w0-shard-00000.tar", "w") as archive:
        _add(archive, record["full_image"]["filename"], full)
        _add(archive, record["face_image"]["filename"], face)
        _add(archive, "human_000000123.json", json.dumps(record).encode())

    config = {"anima500k": {"source": "human", "source_dir": "source", "extract_workers": 2}}
    summary = extract_anima500k_human(config, destination)
    rows = read_records(destination / "download_manifest.parquet")

    assert summary["records"] == 1
    assert summary["synthetic_included"] is False
    assert summary["face_images_extracted"] == 0
    assert len(rows) == 1
    assert rows[0]["style_id"] == "human:artist_a"
    assert rows[0]["local_path"].endswith(".full.webp")
    assert not list(destination.rglob("*.face.webp"))


def test_synthetic_source_is_rejected(tmp_path):
    config = {"anima500k": {"source": "synthetic", "source_dir": "source"}}
    try:
        extract_anima500k_human(config, tmp_path)
    except ValueError as error:
        assert "synthetic" in str(error)
    else:
        raise AssertionError("synthetic source must not be accepted")


def test_empty_human_shard_is_checkpointed(tmp_path):
    destination = tmp_path / "data"
    shard_dir = destination / "source" / "human"
    shard_dir.mkdir(parents=True)
    with tarfile.open(shard_dir / "human-w0-shard-empty.tar", "w"):
        pass

    manifest_dir = destination / "extract_manifests"
    manifest_dir.mkdir()
    shard = shard_dir / "human-w0-shard-empty.tar"
    first = _extract_shard(shard, destination, manifest_dir)
    second = _extract_shard(shard, destination, manifest_dir)

    assert first == second == (None, 0, 0)
    assert list(manifest_dir.glob("*.empty"))
