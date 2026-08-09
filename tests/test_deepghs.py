from __future__ import annotations

import hashlib
from pathlib import Path

from anima_style_data.deepghs import _import_staged_files


def test_import_staged_mirror_file(tmp_path: Path) -> None:
    payload = b"danbooru snapshot image"
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "123.jpg").write_bytes(payload)
    row = {
        "id": 123,
        "file_ext": "jpg",
        "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    }

    imported = _import_staged_files(staged, {123: row}, tmp_path / "images")

    assert imported[123]["metadata_md5_match"] is True
    assert imported[123]["download_source"] == "deepghs/danbooru2024"
    assert Path(imported[123]["local_path"]).read_bytes() == payload
