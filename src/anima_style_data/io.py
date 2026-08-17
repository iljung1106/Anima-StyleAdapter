from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


def read_records(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def write_records(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    if not rows:
        raise ValueError(f"Refusing to write an empty Parquet file: {path}")
    # ``Table.from_pylist`` infers its schema from the first mapping only.  A
    # number of our manifests intentionally mix control and artist rows, and
    # artist-only fields would otherwise be silently discarded when a control
    # row comes first.  Preserve first-seen field order while materializing the
    # union schema for every row.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    rows = [{field: row.get(field) for field in fields} for row in rows]
    temp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temp, compression="zstd")
    temp.replace(path)
    return len(rows)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
