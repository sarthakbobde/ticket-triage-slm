"""Small helpers for .jsonl files (one JSON object per line)."""

import json
from pathlib import Path
from typing import Any, TextIO

Row = dict[str, Any]


def read_jsonl(path: Path) -> list[Row]:
    """Read a .jsonl file into a list of dictionaries."""
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[Row]) -> None:
    """Write rows to a .jsonl file, replacing it. Creates the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_row(file: TextIO, row: Row) -> None:
    """Add one row to an open file and flush, so a crash loses at most one row."""
    file.write(json.dumps(row, ensure_ascii=False) + "\n")
    file.flush()


def read_done_ids(path: Path, ignore_failed: bool = False) -> set[str]:
    """Ids already in a .jsonl file, so a re-run can skip them.

    ignore_failed=True leaves out rows that recorded an "error", so failed calls get retried.
    """
    done: set[str] = set()
    if not path.exists():
        return done
    for row in read_jsonl(path):
        if ignore_failed and row.get("error"):
            continue
        done.add(row["id"])
    return done
