"""Small helpers for .jsonl files.

A .jsonl file holds one JSON object per line. We use it for tickets, labels and model answers.
"""

import json


def read_jsonl(path):
    """Read a .jsonl file and return a list of dictionaries, one per line."""
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            # Skip empty lines.
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    """Write a list of dictionaries to a .jsonl file, replacing the file if it exists."""
    # Make sure the folder exists first.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_row(file, row):
    """Add one row to a file that is already open, and save it to disk straight away.

    Saving after every row means a crash loses at most one row.
    """
    file.write(json.dumps(row, ensure_ascii=False) + "\n")
    file.flush()


def read_done_ids(path, ignore_failed=False):
    """Return the set of ticket ids already in a .jsonl file, so a re-run can skip them.

    With ignore_failed=True, rows that recorded an "error" are left out, so failed calls get retried.
    """
    done = set()
    if not path.exists():
        return done
    for row in read_jsonl(path):
        if ignore_failed and row.get("error"):
            continue
        done.add(row["id"])
    return done
