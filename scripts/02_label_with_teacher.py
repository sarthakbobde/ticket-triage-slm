"""Step 03 — label tickets with the teacher model -> data/labeled/teacher.jsonl

TODO (Step 03):
- load config from configs/models.yaml (teacher section)
- for each ticket: build_messages() -> call teacher -> validate with TicketTriage
- on validation failure: retry once, then log to data/labeled/failures.jsonl
- resumable: skip ids already in the output file (API calls cost money)
- record SCHEMA_VERSION and teacher model name in every row
"""
