# Decision log (ADRs)

An **Architecture Decision Record** captures one decision: the context, the options, what was
chosen, and the consequences. They're short and never deleted — if a decision changes, write a
new ADR that *supersedes* the old one. That history is the point.

Template:
```
# ADR-NNN — Title
Status: Proposed | Accepted | Superseded by ADR-XXX
Context:   what problem / constraint forced a decision
Options:   what was considered, with pros/cons
Decision:  what we chose
Consequences: what this makes easier / harder
```

| ADR | Decision | Status |
|---|---|---|
| [001](ADR-001-task-and-output-schema.md) | Task = ticket triage, output = versioned Pydantic schema | Accepted |
| [002](ADR-002-base-model-size.md) | Base model 1.5B–3B, family picked at Step 06 | Accepted |
| [003](ADR-003-compute-train-in-cloud-serve-locally.md) | Train on Kaggle, serve on laptop via GGUF | Accepted |
| [004](ADR-004-tooling.md) | uv, Pydantic, OpenAI-compatible clients everywhere | Accepted |
| [005](ADR-005-teacher-model.md) | Which large model labels the data | **Proposed — needs you** |
