# Step 01 — Environment setup

**Who:** 🧑 You · **Status:** todo · **Time:** ~20 min

Run these yourself in a terminal inside `E:\ticket-triage`. If anything errors, paste it to Claude.

## 1. Git
```bash
git init
git add .
git commit -m "Scaffold project"
```
**Why:** every experiment from now on should be reproducible from a commit. When a fine-tune
gets worse, you'll want to `git diff` what changed.

## 2. Install uv (Python package manager)
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Restart the terminal, then check: `uv --version`.
**Why uv:** see [ADR-004](../DECISIONS.md#adr-004).

## 3. Create the virtual environment & install deps
```bash
uv sync
```
This reads `pyproject.toml`, creates `.venv/`, and writes `uv.lock` (exact versions — commit it).

Check it worked:
```bash
uv run python -c "from triage.schema import TicketTriage; print(TicketTriage.model_json_schema()['title'])"
```
Expected output: `TicketTriage`

## 4. Move the Hugging Face cache to E:
Your C: drive has ~17 GB free; one 3B model in a few formats is ~10 GB.
```bash
setx HF_HOME "E:\hf_cache"
```
Restart the terminal so it takes effect.

## 5. Create your `.env`
```bash
copy .env.example .env
```
Leave the API key empty for now — we fill it in Step 03.

## 6. Accounts (free)
- **Hugging Face** — https://huggingface.co (to download models & push your fine-tune)
- **Kaggle** — https://kaggle.com → verify phone number (required to unlock GPUs)

## Done when
- [ ] `git log` shows a commit
- [ ] `uv run python -c ...` prints `TicketTriage`
- [ ] `echo %HF_HOME%` (cmd) shows `E:\hf_cache`
- [ ] HF and Kaggle accounts exist, Kaggle phone-verified

## What you learn
Reproducible Python environments, lockfiles, and why ML projects keep caches off the system drive.
