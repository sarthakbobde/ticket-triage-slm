"""Step 04 — clean, dedupe and split -> data/processed/{train,val,test}.jsonl

TODO (Step 04):
- drop rows that fail schema validation
- near-duplicate removal (so test tickets don't leak into train)
- stratified split by category, e.g. 80/10/10, fixed random seed
- write the test set ONCE and never regenerate it
"""
