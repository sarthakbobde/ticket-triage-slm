# Eval: ft-qwen-kaggle

200 gold tickets. Invalid answers count as wrong.

## Headline

| metric | value |
|---|---|
| valid JSON | 100.0% |
| fits the schema | 100.0% |
| category accuracy | 79.5% |
| category macro-F1 | 65.7% |
| priority accuracy | 70.0% |
| priority off by one step | 29.0% |
| sentiment accuracy | 72.0% |
| products F1 (per ticket, averaged) | 88.0% |
| replies that promise refunds or timelines | 0.0% |
| latency p50 / p95 | 2.04s / 2.43s |

Invalid answers: 0 not JSON, 0 JSON that breaks the schema.

## Category per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| technical | 87.8% | 97.7% | 92.5% | 88 |
| product | 69.5% | 96.6% | 80.9% | 59 |
| billing | 66.7% | 75.0% | 70.6% | 8 |
| outage | 88.9% | 61.5% | 72.7% | 13 |
| customer_service | 100.0% | 6.2% | 11.8% | 32 |

Confusion (rows = gold, columns = model):

| gold \ model | technical | product | billing | outage | customer_service | invalid |
|---|---|---|---|---|---|---|
| technical | 86 | 1 | 0 | 1 | 0 | 0 |
| product | 2 | 57 | 0 | 0 | 0 | 0 |
| billing | 2 | 0 | 6 | 0 | 0 | 0 |
| outage | 5 | 0 | 0 | 8 | 0 | 0 |
| customer_service | 3 | 24 | 3 | 0 | 2 | 0 |

## Priority per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| low | 97.1% | 82.5% | 89.2% | 80 |
| medium | 62.9% | 32.8% | 43.1% | 67 |
| high | 53.6% | 98.1% | 69.3% | 53 |

## Sentiment per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| negative | 99.0% | 82.6% | 90.1% | 121 |
| neutral | 66.2% | 55.8% | 60.6% | 77 |
| positive | 2.9% | 50.0% | 5.6% | 2 |

Small classes are noisy: with 8 gold billing tickets, one mistake moves billing recall by 12 points.
