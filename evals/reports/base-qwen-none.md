# Eval: base-qwen-none

200 gold tickets. Invalid answers count as wrong.

## Headline

| metric | value |
|---|---|
| valid JSON | 100.0% |
| fits the schema | 100.0% |
| category accuracy | 75.5% |
| category macro-F1 | 62.5% |
| priority accuracy | 40.5% |
| priority off by one step | 57.0% |
| sentiment accuracy | 63.0% |
| products F1 (per ticket, averaged) | 36.3% |
| replies that promise refunds or timelines | 0.5% |
| latency p50 / p95 | 2.26s / 2.96s |

Invalid answers: 0 not JSON, 0 JSON that breaks the schema.

## Category per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| technical | 79.1% | 98.9% | 87.9% | 88 |
| product | 67.1% | 86.4% | 75.6% | 59 |
| billing | 85.7% | 75.0% | 80.0% | 8 |
| outage | 100.0% | 46.2% | 63.2% | 13 |
| customer_service | 100.0% | 3.1% | 6.1% | 32 |

Confusion (rows = gold, columns = model):

| gold \ model | technical | product | billing | outage | customer_service | invalid |
|---|---|---|---|---|---|---|
| technical | 87 | 1 | 0 | 0 | 0 | 0 |
| product | 8 | 51 | 0 | 0 | 0 | 0 |
| billing | 2 | 0 | 6 | 0 | 0 | 0 |
| outage | 7 | 0 | 0 | 6 | 0 | 0 |
| customer_service | 6 | 24 | 1 | 0 | 1 | 0 |

## Priority per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| low | 0.0% | 0.0% | 0.0% | 80 |
| medium | 28.4% | 46.3% | 35.2% | 67 |
| high | 54.9% | 94.3% | 69.4% | 53 |

## Sentiment per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| negative | 96.3% | 86.8% | 91.3% | 121 |
| neutral | 57.1% | 26.0% | 35.7% | 77 |
| positive | 1.8% | 50.0% | 3.4% | 2 |

Small classes are noisy: with 8 gold billing tickets, one mistake moves billing recall by 12 points.
