# Eval: ft-q4-local

200 gold tickets. Invalid answers count as wrong.

## Headline

| metric | value |
|---|---|
| valid JSON | 100.0% |
| fits the schema | 100.0% |
| category accuracy | 77.5% |
| category macro-F1 | 60.9% |
| priority accuracy | 67.0% |
| priority off by one step | 32.5% |
| sentiment accuracy | 80.0% |
| products F1 (per ticket, averaged) | 84.5% |
| replies that promise refunds or timelines | 0.0% |
| latency p50 / p95 | 2.30s / 2.95s |

Invalid answers: 0 not JSON, 0 JSON that breaks the schema.

## Category per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| technical | 87.6% | 96.6% | 91.9% | 88 |
| product | 67.1% | 96.6% | 79.2% | 59 |
| billing | 60.0% | 75.0% | 66.7% | 8 |
| outage | 87.5% | 53.8% | 66.7% | 13 |
| customer_service | 0.0% | 0.0% | 0.0% | 32 |

Confusion (rows = gold, columns = model):

| gold \ model | technical | product | billing | outage | customer_service | invalid |
|---|---|---|---|---|---|---|
| technical | 85 | 2 | 0 | 1 | 0 | 0 |
| product | 2 | 57 | 0 | 0 | 0 | 0 |
| billing | 2 | 0 | 6 | 0 | 0 | 0 |
| outage | 6 | 0 | 0 | 7 | 0 | 0 |
| customer_service | 2 | 26 | 4 | 0 | 0 | 0 |

## Priority per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| low | 100.0% | 65.0% | 78.8% | 80 |
| medium | 51.7% | 46.3% | 48.8% | 67 |
| high | 58.0% | 96.2% | 72.3% | 53 |

## Sentiment per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| negative | 100.0% | 89.3% | 94.3% | 121 |
| neutral | 78.5% | 66.2% | 71.8% | 77 |
| positive | 3.7% | 50.0% | 6.9% | 2 |

Small classes are noisy: with 8 gold billing tickets, one mistake moves billing recall by 12 points.
