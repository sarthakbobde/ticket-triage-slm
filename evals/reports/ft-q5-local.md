# Eval: ft-q5-local

200 gold tickets. Invalid answers count as wrong.

## Headline

| metric | value |
|---|---|
| valid JSON | 100.0% |
| fits the schema | 100.0% |
| category accuracy | 79.5% |
| category macro-F1 | 66.4% |
| priority accuracy | 70.0% |
| priority off by one step | 30.0% |
| sentiment accuracy | 76.0% |
| products F1 (per ticket, averaged) | 85.4% |
| replies that promise refunds or timelines | 0.0% |
| latency p50 / p95 | 2.72s / 3.48s |

Invalid answers: 0 not JSON, 0 JSON that breaks the schema.

## Category per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| technical | 87.8% | 97.7% | 92.5% | 88 |
| product | 68.7% | 96.6% | 80.3% | 59 |
| billing | 75.0% | 75.0% | 75.0% | 8 |
| outage | 88.9% | 61.5% | 72.7% | 13 |
| customer_service | 100.0% | 6.2% | 11.8% | 32 |

Confusion (rows = gold, columns = model):

| gold \ model | technical | product | billing | outage | customer_service | invalid |
|---|---|---|---|---|---|---|
| technical | 86 | 1 | 0 | 1 | 0 | 0 |
| product | 2 | 57 | 0 | 0 | 0 | 0 |
| billing | 2 | 0 | 6 | 0 | 0 | 0 |
| outage | 5 | 0 | 0 | 8 | 0 | 0 |
| customer_service | 3 | 25 | 2 | 0 | 2 | 0 |

## Priority per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| low | 100.0% | 73.8% | 84.9% | 80 |
| medium | 55.9% | 49.3% | 52.4% | 67 |
| high | 58.5% | 90.6% | 71.1% | 53 |

## Sentiment per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| negative | 99.0% | 86.0% | 92.0% | 121 |
| neutral | 72.3% | 61.0% | 66.2% | 77 |
| positive | 3.3% | 50.0% | 6.2% | 2 |

Small classes are noisy: with 8 gold billing tickets, one mistake moves billing recall by 12 points.
