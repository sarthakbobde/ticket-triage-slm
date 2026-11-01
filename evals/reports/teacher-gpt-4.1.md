# Eval: teacher-gpt-4.1

200 gold tickets. Invalid answers count as wrong.

## Headline

| metric | value |
|---|---|
| valid JSON | 100.0% |
| fits the schema | 99.5% |
| category accuracy | 79.5% |
| category macro-F1 | 66.4% |
| priority accuracy | 68.5% |
| priority off by one step | 30.5% |
| sentiment accuracy | 70.0% |
| products F1 (per ticket, averaged) | 85.0% |
| replies that promise refunds or timelines | 0.0% |
| latency p50 / p95 | 1.26s / 1.65s |

Invalid answers: 0 not JSON, 1 JSON that breaks the schema.

## Category per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| technical | 87.6% | 96.6% | 91.9% | 88 |
| product | 70.4% | 96.6% | 81.4% | 59 |
| billing | 71.4% | 62.5% | 66.7% | 8 |
| outage | 80.0% | 61.5% | 69.6% | 13 |
| customer_service | 100.0% | 12.5% | 22.2% | 32 |

Confusion (rows = gold, columns = model):

| gold \ model | technical | product | billing | outage | customer_service | invalid |
|---|---|---|---|---|---|---|
| technical | 85 | 1 | 0 | 1 | 0 | 1 |
| product | 2 | 57 | 0 | 0 | 0 | 0 |
| billing | 2 | 0 | 5 | 1 | 0 | 0 |
| outage | 5 | 0 | 0 | 8 | 0 | 0 |
| customer_service | 3 | 23 | 2 | 0 | 4 | 0 |

## Priority per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| low | 96.8% | 76.2% | 85.3% | 80 |
| medium | 56.1% | 34.3% | 42.6% | 67 |
| high | 55.8% | 100.0% | 71.6% | 53 |

## Sentiment per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| negative | 98.0% | 82.6% | 89.7% | 121 |
| neutral | 65.0% | 50.6% | 56.9% | 77 |
| positive | 2.7% | 50.0% | 5.1% | 2 |

Small classes are noisy: with 8 gold billing tickets, one mistake moves billing recall by 12 points.
