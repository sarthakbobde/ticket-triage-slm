# Eval: ft-q8-local

200 gold tickets. Invalid answers count as wrong.

## Headline

| metric | value |
|---|---|
| valid JSON | 100.0% |
| fits the schema | 100.0% |
| category accuracy | 78.5% |
| category macro-F1 | 62.4% |
| priority accuracy | 72.0% |
| priority off by one step | 28.0% |
| sentiment accuracy | 79.5% |
| products F1 (per ticket, averaged) | 84.1% |
| replies that promise refunds or timelines | 0.0% |
| latency p50 / p95 | 6.26s / 7.74s |

Invalid answers: 0 not JSON, 0 JSON that breaks the schema.

## Category per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| technical | 88.7% | 97.7% | 93.0% | 88 |
| product | 67.9% | 96.6% | 79.7% | 59 |
| billing | 60.0% | 75.0% | 66.7% | 8 |
| outage | 88.9% | 61.5% | 72.7% | 13 |
| customer_service | 0.0% | 0.0% | 0.0% | 32 |

Confusion (rows = gold, columns = model):

| gold \ model | technical | product | billing | outage | customer_service | invalid |
|---|---|---|---|---|---|---|
| technical | 86 | 1 | 0 | 1 | 0 | 0 |
| product | 2 | 57 | 0 | 0 | 0 | 0 |
| billing | 2 | 0 | 6 | 0 | 0 | 0 |
| outage | 5 | 0 | 0 | 8 | 0 | 0 |
| customer_service | 2 | 26 | 4 | 0 | 0 | 0 |

## Priority per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| low | 97.0% | 81.2% | 88.4% | 80 |
| medium | 61.2% | 44.8% | 51.7% | 67 |
| high | 58.3% | 92.5% | 71.5% | 53 |

## Sentiment per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| negative | 99.1% | 90.9% | 94.8% | 121 |
| neutral | 80.0% | 62.3% | 70.1% | 77 |
| positive | 3.4% | 50.0% | 6.5% | 2 |

Small classes are noisy: with 8 gold billing tickets, one mistake moves billing recall by 12 points.
