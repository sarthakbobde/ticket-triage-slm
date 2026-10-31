# Eval: base-llama-schema

200 gold tickets. Invalid answers count as wrong.

## Headline

| metric | value |
|---|---|
| valid JSON | 100.0% |
| fits the schema | 100.0% |
| category accuracy | 71.0% |
| category macro-F1 | 68.0% |
| priority accuracy | 76.0% |
| priority off by one step | 22.5% |
| sentiment accuracy | 59.0% |
| products F1 (per ticket, averaged) | 64.2% |
| replies that promise refunds or timelines | 0.0% |
| latency p50 / p95 | 5.36s / 12.76s |

Invalid answers: 0 not JSON, 0 JSON that breaks the schema.

## Category per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| technical | 89.7% | 69.3% | 78.2% | 88 |
| product | 68.0% | 86.4% | 76.1% | 59 |
| billing | 88.9% | 100.0% | 94.1% | 8 |
| outage | 26.9% | 53.8% | 35.9% | 13 |
| customer_service | 68.2% | 46.9% | 55.6% | 32 |

Confusion (rows = gold, columns = model):

| gold \ model | technical | product | billing | outage | customer_service | invalid |
|---|---|---|---|---|---|---|
| technical | 61 | 8 | 0 | 18 | 1 | 0 |
| product | 1 | 51 | 0 | 1 | 6 | 0 |
| billing | 0 | 0 | 8 | 0 | 0 | 0 |
| outage | 6 | 0 | 0 | 7 | 0 | 0 |
| customer_service | 0 | 16 | 1 | 0 | 15 | 0 |

## Priority per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| low | 95.2% | 73.8% | 83.1% | 80 |
| medium | 66.2% | 67.2% | 66.7% | 67 |
| high | 68.6% | 90.6% | 78.0% | 53 |

## Sentiment per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| negative | 100.0% | 89.3% | 94.3% | 121 |
| neutral | 52.9% | 11.7% | 19.1% | 77 |
| positive | 1.3% | 50.0% | 2.6% | 2 |

Small classes are noisy: with 8 gold billing tickets, one mistake moves billing recall by 12 points.
