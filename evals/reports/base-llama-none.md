# Eval: base-llama-none

200 gold tickets. Invalid answers count as wrong.

## Headline

| metric | value |
|---|---|
| valid JSON | 2.0% |
| fits the schema | 2.0% |
| category accuracy | 2.0% |
| category macro-F1 | 2.0% |
| priority accuracy | 1.5% |
| priority off by one step | 0.5% |
| sentiment accuracy | 1.0% |
| products F1 (per ticket, averaged) | 2.0% |
| replies that promise refunds or timelines | 0.0% |
| latency p50 / p95 | 2.96s / 3.88s |

Invalid answers: 196 not JSON, 0 JSON that breaks the schema.

## Category per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| technical | 100.0% | 3.4% | 6.6% | 88 |
| product | 100.0% | 1.7% | 3.3% | 59 |
| billing | 0.0% | 0.0% | 0.0% | 8 |
| outage | 0.0% | 0.0% | 0.0% | 13 |
| customer_service | 0.0% | 0.0% | 0.0% | 32 |

Confusion (rows = gold, columns = model):

| gold \ model | technical | product | billing | outage | customer_service | invalid |
|---|---|---|---|---|---|---|
| technical | 3 | 0 | 0 | 0 | 0 | 85 |
| product | 0 | 1 | 0 | 0 | 0 | 58 |
| billing | 0 | 0 | 0 | 0 | 0 | 8 |
| outage | 0 | 0 | 0 | 0 | 0 | 13 |
| customer_service | 0 | 0 | 0 | 0 | 0 | 32 |

## Priority per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| low | 100.0% | 1.2% | 2.5% | 80 |
| medium | 100.0% | 1.5% | 2.9% | 67 |
| high | 50.0% | 1.9% | 3.6% | 53 |

## Sentiment per class

| class | precision | recall | F1 | gold tickets |
|---|---|---|---|---|
| negative | 100.0% | 1.7% | 3.3% | 121 |
| neutral | 0.0% | 0.0% | 0.0% | 77 |
| positive | 0.0% | 0.0% | 0.0% | 2 |

Small classes are noisy: with 8 gold billing tickets, one mistake moves billing recall by 12 points.
