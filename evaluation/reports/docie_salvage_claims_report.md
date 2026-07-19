# DICIE evaluation — `salvage_claims`

Paper-aligned metrics from Raj, Dickinson & Fung (*Document Classification and Information Extraction framework for Insurance Applications*).

- N: **12**
- Eval set: `/Users/morningstar/Desktop/Cold_Storage/smol-doc-analyzer/data/eval/docie_eval_set.jsonl`

## Table I — Document Classification

| Metric | Score |
|--------|------:|
| Accuracy | 1.0000 |
| AUC (OVR) | 1.0000 |
| AUC (OVO) | 1.0000 |
| Macro F1 (diagnostic) | 1.0000 |
| Micro F1 (diagnostic) | 1.0000 |
| Weighted F1 (diagnostic) | 1.0000 |

### Per-class F1

| Label | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| log | 1.0000 | 1.0000 | 1.0000 | 4 |
| sales | 1.0000 | 1.0000 | 1.0000 | 4 |
| other | 1.0000 | 1.0000 | 1.0000 | 4 |

### Confusion matrix

Labels order: `['log', 'sales', 'other']`

```
[
  [
    4,
    0,
    0
  ],
  [
    0,
    4,
    0
  ],
  [
    0,
    0,
    4
  ]
]
```

## Table II — Information Extraction

| Aggregate | Score |
|-----------|------:|
| Micro Precision | 1.0000 |
| Micro Recall | 1.0000 |
| Micro F1 | 1.0000 |
| Macro F1 (field mean) | 1.0000 |

### Per-field Precision / Recall / F1

| Field | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| claim_id | 1.0000 | 1.0000 | 1.0000 | 4 |
| vin | 1.0000 | 1.0000 | 1.0000 | 8 |
| make | 1.0000 | 1.0000 | 1.0000 | 8 |
| model | 1.0000 | 1.0000 | 1.0000 | 8 |
| year | 1.0000 | 1.0000 | 1.0000 | 8 |
