# DICIE evaluation — `medical_bills`

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
| hcfa | 1.0000 | 1.0000 | 1.0000 | 4 |
| ub04 | 1.0000 | 1.0000 | 1.0000 | 4 |
| other | 1.0000 | 1.0000 | 1.0000 | 4 |

### Confusion matrix

Labels order: `['hcfa', 'ub04', 'other']`

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
| name | 1.0000 | 1.0000 | 1.0000 | 12 |
| dob | 1.0000 | 1.0000 | 1.0000 | 8 |
| claim_id | 1.0000 | 1.0000 | 1.0000 | 8 |
| patient_id | 1.0000 | 1.0000 | 1.0000 | 8 |
| address | 1.0000 | 1.0000 | 1.0000 | 8 |
