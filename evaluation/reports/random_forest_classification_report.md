# Random Forest classification report

- Corpus: typed Stage A documents + handwriting/OCR noisy variants
- Fit rows: 408
- Test rows: 72
- Document-type accuracy: **1.0000**
- Document-type macro F1: **1.0000**
- Surface (typed vs handwriting_ocr) accuracy: **0.6528**

## Accuracy by surface

| surface | n | accuracy | macro_f1 |
|---|---:|---:|---:|
| handwriting_ocr | 36 | 1.0000 | 1.0000 |
| typed | 36 | 1.0000 | 1.0000 |

## Artifacts

- `models/random_forest_classifier/random_forest_pipeline.joblib`
- `models/random_forest_classifier/test_predictions.jsonl`
- `models/random_forest_classifier/surface_random_forest_pipeline.joblib`
