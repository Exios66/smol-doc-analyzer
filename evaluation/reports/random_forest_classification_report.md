# Random Forest multilayer classification report

- Fit rows: 8500
- Test rows: 1500
- Best preset (by val macro F1): **shallow**
- Document-type test accuracy: **1.0000**
- Document-type test macro F1: **1.0000**
- Surface accuracy: **0.9927**
- Slice typed accuracy: **1.0000**
- Slice OCR accuracy: **1.0000**
- Confidence ECE: **0.0834**
- Model: `models/random_forest_classifier/random_forest_pipeline.joblib`

## Capacity sweep (Layer 1)

| preset | val_accuracy | val_macro_f1 | n_estimators | analyzer |
|---|---:|---:|---:|---|
| shallow | 1.0000 | 1.0000 | 100 | word |
| balanced | 1.0000 | 1.0000 | 300 | word |
| char_robust | 1.0000 | 1.0000 | 300 | char_wb |
| hybrid_ocr | 1.0000 | 1.0000 | 400 | hybrid |
| deep | 1.0000 | 1.0000 | 500 | word |
