# Random Forest multilayer classification report

- Fit rows: 12
- Test rows: 4
- Best preset (by val macro F1): **shallow**
- Document-type test accuracy: **1.0000**
- Document-type test macro F1: **1.0000**
- Surface accuracy: **0.7500**
- Slice typed accuracy: **1.0000**
- Slice OCR accuracy: **1.0000**
- Confidence ECE: **0.4703**
- Model: `models/random_forest_classifier/random_forest_pipeline.joblib`

## Capacity sweep (Layer 1)

| preset | val_accuracy | val_macro_f1 | n_estimators | analyzer |
|---|---:|---:|---:|---|
| shallow | 1.0000 | 1.0000 | 60 | word |
| char_robust | 1.0000 | 1.0000 | 60 | char_wb |
