# Extraction failure modes

Observed / expected hard fields on synthetic forms:

- **Dates (`date_of_loss`, `effective_date`)**: OCR digit confusions (0/O, 1/l) and format variation.
- **Dollar amounts (`estimated_damage`, `deductible`, `reserve_set`)**: commas, `$` glyphs, and OCR substitutions.
- **Free-text location / narrative-adjacent values**: long spans bleed into neighboring fields under BIO labeling.
- **Noisy variants**: token F1 and field exact-match drop vs clean renders; partial match remains more stable.

## Measured hard-field partial match

- `date_of_loss`: 0.000
- `estimated_damage`: 0.000
- `deductible`: 0.000
- `reserve_set`: 0.000
- `location`: 0.000

## Noisy stress summary

- token_macro_f1: 0.039
- field_exact_mean: 0.007
- field_partial_mean: 0.016
