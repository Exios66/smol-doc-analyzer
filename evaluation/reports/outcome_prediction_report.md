# Claim outcome prediction report

- N scored: **40**
- Skipped (no gold): 0
- Accuracy: **0.4**
- Macro F1: **0.3111111111111111**
- Chain: `to_markdown → classify → extract → vision_llm → predict_outcome → summarize`

## Label distribution (gold)

- `close_without_payment`: 3
- `deny`: 1
- `investigate`: 15
- `pay_full`: 19
- `pay_partial`: 2

## Per-class F1

- `pay_full`: precision=0.000 recall=0.000 f1=0.000 support=19.0
- `pay_partial`: precision=0.000 recall=0.000 f1=0.000 support=2.0
- `deny`: precision=1.000 recall=1.000 f1=1.000 support=1.0
- `investigate`: precision=0.385 recall=1.000 f1=0.556 support=15.0
- `close_without_payment`: precision=0.000 recall=0.000 f1=0.000 support=3.0

## Notes

Gold `expected_outcome` is a deterministic function of skeleton features (complexity, injuries, damage vs deductible/reserve). Accuracy therefore tracks how well upstream extraction recovers those features for the predictive disposition rule — complementary to classification accuracy and extraction field F1.
