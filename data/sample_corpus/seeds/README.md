# Sample corpus seed exports

Regenerable DICIE-shaped JSONL produced by:

```bash
python -m src.storage seed --seed 42 --also-export
```

These files are **synthetic** medical-bill and salvage-claim examples for
analysis and fine-tuning. They are not proprietary insurer documents.

The live SQLite database (`../documents.db`) is gitignored; rebuild it with
the same seed command.
