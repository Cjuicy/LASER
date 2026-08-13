# PI3 Ordinary Prediction Cache

The cache boundary contains only ordinary PI3 window predictions. It is shared
by all segmentation and reconstruction experiments.

```text
RGB window -> PI3 ordinary forward -> depth/confidence/poses/reference intrinsic
                                      -> persistent prediction cache
                                      -> current segmentation/reconstruction mode
```

It excludes segmentation labels, anchor state, loop candidates, joint A/B
prediction, constraints, optimization state, and evaluation settings.

## Modes

| Mode | Read | Write | Missing/corrupt behavior |
|---|---:|---:|---|
| `auto` | yes | yes | compute missing; quarantine corrupt data and rebuild |
| `refresh` | no reuse | yes | invalidate the old entry and rebuild |
| `readonly` | yes | no | fail on missing or corrupt data |
| `off` | no | no | run PI3 for every window |

Within a matrix, requested `refresh` applies to entry zero; later entries use
`readonly`, so one fresh prediction stream is reused safely.

## Identity

The SHA-256 prediction key includes model/checkpoint bytes, ordered image
content, preprocessing/runtime source, dtype, sample stride, window size,
overlap, and the complete `WindowSpec` schedule.

It excludes segmentation method, atomic split mode, anchor settings,
reconstruction mode, loop settings, evaluation config, output directory, Git
metadata, and device identity. Those downstream changes therefore reuse the
same ordinary predictions.

## Layout and integrity

```text
<prediction_cache.root>/v2/<prediction-key>/
  manifest.json
  sequence.json
  complete.json
  windows/000000.pt
  windows/000001.pt
  locks/entry.lock
  invalid/...
```

Each artifact has a digest bound to the prediction key. Writes use temporary
files and atomic replacement; invalid ranges, copied artifacts, truncation,
value mutation, and false completion markers are rejected.

## Validation

```bash
python -m pytest -q \
  tests/test_prediction_cache.py \
  tests/test_prediction_cache_integrity.py \
  tests/test_pipeline_cache_integration.py \
  tests/test_prediction_cache_locking.py

python run_experiment_matrix.py \
  --config configs/experiments/ate_matrix.yaml \
  --dry-run
```

For a two-window cold `auto` run, expect two ordinary forwards. A warm
`readonly` run with the same prediction identity should have two hits and zero
ordinary forwards. Loop modes may still perform joint inference because joint
evidence is intentionally outside the ordinary cache.
