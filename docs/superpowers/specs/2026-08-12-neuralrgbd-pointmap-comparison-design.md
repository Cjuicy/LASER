# NeuralRGBD Point-Map Method Comparison Design

## Objective

Run an auditable NeuralRGBD point-map comparison on the cloud using the exact
nine LASER kf10 sequences. The experiment must first reproduce the published
Pi3+LASER Depth baseline at the paper's reported three-decimal precision, then
compare Geometry and Atomic segmentation while holding every other variable
fixed.

The existing `mv_recon_laser_paper` profile remains immutable and continues to
mean the complete 7-Scenes plus NeuralRGBD Table 4 protocol. Comparison support
is additive and remains inside the evaluation layer.

## Cloud Audit Evidence

The audited cloud checkout is commit
`ac2340cc90c874fd839141aa89b29587de32ed7e` on branch
`codex/laser-paper-pointmap-eval`. The environment uses Python 3.11.15,
PyTorch 2.12.0 with CUDA 13.0, an RTX 5090, NumPy 1.26.4, SciPy 1.17.1, and
Open3D 0.19.0. Python 3.11 builds of both Cython extensions are present.

All nine NeuralRGBD sequences passed a content audit over the exact 1,101 kf10
frames in the checked-in sequence map:

- every RGB file exists, decodes, and has size 640 by 480;
- every depth file exists, decodes, has shape 480 by 640, and contains valid
  depth samples;
- every `poses.txt` parses into finite 4 by 4 matrices and covers the last
  requested frame.

7-Scenes is excluded from this experiment. Its nine currently extracted
paper sequences have valid raw RGB, raw depth, and pose files but no registered
`depth.proj.png`. Office is empty and redkitchen is absent with an unreadable
archive index. No complete 7-Scenes paper result may be claimed from this
cloud dataset.

## Alternatives Considered

### NeuralRGBD-first comparison (selected)

Evaluate all nine valid NeuralRGBD sequences now. This produces a complete
NeuralRGBD dataset row that can be checked against the published row and gives
the strongest immediately available controlled comparison.

### Repair all 7-Scenes before any numerical run

This is required for a complete two-dataset Table 4 reproduction, but it
requires reacquiring Office and redkitchen and generating registered depth for
all required sequences. It delays useful NeuralRGBD evidence without improving
the fairness of the NeuralRGBD comparison.

### Use `protocol.max_sequences=5` without code changes

This avoids Office but evaluates only the first five sequences of each dataset
and cannot run the full valid NeuralRGBD row. It is suitable only for a smoke
test and is rejected as the main comparison.

## Protocol Profiles

Add a comparison mode to the typed evaluation protocol while preserving paper
mode unchanged. Paper mode still requires the exact ordered dataset pair,
Depth segmentation, fixed map paths and hashes, and all existing locked
fields. Comparison mode may select a non-empty ordered subset of the paper
datasets and one of three locked segmentation variants.

Three explicit Hydra profiles make runs inspectable without complex command
line overrides:

1. `mv_recon_laser_nrgbd_depth`
2. `mv_recon_laser_nrgbd_geometry`
3. `mv_recon_laser_nrgbd_atomic`

Every profile selects only `NRGBD-dense`, retains the fixed NeuralRGBD sequence
map and SHA256, and selects all nine entries when `max_sequences` is null.
Comparison runs serialize as `subset` at run and dataset level because they do
not cover the complete two-dataset paper protocol. The NeuralRGBD primary row
still averages all nine selected sequences and is directly comparable to the
published NeuralRGBD row.

## Controlled Variables

All methods use:

- Pi3 checkpoint `weights/model.safetensors` with its recorded content hash;
- the exact 1,101 NeuralRGBD kf10 frames from the checked-in map;
- window size 20 and overlap 5;
- segmentation confidence keep ratio 0.5;
- depth merge threshold 0.1 and temporal IoU threshold 0.3;
- anchor propagation enabled at correspondence IoU 0.4;
- loop closure disabled with traditional aggregation;
- adjacent-window registration confidence keep ratio 0.5;
- a 224 by 224 center crop;
- same-pixel Umeyama Sim(3), then point-to-point Open3D ICP at 0.1 m;
- Open3D default normal estimation;
- the GT validity mask without predicted-confidence metric filtering;
- identical CUDA device, dtype, dependency environment, and metric schema.

Only segmentation variant changes:

| Method | Segmentation lock |
|---|---|
| Depth | `method=depth`; Felzenszwalb `300 / 1.1 / 500` |
| Geometry | `method=geometry`; cross-product normals; 20 degree threshold |
| Atomic | `method=atomic`; conservative split; score threshold 0.10 |

## Ordinary Prediction Cache

Depth runs first with prediction-cache mode `auto`. Geometry and Atomic run
with mode `readonly` against the same cache root. This makes a missing or
incompatible ordinary prediction an error instead of silently recomputing it.
The final comparison verifies that all three sequence records report identical
ordinary-prediction keys. Segmentation, alignment, and metric artifacts remain
method-specific and use separate output directories.

## Execution Gates

Run each method in this order:

1. typed protocol and dataset preflight;
2. one-sequence smoke run;
3. all-nine-sequence run with resume enabled after interruption;
4. result and manifest validation.

Depth is a hard gate. Its all-nine result must:

- have no failures;
- contain exactly the nine expected sequence names;
- report finite primary and diagnostic metrics;
- produce the published NeuralRGBD primary values when each field is rounded
  to three decimal places: `0.020, 0.010, 0.012, 0.004, 0.713, 0.856`.

If Depth misses that gate, stop before Geometry and Atomic and diagnose the
baseline discrepancy. No method ranking is reported from an unverified
baseline environment.

## Outputs and Comparison

Use separate canonical output roots:

```text
outputs/pointmap/nrgbd_depth
outputs/pointmap/nrgbd_geometry
outputs/pointmap/nrgbd_atomic
outputs/pointmap/nrgbd_comparison
```

Add a comparison reader that accepts the three `results.json` files, rejects
different dataset/sequence coverage, non-finite metrics, incomplete failures,
or different ordinary-prediction keys, and writes:

```text
comparison.json
comparison.csv
```

Rows contain the six primary metrics, Chamfer-L1, threshold precision/recall/
F-score, and deltas from Depth. Distance metrics are lower-is-better; normal
consistency, precision, recall, and F-score are higher-is-better. The reader
does not choose a winner by collapsing unlike metrics into one score.

## Failure and Recovery

Every numerical run keeps the existing atomic result writer. A sequence
failure or Ctrl-C preserves completed sequences and records the failure.
Resume requires the same code, protocol identity, pipeline hash, checkpoint,
sequence map, input images, GT point maps/masks, and metric schema.

Cloud commands run in a persistent `screen` session. Before numerical work,
create repository-local lowercase data links for the configured dataset root
and verify the Pi3 checkpoint link. No 7-Scenes archive is extracted or deleted
as part of this experiment.

## Scope Boundary

Allowed changes are limited to:

```text
configs/evaluation/
mv_recon/
tests/mv_recon/
docs/
```

Do not modify `pipeline/`, `inference_engine/`, Pi3 model code, segmentation
implementations, anchor propagation, loop closure, or the ordinary prediction
cache implementation. The profiles only select existing strategies through
the typed evaluation boundary.

## Acceptance Criteria

1. Existing paper-profile tests remain green and paper mode rejects method or
   dataset drift exactly as before.
2. Comparison mode accepts NeuralRGBD-only Depth, Geometry, and Atomic profiles
   and rejects any unlisted segmentation setting.
3. All profiles retain the fixed NeuralRGBD map path, hash, sequence order, and
   paper geometry metric implementation.
4. A full comparison run is visibly `subset`, never a complete two-dataset
   Table 4 claim.
5. The Depth gate checks all six rounded published values before later methods
   may run.
6. Geometry and Atomic reuse the exact ordinary prediction keys produced by
   Depth.
7. The comparison reader rejects coverage, identity, cache-key, failure, and
   non-finite mismatches.
8. The complete local test suite passes before cloud execution.
9. Cloud manifests record the commit, dependency/GPU versions, hashes,
   selected sequences, method, cache diagnostics, and final state.
