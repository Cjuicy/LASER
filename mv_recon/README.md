# Multi-view Reconstruction (Point-Map Estimation)

## LASER paper-compatible evaluation

`mv_recon/eval.py` evaluates the Pi3 streaming reconstruction on the fixed
LASER indoor point-map protocol. The strict profile is
`configs/evaluation/mv_recon_laser_paper.yaml`.

The locked baseline is:

- 7-Scenes and NeuralRGBD, using the shipped kf10 sequence maps;
- window size 20 and overlap 5;
- Depth/Felzenszwalb segmentation (`300 / 1.1 / 500`);
- segmentation and adjacent-window registration confidence keep ratios 0.5;
- anchor propagation enabled at correspondence IoU 0.4;
- loop closure disabled with traditional aggregation;
- 224 by 224 center crop;
- Umeyama Sim(3), then Open3D point-to-point ICP at 0.1 metres;
- Open3D default normal estimation.

Changing a locked value while retaining the paper profile raises a protocol
drift error before Pi3 is constructed. Checkpoint, device, output directory,
pipeline temporary cache, and ordinary prediction cache locations remain
operational settings and are recorded in the manifest.

### Environment and data

Use the repository Python 3.11 environment because Open3D does not provide a
wheel for every newer Python release.

```bash
conda create -n laser-eval -y python=3.11
conda activate laser-eval
python -m pip install -r requirements.txt
python setup.py build_ext --inplace
```

Place the local Pi3 checkpoint at:

```text
weights/model.safetensors
```

The default dataset roots are:

```text
data/7scenes
data/nrgbd
```

The evaluator uses these fixed maps:

```text
datasets/seq-id-maps/7scenes_mv-recon_seq-id-map-kf10.json  e9954bfcf4b4a3273224e8375d468638e1fe4d7b6d926ff32147367bb4574008
datasets/seq-id-maps/NRGBD_mv-recon_seq-id-map-kf10.json      f18f2143f8a373727aa4d7043b779b77639354fda80523cdc2a139164ddc33ba
```

Strict mode validates both the repository-relative paths and SHA256 values.

### Cloud execution

Run preflight in its own output directory. It validates the protocol,
checkpoint, sequence maps, dependency/runtime metadata, and dataset sequence
coverage without constructing Pi3:

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_paper \
  protocol.preflight_only=true \
  output_dir=outputs/mv_recon_laser_paper_preflight
```

Run one sequence from each dataset as a GPU/Open3D smoke test:

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_paper \
  protocol.max_sequences=1 \
  output_dir=outputs/mv_recon_laser_paper_smoke
```

This run has `subset` status and is not a Table 4 result.

Run the full benchmark:

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_paper \
  output_dir=outputs/mv_recon_laser_paper
```

Resume an interrupted run only with the identical code, protocol, pipeline,
checkpoint, sequence maps, image manifests, GT point maps/masks, and metric
schema:

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_paper \
  protocol.resume=true \
  output_dir=outputs/mv_recon_laser_paper
```

Without `protocol.resume=true`, a non-empty result directory is rejected. The
ordinary Pi3 prediction cache is independent from metric result resume and can
still be reused when its existing fingerprint matches.

### Primary metrics

For each sequence, the evaluator reports:

```text
accuracy_mean_m
accuracy_median_m
completion_mean_m
completion_median_m
normal_consistency_mean
normal_consistency_median
```

Accuracy queries predicted points against GT. Completion queries GT points
against prediction. NC mean averages the two directional NC means; NC median
averages the two directional NC medians. Dataset rows are macro averages over
sequences, not pooled point averages.

The final metric point set uses only the GT validity mask. Predicted confidence
does not filter metric points. The two locked 0.5 confidence ratios affect
Depth layer extraction and adjacent-window registration inside reconstruction.

The profile stores these Pi3+LASER Table 4 reference values:

| Dataset | Acc mean | Acc median | Comp mean | Comp median | NC mean | NC median |
|---|---:|---:|---:|---:|---:|---:|
| 7-Scenes | 0.013 | 0.005 | 0.017 | 0.006 | 0.607 | 0.665 |
| NeuralRGBD | 0.020 | 0.010 | 0.012 | 0.004 | 0.713 | 0.856 |

`delta_to_paper` is reported for diagnosis only; the evaluator does not choose
a numerical reproduction tolerance automatically.

### Result artifacts

Each numerical run writes:

```text
resolved_protocol.yaml
resolved_pipeline.yaml
protocol_manifest.json
results.json
summary.csv
sequences.csv
failures.jsonl
```

`results.json` is canonical. `summary.csv` and `sequences.csv` are derived
views. `protocol_manifest.json` records source/config/checkpoint/map hashes,
runtime versions, selected sequences, input manifest hashes, ordinary
prediction keys, GT point-map/mask hashes, and cache counters. Ctrl-C records
the active sequence as interrupted and leaves the run resumable.

Only `complete` is a full paper-protocol result. `subset` is a deliberate
limited run. `incomplete` and `failed` preserve diagnostics and successful
sequence artifacts but the command exits non-zero after a strict sequence
failure.

## Custom sampling experiments

A sequence map records a mapping from sequence name to selected frame IDs.
Precomputed maps keep evaluation deterministic and separate sampling from
metric computation. Use `mv_recon/sampling.py` and the configs under
`configs/data/` to create experiment-specific maps. Such experiments are not
accepted by the immutable `mv_recon_laser_paper` profile as Table 4 runs.
