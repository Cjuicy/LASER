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

## NeuralRGBD segmentation comparison

The comparison profiles evaluate the same nine fixed NeuralRGBD kf10
sequences while changing only the segmentation strategy:

| Profile | Method | Ordinary cache mode |
|---|---|---|
| `mv_recon_laser_nrgbd_depth` | Depth/Felzenszwalb `300 / 1.1 / 500` | `auto` |
| `mv_recon_laser_nrgbd_geometry` | cross-product normals at 20 degrees | `readonly` |
| `mv_recon_laser_nrgbd_atomic` | conservative split at score `0.10` | `readonly` |

All three retain the paper window, confidence, anchor, loop, alignment, ICP,
normal, crop, map, and metric settings. These runs intentionally have
`subset` status because they omit 7-Scenes. A complete nine-sequence
NeuralRGBD dataset row is still produced and may be compared with the paper's
NeuralRGBD row.

Point-map assembly is locked to `laser-incremental-global-map-v1`. It replays
the released LASER Table 4 ordering inside the evaluation layer: the first
window supplies one fixed estimated intrinsic used to reproject every window's
predicted depth; each window is then registered against the already corrected
predecessor, and its global Sim(3) scale and layer scale are applied before it
becomes the next registration reference. Ordinary Pi3 predictions still come
from the shared prediction cache. This compatibility path is deliberately
local to `mv_recon`; the reusable inference, loop-closure, and segmentation
modules are not changed. The resolved protocol and manifest record this
assembly identity, and the comparison reader rejects results created with
deferred aggregation.

Depth is a hard reproduction gate. Geometry and Atomic must not run unless
all six Depth values display as the published row at three decimal places:

```text
0.020  0.010  0.012  0.004  0.713  0.856
```

### Cloud preparation

From the cloud checkout, expose the audited mixed-case data and checkpoint at
the configured paths, then verify the Python 3.11 extensions:

```bash
cd /root/autodl-tmp/LASER-pointmap-eval
ln -sfn NeuralRGBD data/nrgbd
ln -sfn PI3/model.safetensors weights/model.safetensors
test -d data/nrgbd/breakfast_room
test -f weights/model.safetensors

conda activate vggt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
python -c 'from inference_engine.utils.fast_seg import fast_graph_segmentation; from inference_engine.utils._segmentation_cy import merge_regions; print("extensions OK")'
python -c 'import torch, open3d; print(torch.__version__, torch.cuda.get_device_name(), open3d.__version__)'
python -m pytest tests/mv_recon -q
```

The canonical output directories below must be absent before a new run. If a
compatible partial directory already exists, inspect its manifest and repeat
the same command with `protocol.resume=true`; do not mix unrelated outputs.

### Depth reproduction gate

Run preflight and a one-sequence smoke first:

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_nrgbd_depth \
  protocol.preflight_only=true \
  output_dir=outputs/pointmap/nrgbd_depth_preflight

python mv_recon/eval.py \
  evaluation=mv_recon_laser_nrgbd_depth \
  protocol.max_sequences=1 \
  output_dir=outputs/pointmap/nrgbd_depth_smoke
```

Run all nine sequences in a persistent session:

```bash
mkdir -p outputs/pointmap/logs
screen -dmS nrgbd_depth bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vggt
cd /root/autodl-tmp/LASER-pointmap-eval
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
python mv_recon/eval.py evaluation=mv_recon_laser_nrgbd_depth \
  output_dir=outputs/pointmap/nrgbd_depth \
  > outputs/pointmap/logs/nrgbd_depth.log 2>&1
'
```

Monitor with `screen -ls`, `tail -n 80
outputs/pointmap/logs/nrgbd_depth.log`, and `nvidia-smi`. After a recoverable
interruption, use the same command and output directory with
`protocol.resume=true`.

Apply the gate before starting either later method:

```bash
python mv_recon/compare_results.py \
  --depth-run outputs/pointmap/nrgbd_depth \
  --depth-gate-only \
  --output-dir outputs/pointmap/nrgbd_comparison
python -m json.tool outputs/pointmap/nrgbd_comparison/depth_gate.json
```

A non-zero exit means the baseline has not reproduced the paper row; preserve
the output and diagnose it before continuing.

### Geometry and Atomic

After the Depth gate passes, use a one-sequence readonly smoke for each method:

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_nrgbd_geometry \
  protocol.max_sequences=1 \
  output_dir=outputs/pointmap/nrgbd_geometry_smoke

python mv_recon/eval.py \
  evaluation=mv_recon_laser_nrgbd_atomic \
  protocol.max_sequences=1 \
  output_dir=outputs/pointmap/nrgbd_atomic_smoke
```

Both smokes must report ordinary-cache hits, zero misses, and the same
`ordinary_prediction_key` as Depth. Run the full methods sequentially. Wait
for the Geometry screen session to finish successfully before starting Atomic:

```bash
screen -dmS nrgbd_geometry bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vggt
cd /root/autodl-tmp/LASER-pointmap-eval
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
python mv_recon/eval.py evaluation=mv_recon_laser_nrgbd_geometry \
  output_dir=outputs/pointmap/nrgbd_geometry \
  > outputs/pointmap/logs/nrgbd_geometry.log 2>&1
'

screen -dmS nrgbd_atomic bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vggt
cd /root/autodl-tmp/LASER-pointmap-eval
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
python mv_recon/eval.py evaluation=mv_recon_laser_nrgbd_atomic \
  output_dir=outputs/pointmap/nrgbd_atomic \
  > outputs/pointmap/logs/nrgbd_atomic.log 2>&1
'
```

Generate the final auditable comparison:

```bash
python mv_recon/compare_results.py \
  --depth-run outputs/pointmap/nrgbd_depth \
  --geometry-run outputs/pointmap/nrgbd_geometry \
  --atomic-run outputs/pointmap/nrgbd_atomic \
  --output-dir outputs/pointmap/nrgbd_comparison
cat outputs/pointmap/nrgbd_comparison/comparison.csv
python -m json.tool outputs/pointmap/nrgbd_comparison/comparison.json
```

The reader rejects run failures, wrong coverage/order, non-finite values,
results/manifest identity disagreement, changed checkpoint/map/input/GT
hashes, changed ordinary keys or thresholds, and different Git/runtime
provenance. It reports the six paper metrics, Chamfer-L1, sequence-macro
precision/recall/F-score at 1, 2, and 5 cm, metric direction, and method minus
Depth deltas without collapsing them into one score.

The audited cloud 7-Scenes copy is not eligible for this comparison: extracted
sequences lack registered `.depth.proj.png`, Office is empty, and redkitchen
is absent. No complete two-dataset Table 4 claim should be made until those
data are repaired independently.

## NeuralRGBD traditional-loop point-map experiment

`mv_recon_laser_nrgbd_depth_loop_traditional` is an explicitly non-paper
experiment. It keeps the validated nine-sequence NeuralRGBD kf10 Depth setup,
including 20/5 windows, but enables LASER's existing traditional path:
SALAD candidate detection, joint Pi3 constraint estimation, Sim(3)
optimization, and delayed global aggregation. Core loop, Pi3, segmentation,
anchor, and metric implementations are not changed.

The result is labeled `evaluation_mode=experiment` and
`pointmap_assembly=pipeline-loop-aggregate-v1`. This differs from the
paper-compatible no-loop assembly `laser-incremental-global-map-v1`; the
comparison reports their numerical delta but does not describe it as a pure
loop-only causal effect.

### Preparation and preflight

Use the cloud checkout and all three local weights. Preflight validates their
contents, the protocol, the nine-sequence map, data coverage, and runtime
without constructing Pi3 or a loop detector:

```bash
cd /root/autodl-tmp/LASER-pointmap-eval
conda activate vggt
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

ln -sfn NeuralRGBD data/nrgbd
ln -sfn PI3/model.safetensors weights/model.safetensors
test -d data/nrgbd/breakfast_room
test -f weights/model.safetensors
test -f weights/dino_salad.ckpt
test -f weights/dinov2_vitb14_pretrain.pth

python mv_recon/eval.py \
  evaluation=mv_recon_laser_nrgbd_depth_loop_traditional \
  protocol.preflight_only=true \
  output_dir=outputs/pointmap/nrgbd_depth_loop_traditional_preflight
```

Pi3, SALAD, and DINO SHA256 values participate in resume identity. Changing
any weight rejects reuse of an existing result directory.

### One-sequence smoke and diagnostics

Use a new output directory:

```bash
python mv_recon/eval.py \
  evaluation=mv_recon_laser_nrgbd_depth_loop_traditional \
  protocol.max_sequences=1 \
  output_dir=outputs/pointmap/nrgbd_depth_loop_traditional_smoke

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/pointmap/nrgbd_depth_loop_traditional_smoke")
manifest = json.loads((root / "protocol_manifest.json").read_text())
results = json.loads((root / "results.json").read_text())
print("mode:", manifest["evaluation_mode"])
print("assembly:", manifest["pointmap_assembly"])
print("loop:", manifest["loop_method"], manifest["loop_enabled"])
for sequence in results["sequences"]:
    diagnostics = sequence["cache_diagnostics"]
    print(
        sequence["sequence"],
        "candidates=", diagnostics["candidate_count"],
        "constraints=", diagnostics["constraint_count"],
        "rejected=", diagnostics["rejected_candidate_count"],
        "joint_forward=", diagnostics["joint_forward_count"],
        "no_loop_fallback=", diagnostics["used_no_loop_path"],
    )
PY
```

The dispatch and fields prove that the experiment path ran even if
`breakfast_room` has no valid loop. A loop was actually consumed when a
sequence has positive candidates, constraints, and joint forwards, with
`no_loop_fallback=False`. If the first sequence has none, run all fixed
sequences; do not tune the locked detector threshold after observing the
result.

### Full run and comparison

Run all nine sequences in a persistent session:

```bash
mkdir -p outputs/pointmap/logs
screen -dmS nrgbd_depth_loop_traditional bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vggt
cd /root/autodl-tmp/LASER-pointmap-eval
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
python mv_recon/eval.py \
  evaluation=mv_recon_laser_nrgbd_depth_loop_traditional \
  output_dir=outputs/pointmap/nrgbd_depth_loop_traditional \
  > outputs/pointmap/logs/nrgbd_depth_loop_traditional.log 2>&1
'

screen -ls
tail -n 80 outputs/pointmap/logs/nrgbd_depth_loop_traditional.log
```

After the run finishes successfully, compare it with the preserved cloud
no-loop Depth result:

```bash
python mv_recon/compare_loop_results.py \
  --no-loop-run outputs/pointmap/nrgbd_depth_replay \
  --traditional-loop-run \
    outputs/pointmap/nrgbd_depth_loop_traditional \
  --output-dir outputs/pointmap/nrgbd_loop_comparison

cat outputs/pointmap/nrgbd_loop_comparison/loop_comparison.csv
python -m json.tool \
  outputs/pointmap/nrgbd_loop_comparison/loop_comparison.json
```

`outputs/pointmap/nrgbd_depth_replay` is the existing audited cloud result. A
fresh clone must first run the full `mv_recon_laser_nrgbd_depth` profile and
pass that result directory to `--no-loop-run`. The reader accepts the older
audited result format by hash-checking its `resolved_protocol.yaml`; it still
rejects wrong coverage, checkpoints, maps, inputs, GT, ordinary cache keys,
runtime, thresholds, or metric settings.

## Custom sampling experiments

A sequence map records a mapping from sequence name to selected frame IDs.
Precomputed maps keep evaluation deterministic and separate sampling from
metric computation. Use `mv_recon/sampling.py` and the configs under
`configs/data/` to create experiment-specific maps. Such experiments are not
accepted by the immutable `mv_recon_laser_paper` profile as Table 4 runs.
