# NeuralRGBD Traditional-Loop Point-Map Experiment Design

**Date:** 2026-08-12

## Goal

Add an explicitly non-paper NeuralRGBD point-map experiment that changes the
validated Depth reconstruction by enabling LASER's existing traditional loop
closure path. The experiment must produce the same auditable point-map metrics
as the no-loop Depth run while keeping every non-loop input and metric setting
fixed.

The experiment answers one question: how do the nine NeuralRGBD kf10 point-map
results change when traditional loop detection and optimization are enabled?
It does not claim to reproduce LASER Table 4 and must not weaken or alter the
paper or comparison profiles.

## Scope

In scope:

- all nine valid NeuralRGBD kf10 sequences;
- Pi3 windows of 20 selected frames with five-frame overlap;
- Depth/Felzenszwalb segmentation and anchor propagation;
- SALAD loop-candidate detection using the configured DINO checkpoints;
- existing joint Pi3 loop-constraint estimation;
- existing traditional Sim(3) loop strategy and aggregation;
- the existing Umeyama, ICP, Acc, Comp, NC, Chamfer, and F-score evaluator;
- ordinary prediction-cache reuse and full result provenance;
- a controlled comparison against the no-loop Depth run.

Out of scope:

- changes to Pi3 inference, SALAD, candidate selection, joint alignment,
  traditional optimization, segmentation, anchor propagation, or metrics;
- corrected loop closure;
- 7-Scenes, whose current cloud copy is incomplete;
- changing the immutable paper and Depth/Geometry/Atomic comparison profiles;
- treating the result as a paper baseline.

## Alternatives Considered

### Dedicated experiment profile and evaluator dispatch (selected)

Add a typed experiment mode and a dedicated profile. The evaluator chooses a
point-map assembly implementation from the resolved profile: paper-compatible
incremental replay for existing runs, or the existing pipeline loop path for
this experiment. This gives explicit provenance, validation, resume behavior,
and comparable metric artifacts without changing core loop implementations.

### Pose evaluation followed by an ad-hoc point-map script

`eval_launch.py` can exercise loop closure, but it does not provide the strict
point-map result identity, GT hashes, coverage validation, or comparison
artifacts. A second script would duplicate reconstruction and metric logic.
This is rejected because it would be harder to audit and easier to misconfigure.

### Unlock the paper profile

Allowing `loop.enabled=true` in the paper profile would be a small code change,
but it would destroy the existing guarantee that a paper-labeled result is
immutable. This is rejected.

## Configuration

Add a Hydra evaluation profile named:

```text
mv_recon_laser_nrgbd_depth_loop_traditional
```

It inherits the validated NeuralRGBD Depth settings but resolves to an
experiment protocol. The locked reconstruction settings are:

```yaml
eval_datasets: [NRGBD-dense]

protocol:
  name: laser_neuralrgbd_pointmap_loop_experiment
  mode: experiment
  pointmap_assembly: pipeline-loop-aggregate-v1
  prediction_cache_mode: auto

pipeline:
  window:
    size: 20
    overlap: 5
  segmentation:
    method: depth
    confidence_keep_ratio: 0.5
  anchor_propagation:
    enabled: true
    correspondence_iou_threshold: 0.4
  loop:
    enabled: true
    method: traditional
    registration:
      confidence_keep_ratio: 0.5
```

The profile also retains the Depth Felzenszwalb values `300 / 1.1 / 500`,
temporal IoU `0.3`, 224-pixel center crop, Umeyama Sim(3), point-to-point ICP
at 0.1 metres, Open3D default normals, and 1/2/5 cm F-score thresholds.

The existing loop-detection and constraint settings come from the typed
pipeline configuration and are frozen into the resolved pipeline and manifest.
The cloud run requires:

```text
weights/dino_salad.ckpt
weights/dinov2_vitb14_pretrain.pth
weights/model.safetensors
```

Preflight validates all three local files before model or detector creation.

## Protocol and Provenance

Extend the protocol mode enum from `paper | comparison` to
`paper | comparison | experiment`. Experiment mode is accepted only when all
of the following hold:

- dataset coverage is exactly the nine fixed NeuralRGBD sequences;
- segmentation is Depth;
- loop closure is enabled;
- loop method is traditional;
- point-map assembly is `pipeline-loop-aggregate-v1`;
- all non-loop reconstruction and metric fields match the Depth comparison
  profile;
- ordinary prediction-cache mode is `auto`;
- required loop checkpoints are local files.

The result manifest records:

- `evaluation_mode: experiment`;
- `pointmap_assembly: pipeline-loop-aggregate-v1`;
- segmentation and loop method;
- loop detector and optimizer configuration hashes;
- checkpoint, sequence-map, image-manifest, and GT hashes;
- ordinary cache keys and counters;
- loop candidate, accepted constraint, rejected constraint, and joint-forward
  counts for every sequence.

Experiment results use `subset` state because they are a NeuralRGBD-only
controlled experiment, not the two-dataset paper protocol.

## Reconstruction Data Flow

For each sequence:

1. Load the fixed kf10 image list and build 20/5 `WindowSpec` values.
2. Obtain ordinary Pi3 predictions from the shared content-addressed cache.
3. Run the existing window pipeline to create the per-window reconstruction
   caches required by the traditional strategy.
4. Run the existing SALAD candidate detector on the immutable image manifest.
5. Map valid cross-window candidates to joint Pi3 alignment requests.
6. Build traditional relative Sim(3) loop constraints. Candidate-local
   validation failures are recorded using existing rejection semantics; model,
   checkpoint, CUDA, and resource failures terminate the sequence.
7. Optimize and aggregate with the existing traditional loop strategy.
8. Convert aggregated local points and camera poses to global point maps.
9. Resize to GT resolution, then run the unchanged Umeyama/ICP geometry
   evaluator.

The experiment path must not call the paper incremental replay. Conversely,
paper and comparison modes must never call the loop experiment path.

## Cache Semantics

Ordinary Pi3 predictions remain shared with the no-loop Depth run. A warm
experiment should report ordinary hits and no ordinary misses when the image
manifest, checkpoint, model preprocessing, and 20/5 schedule match.

Loop candidate descriptors, candidate-local joint Pi3 predictions, loop
constraints, optimizer state, and aggregated point maps are not stored in the
ordinary prediction cache. Joint forward counts may therefore be non-zero even
when ordinary forward counts are zero.

## Result Comparison

Add a loop-experiment comparison command that accepts:

```text
--no-loop-run <validated NeuralRGBD Depth result>
--traditional-loop-run <experiment result>
--output-dir <comparison directory>
```

The reader validates matching checkpoint, sequence map, sequence order, input
manifests, GT hashes, metric schema, window schedule, segmentation, anchor,
crop, alignment, and runtime-library/GPU identity. It requires different
loop/assembly identities and rejects any failed or incomplete sequence. Git
commits are recorded independently but are not required to match, because the
preserved no-loop run predates the experiment entry point. The no-loop commit
must nevertheless identify the known
`laser-incremental-global-map-v1` assembly, and the experiment commit must
identify `pipeline-loop-aggregate-v1`.

The output contains:

```text
loop_comparison.json
loop_comparison.csv
```

It reports raw metrics for both methods and `traditional_loop - no_loop`
deltas. Lower-is-better fields are Acc, Comp, Chamfer; higher-is-better fields
are NC, precision, recall, and F-score. No single combined score is introduced.
The paper Depth gate is not applied to this experiment comparison.

## Failure Handling

- Missing SALAD, DINO, or Pi3 weights fail during preflight.
- A non-local checkpoint is rejected; there is no network fallback.
- Protocol drift fails before model construction.
- An output directory is never mixed with another protocol identity.
- Interrupted runs preserve completed sequence artifacts and may resume only
  with an identical identity.
- A sequence-level failure is recorded in `failures.jsonl` and makes the run
  incomplete or failed.
- Zero loop candidates are valid and are recorded explicitly; that sequence
  then uses the traditional no-loop fallback but remains part of the
  experiment.

## Test Strategy

Tests are written before implementation and cover:

1. experiment protocol accepts only NeuralRGBD Depth with traditional loop;
2. paper and comparison modes continue rejecting enabled loop closure;
3. missing SALAD/DINO checkpoints fail preflight before model construction;
4. experiment inference dispatch invokes detector, constraint construction,
   optimization, and aggregation exactly once;
5. experiment inference never calls paper incremental replay;
6. paper/comparison inference never calls the experiment loop path;
7. aggregated local points and poses become correctly shaped global point maps;
8. ordinary cache identity is unchanged from the no-loop Depth profile;
9. result manifest records experiment/assembly/loop diagnostics;
10. comparison rejects mismatched inputs, GT, coverage, runtime, or methods;
11. comparison emits correct raw metrics, directions, and loop-minus-no-loop
    deltas;
12. the complete local suite passes;
13. cloud preflight and one-sequence smoke show real candidate/constraint
    diagnostics before the nine-sequence command is handed off.

## Cloud Acceptance Sequence

The cloud commands run in this order:

1. pull the feature branch and build extensions;
2. run `tests/mv_recon`;
3. run experiment preflight;
4. run a one-sequence smoke in a new output directory;
5. verify the manifest reports `experiment`, `traditional`, and
   `pipeline-loop-aggregate-v1`;
6. verify loop candidate/constraint fields exist and cache identity matches the
   no-loop Depth run;
7. run all nine NeuralRGBD sequences;
8. compare the full experiment against the preserved no-loop Depth result.

The implementation is accepted when the smoke proves that real traditional
loop closure was consumed, all local and cloud evaluator tests pass, and the
commands cannot label the output as a paper baseline.
