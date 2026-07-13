# Loop-Closure Cloud-Ready Design

## Objective

Make the existing AutoDL checkout run loop-closure inference reliably on a
flat image sequence such as `data/00/image_2`, while using the current
layer-atomic segmentation by default. The Pi3, SALAD, and DINO weights already
exist in `weights/`; this change does not download, copy, or package them.

The implementation starts from commit
`7dca6248b58ebbeacdda9cdf108aab439c527479`, where layer-atomic geometry
segmentation is already connected to `StreamingWindowEngineLC` whenever depth
refinement is enabled.

## Scope

This change will:

- create one canonical, naturally sorted and sampled image manifest;
- reuse that exact manifest for streaming inference, SALAD retrieval, and loop
  constraint construction;
- enable layer-atomic depth refinement by default in `demo_lc.py`;
- retain an explicit `--no-depth-refine` opt-out;
- use `configs/loop_config.yaml` as the default loop configuration;
- validate the local AutoDL inputs before expensive inference begins;
- verify that inference produced the expected number of window caches before
  loop closure starts;
- print concise mode, frame, window, and loop-pair diagnostics;
- document the exact AutoDL checkout and run commands;
- add focused unit and regression tests.

This change will not:

- change the layer-atomic segmentation algorithm or its thresholds;
- change SALAD similarity thresholds, NMS, Sim(3) estimation, or optimization;
- add Docker, Conda, dataset download, or weight download support;
- merge `demo.py` and `demo_lc.py` into a new runner;
- change cache tensor formats or visualization output formats.

## Canonical Image Manifest

Create `utils/image_paths.py` as the single image-discovery implementation.
It will expose these interfaces:

```text
natural_sort_key(path: str | os.PathLike[str]) -> list[tuple[int, str | int]]
discover_images(data_path: str | os.PathLike[str], sample_interval: int = 1) -> list[str]
```

`discover_images` will:

1. require `data_path` to be an existing directory;
2. require `sample_interval >= 1`;
3. accept `.png`, `.jpg`, and `.jpeg` case-insensitively;
4. sort embedded numeric fields numerically, so `000002.png` and
   `frame2.png` precede `000010.png` and `frame10.png`;
5. apply `sample_interval` exactly once after sorting;
6. return path strings suitable for the existing image loaders.

Both demos will import these functions instead of maintaining separate copies.
The imported names remain available from `demo` and `demo_lc`, preserving the
current test and caller surface.

`demo_lc.py` will discover the image manifest once. It will pass the same
ordered list to the streaming engine and `LoopClosureEngine`. The loop engine
will pass the list to `LoopDetector`. When an explicit manifest is supplied,
neither loop component may scan the directory or apply sampling again.

For backwards compatibility, `LoopClosureEngine` and `LoopDetector` will keep
their directory-based construction paths. They will use the shared discovery
function only when no explicit manifest is supplied.

## Loop-Closure Entry Point

`demo_lc.py` remains the cloud entry point. Its CLI changes are:

```text
--config_path defaults to configs/loop_config.yaml
--depth-refine explicitly enables refinement
--no-depth-refine explicitly disables refinement
depth refinement defaults to enabled
```

Therefore the ordinary loop-closure command uses
`segment_point_map_layer_atomic` without requiring an extra flag. Startup logs
will state both `Loop closure: enabled` and
`Layer-atomic depth refinement: enabled|disabled`.

The scene name will default to the input directory name. For
`data/00/image_2`, the default scene name is `image_2`; callers may continue to
override it with `--scene_name`, for example `--scene_name kitti_00`.

## Validation and Failure Handling

Before model construction, the entry point will validate:

- CUDA is available, because the current runner uses CUDA timing and inference;
- `window_size > overlap >= 1`;
- the Pi3 checkpoint exists;
- the loop configuration exists and loads successfully;
- `Model.loop_enable` is true for this loop-closure-only entry point;
- the configured SALAD and DINO checkpoints exist;
- the image manifest is non-empty and contains more images than `overlap`.

Validation failures will raise a direct error naming the missing or invalid
value before GPU inference starts.

The runner will compute the sliding windows once and record the expected window
count. After streaming inference, it will sort cache files by numeric cache id
and require the actual count to match the expected count. This converts silent
worker failure or incomplete cache output into a clear error before loop
optimization reads inconsistent data.

No detected loop pair is a valid outcome, not a crash. The runner will log zero
pairs, preserve the adjacent-window Sim(3) transforms, aggregate the caches,
and still save visualization results.

## Data Flow

```text
data/00/image_2
    -> shared natural sort and single sampling pass
    -> canonical image manifest
       -> StreamingWindowEngineLC
          -> adjacent-window registration
          -> layer-atomic segmentation and depth refinement (default on)
          -> numbered window caches
       -> SALAD LoopDetector using the same frame indices
       -> LoopClosureEngine loop-window constraints
       -> Sim(3) optimization
       -> corrected cache aggregation
       -> viser_results/kitti_00
```

The invariant is that frame index `i` refers to the same path in every stage.

## Diagnostics

The cloud log will include:

- input directory;
- sampled frame count plus the first and last file names;
- window size, overlap, and expected window count;
- loop-closure and depth-refinement mode;
- number of completed cache windows;
- number of SALAD loop pairs;
- whether Sim(3) optimization ran or original transforms were retained;
- final output directory and scene name.

The log will not dump the complete manifest or large tensors.

## Tests

Focused tests will cover:

1. mixed-case `.png`, `.jpg`, and `.jpeg` filtering;
2. natural numeric ordering and sampling after ordering;
3. invalid directory and sampling interval errors;
4. `demo.py` and `demo_lc.py` using the shared helper;
5. loop CLI defaults: standard config and depth refinement enabled;
6. `--no-depth-refine` disabling the new segmentation path;
7. an explicit manifest passing through `LoopClosureEngine` and
   `LoopDetector` without rescanning or resampling;
8. cache-count mismatch producing a clear error;
9. zero loop pairs preserving the existing transforms;
10. all existing segmentation and demo tests remaining green.

The local verification command is:

```bash
python setup.py build_ext --inplace
pytest -q
```

Full model inference cannot be reproduced in the local checkout because its
large weights and AutoDL GPU are intentionally out of scope. The final handoff
will include a lightweight AutoDL smoke command and the full KITTI 00 command.

## AutoDL Handoff

After the implementation branch is pushed, the cloud checkout will use:

```bash
cd ~/autodl-tmp/LASER
git fetch origin codex/loop-closure-cloud-ready
git switch codex/loop-closure-cloud-ready
python setup.py build_ext --inplace
pytest -q
```

The full KITTI 00 run will use the existing weights:

```bash
python demo_lc.py \
  --data_path data/00/image_2 \
  --scene_name kitti_00 \
  --cache_path inference_cache/kitti_00 \
  --output_path viser_results \
  --window_size 10 \
  --overlap 5
```

No `--depth-refine` flag is needed because layer-atomic depth refinement is the
loop-mode default. Add `--no-depth-refine` only for an intentional baseline
comparison.
