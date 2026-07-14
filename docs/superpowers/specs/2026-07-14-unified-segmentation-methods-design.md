# Unified LASER Segmentation Methods Design

**Date:** 2026-07-14

**Target branch:** `codex/unified-segmentation-methods`

**Branch base:** `feature/layer-atomic-geometry` at `7dca624`

## Goal

Run three already validated segmentation methods from one LASER checkout so
future experiments can compare them under the same model, confidence filtering,
windowing, segment graph, scale estimation, scale propagation, cache, and loop
closure code:

- `depth`: original LASER depth segmentation;
- `geometry`: the geometry-aware method from `Cjuicy/LASER-Geometry`;
- `layer_atomic`: the method from `Cjuicy/LASER` branch
  `feature/layer-atomic-geometry`.

This change unifies selection and execution. It does not add comparison metrics
or experiment reports.

## Fixed Source Versions

The implementation is pinned to these validated source revisions:

- LASER depth baseline: the implementation inherited by `7dca624`;
- LASER-Geometry: `Cjuicy/LASER-Geometry/main` at `340c599`;
- layer-atomic geometry: `Cjuicy/LASER` at `7dca624`.

The three segmentation algorithms remain independently callable. Their
formulas, thresholds, merge decisions, and method-specific function signatures
must not be rewritten merely to make routing uniform.

## Scope

### In scope

- Add `depth`, `geometry`, and `layer_atomic` segmentation modes.
- Preserve the current depth and layer-atomic implementations.
- Port the LASER-Geometry segmentation core and its required normal/geometry
  helpers.
- Use one thin graph-building router in normal streaming and loop-closure
  streaming.
- Expose the same mode choice through `demo.py` and `demo_lc.py`.
- Document effective parameters and commands.
- Add characterization, routing, CLI, engine, and smoke tests.

### Out of scope

- New reconstruction, pose, depth, or segmentation metrics.
- New experiment dashboards, playback pages, or visualization reports.
- LASER-Geometry alignment-debugging and confidence-weighted scale-anchor work.
- Changes to segment matching, scale anchor estimation, scale propagation,
  Sim(3), cache aggregation, or loop-closure algorithms.
- Threshold tuning or performance claims about any segmentation method.

## Architecture

All three modes produce per-frame integer label maps. Only label generation
varies; every downstream operation is shared.

```text
point maps + confidence
          |
          v
thin segmentation-mode router
  |             |               |
  v             v               v
depth        geometry       layer_atomic
labels        labels           labels
  |             |               |
  +-------------+---------------+
                |
                v
     match_segmentation_seq
                |
                v
 shared scale anchors and propagation
                |
                v
 shared streaming/cache/loop-closure flow
```

The unified branch is based on the layer-atomic branch because that branch
already contains the verified new algorithm and its scale-invariance fix. The
LASER-Geometry repository has unrelated Git history, so only its pinned
segmentation core is ported; its repository history is not merged wholesale.

## Segmentation Methods

### `depth`

The router extracts `point_map[..., -1]` and invokes the existing
`segment_depth_felzenszwalb_rag(...)`. No geometry-only arguments enter this
path.

### `geometry`

The router invokes the pinned LASER-Geometry implementation. It keeps:

- the four-channel Felzenszwalb input composed from normalized depth and the
  three normal channels;
- `cross` and `sobel` normal estimation;
- the original region descriptors;
- the original depth, normal-angle, confidence, and union-find merge rules;
- `legacy` and `baseline_params` profiles.

The raw LASER-Geometry functions keep their original defaults for source-level
compatibility. The unified engine and CLI select `baseline_params` by default
so formal comparisons use the same Felzenszwalb parameters as the depth and
layer-atomic modes. `legacy` remains available for reproducing historical
geometry runs.

### `layer_atomic`

The existing `segment_point_map_layer_atomic(...)` and `merge_layer_atoms(...)`
remain unchanged. The mode continues to:

- reuse depth segmentation's initial atoms and coarse layers;
- calculate local atom scales and 3D boundary gaps;
- apply the weak coarse-layer prior;
- preserve full pixel coverage, deterministic compact labels, invalid-boundary
  handling, and global-scale invariance.

## Felzenszwalb Parameter Contract

Formal comparisons use this fixed triplet in all three modes:

| Parameter | Value |
| --- | ---: |
| `scale` | 300 |
| `sigma` | 1.1 |
| `min_size` | 500 |

The source implementations already provide this contract for depth,
layer-atomic, and geometry's `baseline_params` profile. The router passes the
values without transformation. Geometry's `legacy` profile retains its
historical `200 / 1.0 / 300` values and is never selected implicitly.

Parameter equality does not make the algorithms identical: depth and
layer-atomic start Felzenszwalb from scalar depth, while geometry starts from
normalized depth plus normals.

## Interfaces and Routing

Method-specific entry points stay intact:

```python
segment_depth_felzenszwalb_rag(depth_map, ...)
segment_geometry_felzenszwalb_rag(depth_map, point_map=..., ...)
segment_point_map_layer_atomic(point_map, ...)
```

`make_sp_graph(...)` becomes the shared integration boundary. It accepts point
maps and a `segment_mode`, selects one entry point, then sends the resulting
labels to the existing `match_segmentation_seq(...)` exactly once.

The router is responsible only for:

- validating the mode;
- extracting scalar depth when required;
- forwarding confidence and common Felzenszwalb parameters unchanged;
- forwarding normal/profile arguments only to geometry;
- invoking the existing batched image wrapper;
- building the shared temporal segment graph from returned labels.

The router does not contain segmentation formulas or region-merging logic.

## Engine and CLI Configuration

`StreamingWindowEngine` stores:

- `segment_mode`, default `depth`;
- `normal_method`, default `cross`;
- `geometry_seg_profile`, default `baseline_params`;
- the fixed Felzenszwalb triplet.

`StreamingWindowEngineLC` forwards the same configuration to its parent and
uses the same graph router. It must not maintain an independent mode switch.

Both demos expose:

```text
--segment_mode depth|geometry|layer_atomic
--normal_method cross|sobel
--geometry_seg_profile baseline_params|legacy
```

Existing commands without `--segment_mode` continue to select the original
depth baseline. A non-depth mode requires `--depth_refine`; otherwise startup
fails instead of silently running a method that has no effect.

At startup, the effective segmentation mode, geometry options when applicable,
and Felzenszwalb triplet are printed. This gives future metric runs an explicit
configuration record without introducing a metric system in this change.

## Error Handling

- Reject an unknown `segment_mode` before inference begins.
- Reject an unknown geometry profile or normal method when geometry is used.
- Reject `geometry` or `layer_atomic` when depth refinement is disabled.
- Preserve existing point-map shape validation in layer-atomic mode.
- Validate geometry point maps as `(H, W, 3)` per frame and require intrinsics
  only when a point map is absent.
- Preserve existing invalid-value behavior in all pinned implementations.
- Do not silently substitute depth mode after a method-specific error.

## Testing and Verification

### Characterization tests

- Keep every existing layer-atomic test unchanged.
- Verify the exposed depth stages reproduce the original depth wrapper output.
- Port LASER-Geometry's core geometry segmentation tests from `340c599`.
- Cover geometry `cross` and `sobel`, `legacy` and `baseline_params`, batched
  auxiliary-input selection, region merging, and compact labels.

### Routing tests

- Verify each mode invokes exactly its pinned entry point.
- Verify point-map/depth selection and confidence forwarding.
- Verify the formal comparison triplet reaches all three methods unchanged.
- Verify geometry-only arguments do not enter depth or layer-atomic functions.
- Verify returned labels enter the same graph builder.

### Integration tests

- Verify normal streaming and loop-closure streaming accept all three modes.
- Verify both CLI parsers expose the same choices and defaults.
- Verify non-depth modes require depth refinement.
- Run deterministic CPU segmentation smoke tests for all three modes.

### Completion gate

- Rebuild both Cython extensions.
- Run the complete pytest suite from a clean branch state.
- Confirm all existing layer-atomic tests still pass without editing their
  expected behavior.
- Review the complete diff against `feature/layer-atomic-geometry` and confirm
  that algorithm changes are confined to the ported geometry implementation;
  depth and layer-atomic algorithms must have no behavioral edits.
- If compatible CUDA hardware and local weights are available, run a short
  end-to-end sequence in each mode. Otherwise record this as an external GPU
  validation item rather than claiming it ran locally.

## Success Criteria

The branch is ready when one checkout can select `depth`, `geometry`, or
`layer_atomic` in both normal and loop-closure streaming, all three share the
same downstream LASER pipeline, formal-comparison Felzenszwalb parameters are
`300 / 1.1 / 500`, pinned algorithm behavior is covered by tests, and no metric
implementation has been added.
