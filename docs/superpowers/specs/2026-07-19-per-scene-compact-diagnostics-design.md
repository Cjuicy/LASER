# Per-scene compact segmentation diagnostics design

Date: 2026-07-19  
Base branch: `codex/segmentation-diagnostics`  
Target branch: `codex/per-scene-compact-diagnostics`

## Context

The existing diagnostic workflow is strict and global. Its selector reserves
Recovery/Guard event-control pairs and two examples from each diagnostic
family before filling the remaining interval budget. This is appropriate for
one combined KITTI 00-10 run, but a one-sequence invocation forces every
available family into that sequence. For KITTI 00 with window 75 and overlap
30, the strict selector needs 11 intervals. The conservative result budget is
115.24 GiB at 11 intervals and 170.90 GiB at 16 intervals.

The cloud volume has 171 GiB free. It cannot retain full dense Pass 2 outputs
for thirteen independent strict runs. The two TUM sequences also exist only in
their original TUM layout and are absent from the prepared KITTI-style
diagnostic root.

## Goals

1. Run exactly one scene to completion before starting the next.
2. Preserve full-sequence ATE/RPE and scalar Pass 1 diagnostics.
3. Generate an offline visual report with one representative anomaly and one
   matched control per scene when both exist.
4. Keep strict global behavior unchanged by default.
5. After validating a completed report, remove regenerable dense Pass 2
   intermediates while preserving the report, cases, metrics, provenance and
   Pass 1 results.
6. Prepare the two TUM RGB sequences deterministically in the layout expected
   by the existing runner.
7. Provide a resumable `screen` workflow that is unaffected by SSH disconnects.

## Non-goals

- No loop closure, parameter sweep or change to the three official diagnostic
  profiles.
- No change to ATE/RPE evaluation or Sim(3) alignment.
- No replacement of the existing strict global selector.
- No deletion of source datasets, checkpoints, Pass 1 trajectories, reports or
  rendered cases.
- No claim that a two-case report provides global diagnostic-family diversity.

## Command-line interface

The diagnostic runner gains:

```text
--selection-policy strict-global|per-sequence-compact
```

The default is `strict-global`, preserving all current behavior and manifests.

`per-sequence-compact` requires exactly one requested sequence and uses
`--max-selected` as a hard per-scene case cap. The supported production
command uses `--max-selected 2`, `--window-size 75` and `--overlap 30`.
The selection policy is included in the experiment contract and config hash,
so strict and compact outputs cannot be resumed into one another.

A separate compaction command is added:

```text
python scripts/compact_segmentation_diagnostics.py \
  --output-dir <completed-scene-output> \
  --confirm
```

Without `--confirm`, it prints the deletion plan and changes nothing.

## Compact per-sequence selection

Candidate generation, scoring, context expansion and interval merging remain
unchanged. Only the reservation policy changes.

For one scene:

1. Select the highest-ranked non-control candidate, if one exists.
2. Select the highest-ranked matched control that is not already represented,
   if one exists.
3. Fill any remaining slots by existing weighted rank without duplicates.
4. Never exceed `max_selected`.
5. Do not raise the global mandatory-family-diversity error.

Coverage metadata continues to report every reason accurately:

- selected evidence: `available=true, selected=true`;
- available but excluded by the compact cap:
  `reason=selection_limit_excluded`;
- no qualifying evidence: `reason=no_qualifying_window`.

The report must label compact selection explicitly and must not describe
unselected families as unavailable when qualifying candidates existed.

With two selected intervals, KITTI 00 has the already observed conservative
upper bound of about 15.04 GiB during execution instead of more than 115 GiB.
Full-sequence ATE/RPE remains unchanged because it is produced during Pass 1.

## Safe post-report compaction

Compaction is allowed only when all of the following hold:

- `manifest.json` exists and has `status=complete`;
- the diagnostic lock is absent;
- `report/index.html` and `report/metrics.csv` exist;
- every selected case passes the existing case-artifact validator;
- the output has not already been compacted with a conflicting inventory.

The command first writes an atomic pending inventory containing every proposed
path, byte size and checksum. It then removes only the allowlisted,
regenerable paths:

- `artifacts/<profile>/<sequence>/pass2/`;
- `trajectory/pass2/`;
- `checkpoints/pass2/`.

It preserves:

- `manifest.json`, `summary.json`, `selection_records.json`,
  `selected_intervals.json`;
- all Pass 1 artifacts and trajectories;
- `trajectory/regret/`;
- `cases/` and `report/`;
- logs stored outside the scene output.

After deletion it atomically writes `compaction.json` with the removed
inventory, reclaimed bytes, source run ID and completion time. The manifest
retains `status=complete` and records that dense Pass 2 data was compacted.
A repeated compaction command is idempotent. A `--report-only` rebuild must
continue to work from the preserved cases and metrics.

Deleted dense intermediates are not locally recoverable, but are reproducible
from the preserved experiment contract, source data and checkpoint. The
screen runner never invokes `--resume` on an already compacted completed
scene; it skips that scene.

## TUM preparation

A new preparation command accepts a TUM sequence directory containing
`rgb.txt`, `rgb/` and `groundtruth.txt`.

For each RGB timestamp, it selects the nearest ground-truth timestamp within a
default 0.02 second tolerance. Unmatched RGB frames are skipped. Each matched
frame is emitted deterministically as:

```text
prepared_data/all13_diagnostics/
  sequences/<sequence>/image_2/000000.png
  poses/<sequence>.txt
```

Images are relative symbolic links by default, with a copy option for
filesystems that do not support links. TUM translation and quaternion
`tx ty tz qx qy qz qw` are converted into a camera-to-world 3x4 pose row.
The command reports RGB count, matched count, skipped count and maximum
association error, and refuses fewer than two matched frames. Existing output
requires an explicit overwrite option.

The two production sequence IDs are:

- `rgbd_dataset_freiburg1_desk`;
- `rgbd_dataset_freiburg1_360`.

## Serial screen runner

A repository script runs these scene IDs in order:

`00 01 02 03 04 05 06 07 08 09 10
rgbd_dataset_freiburg1_desk rgbd_dataset_freiburg1_360`.

For each scene it:

1. validates images and poses;
2. skips a completed and compacted result;
3. performs a dry run in a new output;
4. resumes that manifest for the real run;
5. verifies `manifest.status=complete` and required report files;
6. runs confirmed compaction;
7. verifies the preserved report again;
8. proceeds to the next scene only after success.

Any failure stops the background script with a nonzero status while leaving
the `screen` session log and per-scene log intact. Shell `exit` statements
run only inside the background script, never in the user's interactive SSH
shell.

The launch interface is one command similar to:

```bash
screen -dmS laser-all13 bash -lc \
  'cd ~/autodl-tmp/LASER-segmentation-diagnostics &&
   scripts/run_all13_compact.sh'
```

Users reconnect with `screen -r laser-all13` and detach with `Ctrl-A D`.

## Error handling

- Missing TUM preparation is detected before launching inference.
- Existing manifests with a different selection policy or configuration fail
  fingerprint validation rather than being reused.
- Disk preflight runs for every scene.
- Compaction refuses incomplete or locked runs.
- Interrupted inference uses the existing resume mechanism.
- Interrupted compaction is recoverable by replaying its pending inventory.
- A completed compacted scene is never automatically expanded or rerun.

## Tests

1. Existing strict selection tests remain unchanged and passing.
2. Compact selection returns at most two intervals and prefers anomaly plus
   matched control.
3. Compact selection handles absent controls and missing diagnostic families
   without raising the strict diversity error.
4. Coverage differentiates unavailable evidence from cap-excluded evidence.
5. Config hashes differ between strict and compact policies.
6. TUM timestamp association, quaternion conversion, deterministic naming and
   overwrite refusal use synthetic fixtures.
7. Compaction dry-run changes nothing.
8. Compaction rejects incomplete and locked outputs.
9. Confirmed compaction deletes only allowlisted Pass 2 paths, records an
   inventory, remains idempotent and leaves the offline report usable.
10. A serial-runner smoke test proves that the next scene starts only after
    report verification and compaction succeed.

## Acceptance criteria

- KITTI 00 completes with `s1-w75-o30`,
  `per-sequence-compact` and two selected cases without the mandatory
  diversity exception.
- Its report opens offline and contains full-sequence ATE/RPE plus preserved
  visual cases after compaction.
- Both TUM sequences pass existing dataset preflight.
- An SSH disconnect does not stop the screen job.
- Rerunning the launcher skips compacted completed scenes and resumes only the
  current incomplete scene.
- The strict global workflow remains backward compatible.
