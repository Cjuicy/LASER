# Window-reference campaign: fresh clone and cloud runbook

This runbook is for a clean cloud workspace. It uses `/root/autodl-tmp` for
the clone, campaign output, and scratch work. The checkpoint and dataset are
external inputs; no account, host, password, SSH URL, or user-info is encoded
in these commands.

## Fresh clone

```bash
cd /root/autodl-tmp
git clone --recursive --branch codex/laser-paper-pointmap-eval \
  https://github.com/Cjuicy/LASER.git LASER-Window-Reference
cd /root/autodl-tmp/LASER-Window-Reference
```

Create or activate the supported Python 3.11 environment and execute the
checked-in bootstrap actions. Supply the PI3 checkpoint from its separately
managed location; bootstrap does not download KITTI, 7-Scenes, or NeuralRGBD.

```bash
conda run -n vggt python run_window_reference_campaign.py bootstrap \
  --execute \
  --repository /root/autodl-tmp/LASER-Window-Reference \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors
```

The declared KITTI data must be obtained through the official KITTI
registration and purpose declaration, then unpacked under the external data
root below. The campaign intentionally has no download command.

## Read-only plan and preflight

First inspect the complete formal plan. The protocol is fixed at 75-frame
windows with 30-frame overlap and `no_loop` reconstruction.

```bash
conda run -n vggt python run_window_reference_campaign.py plan \
  --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign
```

Preflight is read-only with respect to the data and weights. `--allow-no-gpu`
records a warning for an intentionally CPU-only check; a real campaign still
requires a CUDA-capable device and a bfloat16-capable selected GPU.

```bash
conda run -n vggt python run_window_reference_campaign.py preflight \
  --allow-no-gpu --preset pointcloud-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign

conda run -n vggt python run_window_reference_campaign.py preflight \
  --allow-no-gpu --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign
```

## Foreground smoke and GPU campaign

Run the short KITTI smoke preset first, then the point-cloud subset, and only
then the formal KITTI subset. A foreground command is useful for a first
validation because its stderr and exit code remain visible:

```bash
conda run -n vggt python run_window_reference_campaign.py run \
  --preset kitti-smoke \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign \
  --fail-fast
```

For a long run, use a named screen session. The run is serial and resumable;
the second invocation skips exact completed records and uses the existing
ordinary prediction cache when its identity matches.

```bash
screen -dmS window-reference-campaign
screen -S window-reference-campaign -X stuff $'cd /root/autodl-tmp/LASER-Window-Reference\n'
screen -S window-reference-campaign -X stuff $'conda run -n vggt python run_window_reference_campaign.py run --preset pointcloud-small --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors --output-root /root/autodl-tmp/window-reference-campaign --keep-going\n'
```

After that campaign succeeds, start the formal subset with the same output
root and then resume it with the explicit flag:

```bash
conda run -n vggt python run_window_reference_campaign.py run \
  --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign \
  --keep-going

conda run -n vggt python run_window_reference_campaign.py run \
  --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign \
  --resume --keep-going
```

The six-run matrix is canonical: depth, geometry, and atomic, each with
window-reference `off` and `on`. The compact records retain one shared
prediction key per scene; changing checkpoint, source revision, staged input,
or fixed protocol values creates a new identity rather than silently mixing
results.

## Summary and non-mutating status

Summarization validates the compact records against the requested plan and
writes the six summary files under the campaign's `summary/` directory. It
does not reconstruct, evaluate, stage, or delete data.

```bash
conda run -n vggt python run_window_reference_campaign.py summarize \
  --preset kitti-small \
  --data-root /root/autodl-tmp/LASER-Paper-Pointmap-Eval/data \
  --checkpoint /root/autodl-tmp/LASER-Paper-Pointmap-Eval/weights/PI3/model.safetensors \
  --output-root /root/autodl-tmp/window-reference-campaign
```

Use screen inspection and log tails as non-mutating status checks:

```bash
screen -ls
tail -n 80 /root/autodl-tmp/window-reference-campaign/window-reference-v1/runs/kitti/kitti-04/f000000-end-s1/depth__wr-off/attempts/0001/stdout.log
```

The run and summary exit codes are authoritative. `campaign.json` records
status, source provenance, resolved-config digest, runtime details, and a
redacted argument vector. A failed run is resumed only with `--resume` after
checking its compact failure row.

## Safe cleanup

Only remove the campaign-owned work area after its records and summaries are
backed up. The guard rejects the ownership root and paths outside it; do not
replace this with a recursive shell glob.

```bash
conda run -n vggt python -c \
  'from experiments.window_reference_campaign.staging import guarded_remove; guarded_remove("/root/autodl-tmp/window-reference-campaign/window-reference-v1/work", "/root/autodl-tmp/window-reference-campaign/window-reference-v1")'
```

This command does not remove the external checkpoint or dataset. It also does
not remove the campaign's compact records, metadata, or summaries.

## Metric and truth boundaries

The KITTI trajectory rows named `internal_ate_*` and `internal_rpe_*` are the
repository's internal Sim(3) ATE/RPE measurements. They are not official KITTI
devkit metrics and must not be presented as a leaderboard submission.
Historical corrected KITTI rows are contextual references only; they are not
pooled into a fresh campaign summary. No real GPU campaign, CUDA-memory result,
or empirical quality claim is made until the commands above have actually run
on the declared data, checkpoint, and GPU.

## Residual risks

* 75-frame refinement CPU cost still needs measurement.
* `thin_geometry` is a single partial window.
* Each scene's first new run must regenerate its missing ordinary cache.
* Historical corrected KITTI rows are context only.
* A no-card instance cannot validate PI3, CUDA memory, or empirical quality.
