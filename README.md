<div align="center">
<h1>LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction</h1>
<a href="http://arxiv.org/abs/2512.13680"><img src="https://img.shields.io/badge/arXiv-2512.13680-b31b1b" alt="arXiv"></a>
<a href="https://neu-vi.github.io/LASER/"><img src="https://img.shields.io/badge/Project-Website-orange" alt="Project Page"></a>

[Tianye Ding<sup>1*</sup>](https://jerrygcding.github.io/),
[Yiming Xie<sup>1*</sup>](https://ymingxie.github.io/),
[Yiqing Liang<sup>2*</sup>](https://lynl7130.github.io/),
[Moitreya Chatterjee<sup>3</sup>](https://sites.google.com/site/metrosmiles/),
[Pedro Miraldo<sup>3</sup>](https://pmiraldo.github.io/),
[Huaizu Jiang<sup>1</sup>](https://jianghz.me/)<br>
<sup>1</sup> Northeastern University, <sup>2</sup> Independent Researcher, <sup>3</sup> Mitsubishi Electric Research Laboratories<br>
<sup>*</sup> Equal Contribution
</div>

## Updates

- **[2026-08-13]** Reconstruction and evaluation paths decoupled into typed artifacts.
- **[2026-03-12]** Loop-closure module released with robustness fix.
- **[2026-02-21]** Paper accepted by CVPR 2026.

## Abstract

LASER is a training-free framework that converts an offline reconstruction
model into a streaming system by aligning predictions across consecutive
temporal windows. Layer-wise scale alignment segments depth predictions,
computes per-layer scale factors, and propagates them across adjacent windows
and timestamps.

## Installation

```bash
git clone --recursive --branch codex/laser-paper-pointmap-eval https://github.com/Cjuicy/LASER.git
cd LASER
conda create -n laser-decoupled python=3.11 -y
conda activate laser-decoupled
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e viser
bash scripts/download_weights.sh
```

The ordinary PI3 model is expected at `weights/model.safetensors`. Traditional
and Corrected reconstruction also require `weights/dino_salad.ckpt` and
`weights/dinov2_vitb14_pretrain.pth`.

## Decoupled architecture

```text
PI3 prediction stream
  -> depth | geometry | atomic segmentation
  -> optional window-reference segmentation refinement (merge-only)
  -> original LASER AnchorPropagator
  -> no_loop | traditional | corrected reconstruction
  -> ReconstructionArtifact
       -> ATE evaluator (all reconstruction modes)
       -> point-cloud evaluator (no_loop only)
```

The evaluator boundary is strict: evaluators load artifact views and calculate
metrics; they never run PI3, segmentation, anchor propagation, registration, or
loop closure.

The original three reconstruction modes preserve their operation order, and
the second-global mode is an opt-in extension of Traditional:

- `no_loop`: incremental adjacent registration, segmentation, anchor
  propagation, and point-map assembly. No loop service is constructed.
- `traditional`: record all windows first, detect loops, estimate constraints,
  optimize, then apply deferred alignment and depth-scale correction.
- `corrected`: apply adjacent alignment and anchor correction online, detect
  loops after all windows, optimize sequential edges, then apply one final
  correction delta.
- `traditional_second_global`: run Traditional unchanged as Stage 1, rebuild
  residual adjacent and loop constraints from its corrected geometry, then
  apply one second global Sim(3) optimization delta.

## One reconstruction

Choose any segmentation, reconstruction mode, window size, and overlap:

```bash
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser.yaml \
  --set input.image_dir=/data/sequence/images \
  --set output.scene_name=my-sequence \
  --set segmentation.method=geometry \
  --set reconstruction.mode=traditional \
  --set window.size=20 \
  --set window.overlap=5
```

Use `configs/reconstruction/pi3_laser_no_loop.yaml` when loop checkpoints are
unavailable. Valid windows satisfy `window.size > window.overlap >= 1`; `10/5`,
`20/5`, and `20/10` are supported examples.

Each run writes:

```text
outputs/reconstruction/<scene>/<segmentation>-<mode>/
  manifest.json
  trajectory.pt
  pointmap.pt
  confidence.pt
```

### Traditional second-global optimization

Select `traditional_second_global` to preserve the original Traditional result
as Stage 1 and rebuild a second residual Sim(3) graph from the corrected Stage
1 window geometry:

```bash
python run_reconstruction.py \
  --config configs/reconstruction/pi3_laser.yaml \
  --set input.image_dir=/data/sequence/images \
  --set output.scene_name=my-sequence \
  --set segmentation.method=geometry \
  --set reconstruction.mode=traditional_second_global \
  --set window.size=20 \
  --set window.overlap=5
```

The result root is the Stage 2 artifact. The exact Traditional Stage 1
artifact is nested below it:

```text
outputs/reconstruction/my-sequence/geometry-traditional_second_global/
  manifest.json
  trajectory.pt
  pointmap.pt
  confidence.pt
  stage1/
    manifest.json
    trajectory.pt
    pointmap.pt
    confidence.pt
```

Both directories use the standard artifact contract, so evaluate Stage 1 at
`<result>/stage1` and Stage 2 at `<result>` with the existing ATE command. See
[`docs/traditional-second-global-cloud-validation.md`](docs/traditional-second-global-cloud-validation.md)
for a complete cloud setup, baseline equality check, and dual-ATE workflow.

Ordinary PI3 predictions are cached independently of segmentation,
reconstruction mode, anchor, loop, and evaluation settings.

## Artifact-only evaluation

ATE accepts artifacts from all three reconstruction modes:

```bash
python evaluate_ate.py \
  --artifact outputs/reconstruction/my-sequence/geometry-traditional \
  --config configs/evaluation/ate.yaml \
  --ground-truth /data/sequence/groundtruth.txt \
  --ground-truth-format tum \
  --output outputs/evaluation/ate/my-sequence
```

Point-cloud evaluation accepts only `no_loop` artifacts:

```bash
python evaluate_pointcloud.py \
  --artifact outputs/reconstruction/my-sequence/depth-no_loop \
  --config configs/evaluation/pointcloud.yaml \
  --ground-truth /data/sequence/groundtruth_pointmaps.npz \
  --dataset NeuralRGBD \
  --sequence my-sequence \
  --output outputs/evaluation/pointcloud/my-sequence
```

Point-cloud ground truth is an NPZ with `point_maps` shaped `(N,H,W,3)` and
`valid_mask` shaped `(N,H,W)`. A loop artifact is rejected before the Open3D
backend is constructed.

## Experiment matrices

- ATE: `depth|geometry|atomic × no_loop|traditional|corrected` = 9 entries.
- Point cloud: `depth|geometry|atomic × no_loop` = 3 entries.

Set dataset paths in the experiment YAML and run:

```bash
python run_experiment_matrix.py --config configs/experiments/ate_matrix.yaml
python run_experiment_matrix.py --config configs/experiments/pointcloud_matrix.yaml
```

The matrix runner shares ordinary prediction cache entries and reuses complete
reconstruction artifacts by content identity. Point-cloud matrix runs
explicitly use the LASER Table 4 `nearest` confidence quantile; ATE retains the
`higher` rule.

For complete setup, dry-run, output, and cloud verification commands, see
[docs/reconstruction-evaluation-cloud-validation.md](docs/reconstruction-evaluation-cloud-validation.md).

## Visualization

```bash
python viser/visualizer_monst3r.py --data viser_results/SEQ_NAME
```

## Citation

```bibtex
@article{ding2025laser,
  title={LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction},
  author={Ding, Tianye and Xie, Yiming and Liang, Yiqing and Chatterjee, Moitreya and Miraldo, Pedro and Jiang, Huaizu},
  year={2025}
}
```

## Acknowledgements

We thank the authors of [VGGT](https://github.com/facebookresearch/vggt),
[PI3](https://github.com/yyfz/Pi3),
[MonST3R](https://github.com/Junyi42/monst3r),
[CUT3R](https://github.com/CUT3R/CUT3R), and
[VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long).
