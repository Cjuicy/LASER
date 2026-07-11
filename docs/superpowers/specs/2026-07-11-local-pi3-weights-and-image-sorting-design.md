# Local Pi3 Weights and Image Sorting Design

## Goal

Make the demo run from the repository's Pi3 checkpoint without downloading model weights, and ensure numbered image frames enter the sliding-window engine in chronological order.

## Model loading

- `--model_ckpt` defaults to `weights/model.safetensors`.
- The command-line option remains available so callers can override the checkpoint.
- Model loading always uses the resolved local checkpoint path. The Hugging Face `from_pretrained` fallback is removed, so the demo never downloads Pi3 weights implicitly.
- If the checkpoint does not exist, model loading raises a clear `FileNotFoundError` containing the missing path.
- `.safetensors` checkpoints continue to use `safetensors.torch.load_file`; other explicitly supplied local checkpoint formats continue to use `torch.load`.

## Image ordering

- Filter the input directory to supported image extensions before sorting.
- Sort filenames with a natural, case-insensitive key so embedded digit groups compare numerically. For example, `frame1.jpg`, `frame2.jpg`, and `frame10.jpg` retain that order.
- Apply `sample_interval` only after sorting, ensuring sampling follows the actual frame sequence.
- Preserve the current supported extensions: `.png`, `.jpg`, and `.jpeg`, while accepting uppercase variants.

## Structure

Introduce a small natural-sort-key helper in `demo.py`. Keep image discovery in `run_dynamic_scene` and preserve the existing inference flow and the user's explanatory comments. No unrelated CUDA, timing, or model-engine refactoring is included.

## Error handling

- Missing local Pi3 checkpoint: fail immediately with a path-specific message and no network fallback.
- Existing argument parsing and filesystem errors remain unchanged.

## Tests

Add focused tests that verify:

1. the parser defaults `--model_ckpt` to `weights/model.safetensors`;
2. callers can override the default checkpoint path;
3. natural sorting orders numbered image filenames numerically;
4. image filtering and sorting happen before interval sampling;
5. a missing checkpoint produces `FileNotFoundError` before model construction or network access.

Tests should avoid loading the real multi-gigabyte model checkpoint.
