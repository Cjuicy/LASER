# Pi3 Ordinary Prediction Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist Pi3 ordinary sliding-window predictions once and replay them across all segmentation and loop-method experiments without changing LASER's segmentation, anchor, Sim(3), optimizer, or aggregation mathematics.

**Architecture:** A strict Pi3 adapter is owned by one lazy model handle that distinguishes ordinary and joint forwards. Before downstream processing, `OrdinaryPredictionProvider` resolves explicit `WindowSpec` objects through a content-addressed, atomic v2 store containing only depth, confidence, poses, and one sequence reference intrinsic. Method-specific `WindowCache` objects remain isolated, while SALAD and loop-candidate joint A/B prediction remain uncached.

**Tech Stack:** Python 3.11+, PyTorch, safetensors, OmegaConf, NumPy 1.26.4, pytest, Cython, and standard-library `hashlib`, `json`, `tempfile`, `os`, and `fcntl`.

## Global Constraints

- Work only on `codex/pi3-ordinary-prediction-cache`, based on `origin/codex/modular-segmentation-loop-integration@3f8a3b8277f22ef1c6d820987cab1ff707b0741d`.
- Use `origin/codex/pi3x-ordinary-prediction-cache@b930a9c933457199ea5d2e0b15ccc2182353a1fa` only as a behavioral reference; do not merge or cherry-pick it wholesale.
- First-phase model names are exactly `pi3`; reject `pi3x` and all other names.
- Do not add Pi3X runtime files, Pi3X checkpoint handling, multimodal inputs, or Surface-SfM.
- Preserve `pi3/models/pi3.py`; load the existing Pi3 checkpoint with `strict=True`.
- Normalize every model call to `local_points`, `camera_poses`, `conf`, and `images`, each with an explicit model batch.
- Persist only ordinary depth `(N,H,W)`, confidence `(N,H,W)`, poses `(N,4,4)`, and one reference intrinsic `(3,3)`.
- Do not persist full XYZ, RGB images, global points, SALAD data, joint A/B predictions, alignments, loop constraints, optimizer state, or final results.
- Use the first ordinary window's full predicted local points to derive the fixed reference intrinsic; reconstruct every ordinary window from depth plus that intrinsic.
- Use explicit `WindowSpec(index, frame_start, frame_end)` end-to-end; never reverse-derive ranges with cache index arithmetic.
- Cache modes are exactly `auto`, `refresh`, `readonly`, and `off`.
- Use the v2 prediction-cache layout with content digests, atomic writes, entry locking, corruption quarantine, and partial-run resume.
- Hardware and Git metadata are provenance only and do not change the prediction key.
- Keep method-specific `WindowCache` isolated and keep traditional/corrected cross-loading rejection.
- Do not quantize cached tensors.
- Do not modify `inference_engine/segmentation/{depth,geometry,atomic}.py`, `inference_engine/anchor_propagation.py`, `loop_closure/methods/shared.py`, or `loop_closure/utils/sim3loop.py`.
- Keep the generated `inference_engine/utils/_segmentation_cy.cpp` and `inference_engine/utils/fast_seg.cpp` untracked.
- Follow strict RED-GREEN-REFACTOR: every production behavior starts with a focused failing test whose failure is observed.
- Do not claim real Pi3/CUDA/KITTI validation unless weights, CUDA, and data were actually available.
- After CPU verification and review, push `codex/pi3-ordinary-prediction-cache` to `origin` and verify the remote SHA.

## Execution Baseline

Before Task 1:

```bash
git status --short --branch
git merge-base --is-ancestor \
  origin/codex/modular-segmentation-loop-integration HEAD
python setup.py build_ext --inplace
python -m pytest -q
```

Expected:

```text
branch: codex/pi3-ordinary-prediction-cache
baseline ancestor check: exit 0
tests: 160 passed, 1 warning
untracked: only the two generated Cython .cpp files
```

If an unrelated baseline test fails, stop and record it before changing production code.

## Final File Map

Create:

```text
inference_engine/models/__init__.py
inference_engine/models/adapters.py
inference_engine/models/lazy.py
inference_engine/models/loader.py
inference_engine/prediction_cache/__init__.py
inference_engine/prediction_cache/types.py
inference_engine/prediction_cache/fingerprint.py
inference_engine/prediction_cache/store.py
inference_engine/prediction_cache/provider.py
tests/test_model_adapters.py
tests/test_model_loader.py
tests/test_prediction_types.py
tests/test_prediction_fingerprint.py
tests/test_prediction_store.py
tests/test_prediction_provider.py
tests/test_prediction_cache_matrix.py
docs/pi3-prediction-cache-validation.md
```

Modify:

```text
pipeline/config.py
pipeline/preflight.py
pipeline/runner.py
pipeline/diagnostics.py
configs/pipeline/default.yaml
configs/pipeline/test.yaml
inference_engine/streaming_window_engine.py
loop_closure/constraint_estimation.py
loop_closure/methods/base.py
loop_closure/methods/traditional.py
loop_closure/methods/corrected.py
scripts/verify_pipeline_matrix.py
eval_launch.py
utils/interfaces.py
README.md
docs/pipeline-configuration.md
tests/test_pipeline_config.py
tests/test_pipeline_runner.py
tests/test_pipeline_matrix.py
tests/test_loop_constraint_estimation.py
tests/test_loop_method_contracts.py
tests/test_traditional_loop_method.py
tests/test_corrected_loop_method.py
tests/test_eval_launch_lc.py
```

---

### Task 1: Strict Pi3 and Prediction-Cache Configuration

**Files:**

- Modify: `pipeline/config.py`
- Modify: `pipeline/preflight.py`
- Modify: `configs/pipeline/default.yaml`
- Modify: `configs/pipeline/test.yaml`
- Modify: `tests/test_pipeline_config.py`
- Modify: `tests/test_pipeline_runner.py`

**Interfaces:**

- Consumes: existing `PipelineConfig`, `_normalize_enum_values`, `_validate_config`, and `validate_preflight`.
- Produces:

```python
class ModelName(str, Enum):
    PI3 = "pi3"


class PredictionCacheMode(str, Enum):
    AUTO = "auto"
    REFRESH = "refresh"
    READONLY = "readonly"
    OFF = "off"


@dataclass(frozen=True)
class ModelConfig:
    name: ModelName
    checkpoint: str
    inference_device: str
    process_device: str
    dtype: str


@dataclass(frozen=True)
class PredictionCacheConfig:
    root: str
    mode: PredictionCacheMode
```

- [ ] **Step 1: Write the failing configuration tests**

Add assertions to `tests/test_pipeline_config.py`:

```python
assert loaded.config.model.name is ModelName.PI3
assert (
    loaded.config.prediction_cache.mode
    is PredictionCacheMode.AUTO
)
assert loaded.config.prediction_cache.root == "inference_cache/predictions"

for mode in ("auto", "refresh", "readonly", "off"):
    loaded = load_pipeline_config(
        TEST_CONFIG,
        [f"prediction_cache.mode={mode}"],
    )
    assert loaded.config.prediction_cache.mode.value == mode

with pytest.raises(ValueError, match="model.name"):
    load_pipeline_config(TEST_CONFIG, ["model.name=pi3x"])

with pytest.raises(ValueError, match="prediction_cache.mode"):
    load_pipeline_config(TEST_CONFIG, ["prediction_cache.mode=warm"])
```

Extend the preflight-order test so an invalid model name fails before the injected model factory records a call.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_pipeline_config.py \
  tests/test_pipeline_runner.py
```

Expected: collection or config loading fails because the model/cache enums and fields do not exist.

- [ ] **Step 3: Implement the strict schema and YAML defaults**

Add the enums and dataclass fields, include both enum paths in `_normalize_enum_values`, and add:

```yaml
model:
  name: pi3
  checkpoint: weights/model.safetensors
  inference_device: cuda
  process_device: cpu
  dtype: bfloat16

prediction_cache:
  root: inference_cache/predictions
  mode: auto
```

Keep preflight's existing dtype, device, manifest, checkpoint, and conditional SALAD checks. No `multimodal` field is introduced.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_pipeline_config.py \
  tests/test_pipeline_runner.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add \
  pipeline/config.py pipeline/preflight.py \
  configs/pipeline/default.yaml configs/pipeline/test.yaml \
  tests/test_pipeline_config.py tests/test_pipeline_runner.py
git commit -m "feat: configure Pi3 prediction cache"
```

---

### Task 2: Validated Pi3 Adapter, Strict Loader, and Lazy Handle

**Files:**

- Create: `inference_engine/models/__init__.py`
- Create: `inference_engine/models/adapters.py`
- Create: `inference_engine/models/loader.py`
- Create: `inference_engine/models/lazy.py`
- Create: `tests/test_model_adapters.py`
- Create: `tests/test_model_loader.py`

**Interfaces:**

- Consumes: Task 1 `ModelConfig` and `ModelName.PI3`.
- Produces:

```python
MODEL_ADAPTER_CONTRACT_VERSION = 1
REQUIRED_PREDICTION_KEYS = (
    "local_points",
    "camera_poses",
    "conf",
    "images",
)


class Pi3Adapter(torch.nn.Module):
    model_name = ModelName.PI3

    def forward(
        self,
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]: ...


def build_model_adapter(config: ModelConfig) -> Pi3Adapter: ...


class ModelForwardKind(str, Enum):
    ORDINARY = "ordinary"
    JOINT = "joint"


@dataclass
class ModelExecutionStats:
    model_constructed: bool = False
    ordinary_forward_count: int = 0
    joint_forward_count: int = 0


class LazyModelHandle:
    def __init__(
        self,
        factory: Callable[[], torch.nn.Module],
        *,
        inference_device: str | torch.device,
        dtype: torch.dtype,
    ) -> None: ...

    def predict(
        self,
        images: torch.Tensor,
        *,
        kind: ModelForwardKind,
    ) -> dict[str, torch.Tensor]: ...


def build_model_handle(config: ModelConfig) -> LazyModelHandle: ...
```

- [ ] **Step 1: Write failing adapter behavior tests**

Use a tiny `torch.nn.Module` backend. Verify 4D input normalization, 5D preservation, exact output keys, and rejection of wrong ranks, missing keys, wrong shapes, non-tensors, NaN, and Inf:

```python
prediction = Pi3Adapter(backend)(torch.ones(2, 3, 4, 5))
assert tuple(prediction) == REQUIRED_PREDICTION_KEYS
assert prediction["local_points"].shape == (1, 2, 4, 5, 3)
assert prediction["camera_poses"].shape == (1, 2, 4, 4)
assert prediction["conf"].shape == (1, 2, 4, 5)
assert prediction["images"].shape == (1, 2, 3, 4, 5)
```

Before each malformed fixture, name the exact production mutation it catches in the test name, such as `test_adapter_rejects_confidence_with_wrong_spatial_shape`.

- [ ] **Step 2: Write failing strict loader and lazy-count tests**

Patch `_construct_pi3` and checkpoint readers. Assert:

```python
adapter = build_model_adapter(config)
backend.load_state_dict.assert_called_once_with(
    checkpoint,
    strict=True,
)
assert isinstance(adapter, Pi3Adapter)
assert adapter.training is False
```

Assert `.safetensors` uses `safetensors.torch.load_file(..., device="cpu")`; other suffixes use `torch.load(..., map_location="cpu", weights_only=False)`.

For the lazy handle:

```python
handle.predict(images, kind=ModelForwardKind.ORDINARY)
handle.predict(images, kind=ModelForwardKind.JOINT)
assert factory_calls == 1
assert handle.stats.model_constructed is True
assert handle.stats.ordinary_forward_count == 1
assert handle.stats.joint_forward_count == 1
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_model_adapters.py \
  tests/test_model_loader.py
```

Expected: collection fails because `inference_engine.models` does not exist.

- [ ] **Step 4: Implement the four-key adapter contract**

Normalize input:

```python
if images.ndim == 4:
    images = images.unsqueeze(0)
if images.ndim != 5 or images.shape[2] != 3:
    raise ValueError("model images must have shape (B,N,3,H,W)")
```

Call the Pi3 backend positionally, validate exact output shapes and finite values, and return a new four-key dictionary. Do not modify `pi3/models/pi3.py`.

- [ ] **Step 5: Implement strict loading and lazy execution**

`build_model_adapter` constructs existing `Pi3`, loads the state dict with `strict=True`, moves the adapter to `config.inference_device`, and calls `.eval()`.

`LazyModelHandle.predict` moves images only when forwarding, uses `torch.no_grad()`, uses autocast only for configured FP16/BF16, increments the selected counter, and rejects non-dictionary results.

- [ ] **Step 6: Run tests and verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_model_adapters.py \
  tests/test_model_loader.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add inference_engine/models tests/test_model_adapters.py tests/test_model_loader.py
git commit -m "feat: add validated lazy Pi3 adapter"
```

---

### Task 3: Canonical Window Schedule and Compact Artifact Types

**Files:**

- Create: `inference_engine/prediction_cache/__init__.py`
- Create: `inference_engine/prediction_cache/types.py`
- Create: `tests/test_prediction_types.py`
- Modify: `pipeline/runner.py`

**Interfaces:**

- Consumes: existing `expected_window_count` behavior.
- Produces:

```python
PREDICTION_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class WindowSpec:
    index: int
    frame_start: int
    frame_end: int

    @property
    def frame_count(self) -> int: ...


def build_window_specs(
    frame_count: int,
    window_size: int,
    overlap: int,
) -> tuple[WindowSpec, ...]: ...


def validate_window_specs(
    specs: Sequence[WindowSpec],
    *,
    frame_count: int,
    window_size: int,
    overlap: int,
) -> tuple[WindowSpec, ...]: ...


@dataclass(frozen=True)
class OrdinaryWindowArtifact:
    spec: WindowSpec
    depth: torch.Tensor
    confidence: torch.Tensor
    camera_poses: torch.Tensor


@dataclass(frozen=True)
class SequenceArtifact:
    reference_intrinsic: torch.Tensor
```

- [ ] **Step 1: Write failing schedule tests**

Use literal expectations:

```python
assert build_window_specs(15, 10, 5) == (
    WindowSpec(0, 0, 10),
    WindowSpec(1, 5, 15),
)
assert build_window_specs(12, 10, 5) == (
    WindowSpec(0, 0, 10),
    WindowSpec(1, 5, 12),
)
assert build_window_specs(6, 10, 5) == (
    WindowSpec(0, 0, 6),
)
```

Verify a trailing range containing only `overlap` frames is omitted. Reject booleans/non-integers, invalid ranges, noncanonical index/range sequences, and invalid window parameters.

- [ ] **Step 2: Write failing artifact validation tests**

Assert artifacts accept only CPU floating finite tensors with:

```text
depth/confidence: (spec.frame_count,H,W)
camera_poses:     (spec.frame_count,4,4)
intrinsic:        (3,3)
```

Use independent literal tensors. Cover shape mismatch, integer dtype, CUDA when available, NaN, and Inf.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_prediction_types.py
```

Expected: import fails because the prediction-cache types do not exist.

- [ ] **Step 4: Implement immutable types and one schedule source**

Implement `to_payload`/`from_payload` for `WindowSpec` and `OrdinaryWindowArtifact`. Make `expected_window_count` return:

```python
return len(build_window_specs(frame_count, window_size, overlap))
```

Do not clone tensors inside every dataclass constructor; clone/detach at the storage/provider boundary.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_prediction_types.py \
  tests/test_pipeline_runner.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add inference_engine/prediction_cache pipeline/runner.py tests/test_prediction_types.py
git commit -m "feat: define canonical prediction artifacts"
```

---

### Task 4: Content-Addressed Pi3 Prediction Fingerprint

**Files:**

- Create: `inference_engine/prediction_cache/fingerprint.py`
- Create: `tests/test_prediction_fingerprint.py`

**Interfaces:**

- Consumes: `ModelConfig`, `ImageManifest`, `WindowSpec`, `MODEL_ADAPTER_CONTRACT_VERSION`, and `PREDICTION_CACHE_SCHEMA_VERSION`.
- Produces:

```python
@dataclass(frozen=True)
class PredictionFingerprint:
    key: str
    checkpoint_sha256: str
    image_manifest_sha256: str
    runtime_source_sha256: str
    canonical_payload: Mapping[str, object]


def sha256_file(path: str | Path) -> str: ...
def digest_image_manifest(manifest: ImageManifest) -> str: ...
def build_prediction_fingerprint(
    *,
    model: ModelConfig,
    manifest: ImageManifest,
    image_shape: tuple[int, int, int, int],
    sample_stride: int,
    window_size: int,
    overlap: int,
    specs: Sequence[WindowSpec],
) -> PredictionFingerprint: ...
```

- [ ] **Step 1: Write failing fingerprint inclusion tests**

Create literal checkpoint/image bytes. Assert identical semantic inputs produce the same 64-character key. Assert the key changes for checkpoint bytes, image bytes/order, dtype, model name, adapter/schema version, runtime source bytes, preprocessed shape, stride, size/overlap, or ordered specs.

- [ ] **Step 2: Write failing exclusion tests**

Prove the builder cannot accept segmentation, anchor, loop, SALAD, optimizer, process-device, output, Git, or hardware fields. Use `dataclasses.replace` on `ModelConfig` to show inference-device changes do not affect the key while dtype changes do:

```python
assert fingerprint(
    replace(model, inference_device="cuda:1")
).key == original.key
assert fingerprint(
    replace(model, dtype="float32")
).key != original.key
```

Assert mtime-only changes do not alter file digests and `allow_nan=False` rejects nonfinite canonical values.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_prediction_fingerprint.py
```

Expected: import fails because `fingerprint.py` does not exist.

- [ ] **Step 4: Implement canonical length-delimited hashing**

Hash image contents in manifest order, not paths or mtimes. The fixed Pi3 runtime-source list is:

```text
inference_engine/models/adapters.py
inference_engine/models/loader.py
utils/load_fn.py
pi3/models/pi3.py
pi3/models/layers/attention.py
pi3/models/layers/block.py
pi3/models/layers/camera_head.py
pi3/models/layers/pos_embed.py
pi3/models/layers/transformer_head.py
pi3/utils/geometry.py
all Python files below pi3/models/dinov2, recursively
```

Sort and deduplicate these paths before hashing. Canonical JSON uses:

```python
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
```

The builder accepts only key-relevant arguments, preventing downstream experiment configuration from polluting the cache key.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_prediction_fingerprint.py \
  tests/test_pipeline_manifest.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add inference_engine/prediction_cache/fingerprint.py tests/test_prediction_fingerprint.py
git commit -m "feat: fingerprint Pi3 ordinary predictions"
```

---

### Task 5: Atomic, Resumable v2 Prediction Store

**Files:**

- Create: `inference_engine/prediction_cache/store.py`
- Modify: `inference_engine/prediction_cache/__init__.py`
- Create: `tests/test_prediction_store.py`

**Interfaces:**

- Consumes: Task 1 `PredictionCacheMode`, Task 3 artifact types, and Task 4 `PredictionFingerprint`.
- Produces:

```python
class PredictionCacheMissError(RuntimeError): ...
class PredictionCacheCorruptError(RuntimeError): ...


@dataclass
class PredictionStoreStats:
    ordinary_hits: int = 0
    ordinary_misses: int = 0
    corrupt_count: int = 0
    read_ms: float = 0.0
    write_ms: float = 0.0
    saved_window_count: int = 0
    stored_bytes: int = 0


class OrdinaryPredictionStore:
    entry_path: Path

    @contextmanager
    def entry_lock(self): ...

    def read_sequence(self) -> SequenceArtifact | None: ...
    def write_sequence(self, artifact: SequenceArtifact) -> None: ...
    def read_window(
        self,
        spec: WindowSpec,
    ) -> OrdinaryWindowArtifact | None: ...
    def write_window(
        self,
        artifact: OrdinaryWindowArtifact,
    ) -> None: ...
    def finalize(self) -> None: ...
```

- [ ] **Step 1: Write failing mode and layout tests**

Assert the exact layout:

```text
<root>/v2/<key>/
  manifest.json
  sequence.json
  complete.json
  windows/000000.pt
  invalid/
  locks/entry.lock
```

Verify:

```python
assert auto_store.read_window(spec) is None
assert auto_store.stats.ordinary_misses == 1

with pytest.raises(PredictionCacheMissError, match="frame range"):
    readonly_store.read_window(spec)

assert not off_store.entry_path.exists()
```

Also verify `refresh` ignores old artifacts, `readonly` creates nothing, valid warm reads count hits, incomplete entries resume, and `finalize()` refuses missing sequence/windows.

- [ ] **Step 2: Write failing corruption, digest, and lock tests**

Cover truncated `.pt`, wrong key/schema/spec/shape/dtype/digest, missing declared file, and invalid sequence intrinsic. Assert:

- `auto` quarantines corruption below `invalid/` and returns a miss;
- `readonly` raises without moving anything;
- an exception before replace leaves no visible final file;
- another process cannot concurrently acquire the same entry lock.

The expected digest must be derived from literal tensor bytes, not by calling the production digest helper.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_prediction_store.py
```

Expected: import fails because `store.py` does not exist.

- [ ] **Step 4: Implement atomic envelopes and validation**

Use same-directory temporary files, flush, `os.fsync`, reread/validate, and `Path.replace`. Use `fcntl.flock(..., LOCK_EX)` for `entry.lock`. Load windows with:

```python
torch.load(path, map_location="cpu", weights_only=False)
```

Validate every envelope's schema, key, spec, spatial geometry, tensor dtype/finite values, and canonical tensor digest before returning it.

- [ ] **Step 5: Implement resumability and quarantine**

Missing `complete.json` alone means resumable, not corrupt. In `auto`, snapshot corrupt files under a timestamped `invalid/` directory and rebuild only what is missing. In `refresh`, snapshot existing artifacts before rebuilding. Never recursively delete an entry.

- [ ] **Step 6: Run tests and verify GREEN**

Run:

```bash
python -m pytest -q tests/test_prediction_store.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add \
  inference_engine/prediction_cache/store.py \
  inference_engine/prediction_cache/__init__.py \
  tests/test_prediction_store.py
git commit -m "feat: persist Pi3 predictions atomically"
```

---

### Task 6: Ordinary Prediction Provider and Warm No-GPU Path

**Files:**

- Create: `inference_engine/prediction_cache/provider.py`
- Modify: `inference_engine/prediction_cache/__init__.py`
- Create: `tests/test_prediction_provider.py`

**Interfaces:**

- Consumes: `LazyModelHandle`, `ModelForwardKind.ORDINARY`, `OrdinaryPredictionStore`, artifact types, and existing depth/intrinsic helpers.
- Produces:

```python
class OrdinaryPredictionProvider(torch.nn.Module):
    def get(
        self,
        spec: WindowSpec,
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]: ...

    def forward(
        self,
        images: torch.Tensor,
        *,
        window_spec: WindowSpec,
    ) -> dict[str, torch.Tensor]: ...
```

- [ ] **Step 1: Write failing cold/warm/partial/off tests**

Use a counting fake adapter and literal two-window schedule. Assert:

```python
provider.get(specs[0], images[:10])
provider.get(specs[1], images[5:15])
assert handle.stats.ordinary_forward_count == 2

warm_handle = LazyModelHandle(
    failing_factory,
    inference_device="cpu",
    dtype=torch.float32,
)
warm_provider.get(specs[0], images[:10])
warm_provider.get(specs[1], images[5:15])
assert warm_handle.stats.model_constructed is False
assert warm_handle.stats.ordinary_forward_count == 0
```

For partial cache, first window hits and second miss causes exactly one forward. `off` forwards every call but returns the same canonical representation. `readonly` miss/corrupt raises while factory calls remain zero.

- [ ] **Step 2: Write failing geometry and isolation tests**

Assert:

- first window full points establish the intrinsic;
- every returned `local_points` equals `unproject_depth_to_local_points(depth, intrinsic)`;
- only `local_points[..., -1]` is stored;
- RGB is attached from the current call and never appears in stored payloads;
- returned tensors are independent clones;
- later-window first access fails when the reference intrinsic is absent;
- images must be finite `(N,3,H,W)` matching `WindowSpec.frame_count`.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_prediction_provider.py
```

Expected: import fails because the provider does not exist.

- [ ] **Step 4: Implement canonical miss and replay paths**

Algorithm:

```text
validate spec/images
load sequence intrinsic once
read compact artifact
if hit and intrinsic exists:
    reconstruct local points and attach current images
if intrinsic absent and spec.index != 0:
    fail
ordinary model forward
derive first intrinsic if absent and persist sequence
persist depth/confidence/poses
finalize after final expected window
reconstruct local points and attach current images
return fresh batched tensors
```

The provider owns the only ordinary model call. Both cold and warm paths return the adapter's four-key batched contract.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_prediction_provider.py \
  tests/test_model_adapters.py \
  tests/test_prediction_store.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add \
  inference_engine/prediction_cache/provider.py \
  inference_engine/prediction_cache/__init__.py \
  tests/test_prediction_provider.py
git commit -m "feat: replay Pi3 ordinary predictions"
```

---

### Task 7: Explicit Provider Integration and Method-Cache Provenance

**Files:**

- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `loop_closure/methods/base.py`
- Modify: `loop_closure/methods/traditional.py`
- Modify: `loop_closure/methods/corrected.py`
- Modify: `tests/test_loop_method_contracts.py`
- Modify: `tests/test_traditional_loop_method.py`
- Modify: `tests/test_corrected_loop_method.py`

**Interfaces:**

- Consumes: `WindowSpec`, `OrdinaryPredictionProvider`, `PredictionFingerprint`.
- Produces:

```python
@dataclass(frozen=True)
class WindowRequest:
    spec: WindowSpec
    images: torch.Tensor


class StreamingWindowEngine:
    def forward(
        self,
        sample: torch.Tensor,
        *,
        window_spec: WindowSpec,
    ) -> None: ...
```

`WindowCache` schema becomes version 2 and adds:

```python
prediction_key: str
model_name: ModelName
checkpoint_digest: str
```

- [ ] **Step 1: Write failing inference/worker contract tests**

For traditional and corrected, assert:

- the inference worker calls `provider.get(spec, images)`;
- registration receives the same spec;
- the short final window preserves exact `[frame_start, frame_end)`;
- duplicate/out-of-order indices fail;
- segmentation receives canonical local points, confidence, and current RGB;
- workers do not call `estimate_pseudo_depth_and_intrinsics` or `unproject_depth_to_local_points`;
- existing distinct traditional/corrected Sim(3) and anchor behavior remains.

- [ ] **Step 2: Write failing cache-provenance tests**

Round-trip a v2 payload:

```python
restored = WindowCache.from_payload(
    payload,
    expected_method=LoopMethod.TRADITIONAL,
    expected_prediction_key="abc123",
    expected_model_name=ModelName.PI3,
    expected_checkpoint_digest="def456",
)
```

Reject schema 1, wrong key/model/checkpoint, malformed provenance, and traditional/corrected cross-loading.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_loop_method_contracts.py \
  tests/test_traditional_loop_method.py \
  tests/test_corrected_loop_method.py
```

Expected: failures show the queue still carries tensors and `WindowCache` lacks provenance.

- [ ] **Step 4: Refactor the two-thread pipeline**

Keep:

```text
WindowRequest queue
  -> provider worker
  -> registration queue carrying WindowSpec
  -> method worker
  -> method-specific WindowCache
```

Remove autocast/model invocation from `_model_inference_worker`; time `provider.get`. Remove worker-local reference intrinsic estimation and depth re-unprojection. Use only `spec.index`, `spec.frame_start`, and `spec.frame_end` for cache identity/range.

- [ ] **Step 5: Implement v2 method-cache provenance**

Include all three provenance fields in payloads and require exact expected values during load. Preserve the method tag and every existing loop-state field.

- [ ] **Step 6: Run focused and segmentation regression tests**

Run:

```bash
python -m pytest -q \
  tests/test_loop_method_contracts.py \
  tests/test_traditional_loop_method.py \
  tests/test_corrected_loop_method.py \
  tests/test_anchor_propagation_contract.py \
  tests/test_segmentation_strategies.py \
  tests/test_registration_confidence.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add \
  inference_engine/streaming_window_engine.py \
  loop_closure/methods/base.py \
  loop_closure/methods/traditional.py \
  loop_closure/methods/corrected.py \
  tests/test_loop_method_contracts.py \
  tests/test_traditional_loop_method.py \
  tests/test_corrected_loop_method.py
git commit -m "refactor: feed cached predictions into loop methods"
```

---

### Task 8: Runner, Uncached Joint Inference, Diagnostics, and Legacy Entry

**Files:**

- Modify: `pipeline/runner.py`
- Modify: `pipeline/diagnostics.py`
- Modify: `loop_closure/constraint_estimation.py`
- Modify: `eval_launch.py`
- Modify: `utils/interfaces.py`
- Modify: `tests/test_pipeline_runner.py`
- Modify: `tests/test_loop_constraint_estimation.py`
- Modify: `tests/test_eval_launch_lc.py`

**Interfaces:**

- Consumes: all prior task interfaces.
- Produces:

```python
class JointAlignmentEstimator:
    def __init__(
        self,
        *,
        model: LazyModelHandle,
        images: torch.Tensor,
        manifest: ImageManifest,
        chunk_size: int,
        confidence_keep_ratio: float,
    ) -> None: ...
```

It calls:

```python
model.predict(joint_images, kind=ModelForwardKind.JOINT)
```

Diagnostics:

```text
model_name
checkpoint_digest
ordinary_prediction_key
prediction_cache_mode
model_constructed
ordinary_hits
ordinary_misses
ordinary_forward_count
joint_forward_count
corrupt_count
prediction_cache_read_ms
prediction_cache_write_ms
saved_window_count
stored_bytes
```

- [ ] **Step 1: Write failing runner lifecycle tests**

Use injected fakes and assert:

```text
manifest/preflight
CPU image loading
WindowSpec construction
fingerprint/store
lazy handle/provider
ordinary windows under entry lock
SALAD unchanged
joint estimator only when candidates exist
constraints/optimization/aggregation
diagnostics
```

Verify a full ordinary hit with loop disabled or no candidates never constructs Pi3. Verify a full ordinary hit with candidates reports ordinary forwards `0` and joint forwards `>0`. Verify the same handle reaches the joint estimator.

- [ ] **Step 2: Write failing generic joint-estimator tests**

Rename Pi3-specific test fakes and assert:

```python
estimator(
    cache_a,
    cache_b,
    LoopCandidate(frame_a=8, frame_b=1, similarity=0.9),
    keep_ratio=0.5,
)
assert handle.stats.ordinary_forward_count == 0
assert handle.stats.joint_forward_count == 1
```

Keep existing centered range, cached-A/joint-A and cached-B/joint-B registration, confidence mask, candidate `ValueError` isolation, and model/CUDA error propagation tests.

- [ ] **Step 3: Write failing legacy-entry tests**

Assert `eval_launch.py` and `utils/interfaces.py` prepare the same `StreamingPipelineModel`/provider-backed engine and do not construct `Pi3` directly for the modular streaming path.

- [ ] **Step 4: Run tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_pipeline_runner.py \
  tests/test_loop_constraint_estimation.py \
  tests/test_eval_launch_lc.py
```

Expected: runner still constructs Pi3 eagerly and joint estimator is Pi3-specific.

- [ ] **Step 5: Wire runner and exact window execution**

Replace `load_pi3` injection with factories for model handle, fingerprint, store, provider, and generic joint estimator. `run_windows` slices CPU images from each `WindowSpec`, passes `window_spec=spec`, and loads method caches using expected provenance.

- [ ] **Step 6: Keep joint A/B inference uncached**

Rename `JointPi3AlignmentEstimator` to `JointAlignmentEstimator`, remove its direct device/autocast ownership, and call the shared handle with `kind=JOINT`. Do not add a joint-cache key or directory.

- [ ] **Step 7: Emit prediction diagnostics and update legacy wiring**

Collect handle/store stats into `run_summary.json` and `ReconstructionResult.summary`. Update the legacy streaming path to call one preparation interface; do not change evaluation formulas or image order.

- [ ] **Step 8: Run focused regression tests and verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_pipeline_runner.py \
  tests/test_loop_constraint_estimation.py \
  tests/test_eval_launch_lc.py \
  tests/test_traditional_loop_method.py \
  tests/test_corrected_loop_method.py
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add \
  pipeline/runner.py pipeline/diagnostics.py \
  loop_closure/constraint_estimation.py \
  eval_launch.py utils/interfaces.py \
  tests/test_pipeline_runner.py \
  tests/test_loop_constraint_estimation.py \
  tests/test_eval_launch_lc.py
git commit -m "feat: wire cached ordinary and uncached joint inference"
```

---

### Task 9: Experiment-Matrix Reuse and User Validation Guide

**Files:**

- Modify: `scripts/verify_pipeline_matrix.py`
- Modify: `tests/test_pipeline_matrix.py`
- Create: `tests/test_prediction_cache_matrix.py`
- Modify: `README.md`
- Modify: `docs/pipeline-configuration.md`
- Create: `docs/pi3-prediction-cache-validation.md`

**Interfaces:**

- Consumes: `PredictionCacheMode`, runner summaries, and the existing ten-entry matrix.
- Produces:

```python
def effective_matrix_cache_mode(
    requested: PredictionCacheMode,
    entry_index: int,
) -> PredictionCacheMode: ...
```

Matrix entry names:

```text
pi3_depth_traditional
pi3_geometry_traditional
pi3_atomic_conservative_corrected
```

- [ ] **Step 1: Write failing matrix naming/mode tests**

Assert:

- the Pi3 matrix contains the existing ten segmentation/loop combinations;
- every entry name starts with `pi3_`;
- prediction-cache root is identical across all entries;
- method cache/result paths remain unique;
- `auto`, `readonly`, and `off` remain unchanged;
- `refresh` maps entry zero to `refresh` and entries 1–9 to `readonly`.

- [ ] **Step 2: Write failing end-to-end cache-reuse test**

Use 15 tiny frames, two specs, one temporary prediction root, a counting fake adapter, and fake SALAD/solver boundaries. Assert:

```python
assert summaries[0]["ordinary_forward_count"] == 2
assert [
    item["ordinary_forward_count"]
    for item in summaries[1:]
] == [0] * 9
assert sum(
    item["ordinary_forward_count"]
    for item in summaries
) == 2
```

Run the matrix again with a failing factory and assert all ten ordinary counts are zero. Do not treat SALAD or joint calls as ordinary misses.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python -m pytest -q \
  tests/test_pipeline_matrix.py \
  tests/test_prediction_cache_matrix.py
```

Expected: matrix paths/modes are not cache-aware and the integration test fails.

- [ ] **Step 4: Implement matrix reuse**

Read model name from resolved config, keep one prediction root, and continue suffixing only method-specific cache/results. A requested refresh is allowed only for entry zero; later entries use readonly.

- [ ] **Step 5: Write the Pi3 validation guide**

Document:

- exact configuration and four modes;
- v2 layout and fingerprint inclusions/exclusions;
- what is and is not cached;
- corruption quarantine and readonly behavior;
- cold/warm diagnostics;
- 15-frame/two-window smoke;
- depth/geometry/atomic replay;
- traditional/corrected replay;
- uncached joint behavior;
- ten-entry `W,0,...,0` acceptance;
- full KITTI sequence measurements when CUDA/weights/data are available;
- clone and checkout commands for the pushed branch.

- [ ] **Step 6: Run matrix tests and documentation dry-run**

Run:

```bash
python -m pytest -q \
  tests/test_pipeline_matrix.py \
  tests/test_prediction_cache_matrix.py

python scripts/verify_pipeline_matrix.py \
  --config configs/pipeline/default.yaml \
  --set model.name=pi3 \
  --dry-run
```

Expected: tests pass; dry-run prints ten unique `pi3_` entries sharing one prediction root.

- [ ] **Step 7: Commit**

```bash
git add \
  scripts/verify_pipeline_matrix.py \
  tests/test_pipeline_matrix.py \
  tests/test_prediction_cache_matrix.py \
  README.md docs/pipeline-configuration.md \
  docs/pi3-prediction-cache-validation.md
git commit -m "docs: add Pi3 cache matrix validation"
```

---

### Task 10: Full Verification, Review, and Cloud Publication

**Files:**

- Modify only files required by verified failures or accepted review findings.

**Interfaces:**

- Consumes: the approved design, this plan, all implementation commits, and `origin`.
- Produces: a reviewed remote branch whose SHA matches local `HEAD`.

- [ ] **Step 1: Scan for forbidden architecture drift**

Run:

```bash
rg -n \
  "Pi3X|pi3x|JointPi3AlignmentEstimator|def _load_pi3|load_pi3" \
  pipeline inference_engine loop_closure eval_launch.py \
  scripts/verify_pipeline_matrix.py

rg -n \
  "joints/|detector/|salad.*cache|joint.*cache|alignment.*cache" \
  inference_engine pipeline loop_closure

rg -n \
  "cache_id \\*|window_size - self.overlap" \
  loop_closure/methods inference_engine/streaming_window_engine.py
```

Expected:

- no Pi3X implementation or old eager/Pi3-specific estimator interface;
- no SALAD/joint/alignment cache;
- no reverse-derived frame ranges in window workers.

- [ ] **Step 2: Run every focused feature test**

Run:

```bash
python -m pytest -q \
  tests/test_model_adapters.py \
  tests/test_model_loader.py \
  tests/test_prediction_types.py \
  tests/test_prediction_fingerprint.py \
  tests/test_prediction_store.py \
  tests/test_prediction_provider.py \
  tests/test_prediction_cache_matrix.py \
  tests/test_pipeline_config.py \
  tests/test_pipeline_runner.py \
  tests/test_pipeline_matrix.py \
  tests/test_loop_constraint_estimation.py \
  tests/test_loop_method_contracts.py \
  tests/test_traditional_loop_method.py \
  tests/test_corrected_loop_method.py \
  tests/test_eval_launch_lc.py
```

Expected: zero failures.

- [ ] **Step 3: Build extensions and run the full CPU suite**

Run:

```bash
python setup.py build_ext --inplace
python -m pytest -q
```

Expected: extensions build and the entire suite passes. Record the exact test count and warnings.

- [ ] **Step 4: Run matrix dry-run and repository checks**

Run:

```bash
python scripts/verify_pipeline_matrix.py \
  --config configs/pipeline/default.yaml \
  --set model.name=pi3 \
  --dry-run

git diff --check \
  origin/codex/modular-segmentation-loop-integration...HEAD
git status --short
git log --oneline --decorate \
  origin/codex/modular-segmentation-loop-integration..HEAD
```

Expected: ten valid Pi3 entries, no whitespace errors, only the two generated `.cpp` files untracked, and focused task commits.

- [ ] **Step 5: Apply `superpowers:verification-before-completion`**

Record:

```text
full CPU test result
focused test result
window count W in fake matrix
cold ordinary count W
remaining first-matrix counts 0 x 9
second-matrix counts 0 x 10
warm/no-candidate model_constructed=false
candidate-run ordinary/joint counts
corruption quarantine result
readonly no-construction result
GPU/Pi3-weight smoke: run or not run
KITTI validation: run or not run
```

- [ ] **Step 6: Apply `superpowers:requesting-code-review`**

Give the reviewer the approved spec, plan, baseline SHA, and final SHA. Require review of:

- Pi3 strict-load and existing runtime preservation;
- fingerprint inclusions/exclusions;
- atomicity, locking, digests, resumability, and quarantine;
- warm no-model path;
- ordinary/joint accounting;
- traditional/corrected behavior preservation;
- absence of Pi3X, Surface-SfM, SALAD cache, and joint cache.

- [ ] **Step 7: Fix accepted Critical/Important findings with TDD**

For every behavioral finding, first add a failing regression test, observe RED, implement the minimum fix, and rerun affected plus full tests. Commit with a message describing the actual fix.

- [ ] **Step 8: Push and verify the remote branch**

Run:

```bash
git push -u origin codex/pi3-ordinary-prediction-cache
git ls-remote \
  origin refs/heads/codex/pi3-ordinary-prediction-cache
git rev-parse HEAD
```

Expected: the remote SHA exactly equals local `HEAD`.

- [ ] **Step 9: Provide clone/test handoff**

Provide:

```bash
git clone \
  --branch codex/pi3-ordinary-prediction-cache \
  --single-branch \
  git@github.com:Cjuicy/LASER.git

cd LASER
python setup.py build_ext --inplace
python -m pytest -q
```

Also link the remote branch and `docs/pi3-prediction-cache-validation.md`, report CPU/GPU/KITTI validation truthfully, and include the final local/remote SHA.
