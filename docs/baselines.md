# Depth Model Adapters

Normal inference and the paper evaluation share the same set of depth adapters.
The difference is in how the caller prepares the inputs and uses the outputs.
The built-in implementations are in [`baselines/`](../baselines/). Good places
to start reading are the shorter [`DepthPro`](../baselines/depth_pro.py) or
[`MoGe2`](../baselines/moge.py), which takes a horizontal FOV.

| Purpose | Entry point | Inputs and outputs |
| --- | --- | --- |
| Normal use | [`ContextDepthPipeline`](../contextdepth/pipeline.py) | RGB and an optional pixel K. Returns depth on the complete input grid and an optional camera estimate. No experiment manifest is needed. |
| Paper evaluation | [`python -m evaluation`](../evaluation/cli.py) | Prepares ROIs, reference images, and candidates from the manifest, calls the same adapters, then projects onto the experiment grid, selects, and scores. |

An adapter handles model loading, preprocessing, the camera interface, and
output conversion. The public pipeline does not branch on model names.
Model-specific FOV, focal length, full K, or ray interfaces are encapsulated in
the respective adapters.

## Direct Inference on a Complete Image

```python
from contextdepth import ContextDepthPipeline

with ContextDepthPipeline(depth_model="moge2") as pipeline:
    prediction = pipeline.estimate_depth("query.png")

depth = prediction.depth       # float32, shape H × W of the original image, in meters
camera = prediction.camera     # Optional camera estimate from the model
```

For a known camera, call `pipeline.estimate_depth("query.png", camera=K)`, where
K is the 3×3 intrinsic matrix in the original image's pixel coordinates. The
input can also be a PIL image or a `uint8` RGB array. Calling only the depth
interface does not load the generation model or the reference retrieval model,
and needs no `experiments` configuration. Resources and defaults for normal use
come from [`models.yaml`](../models.yaml).

`pipeline(rgb)` returns predictions for the original image and for three
complete outpainted images, extending each side by 10% by default. Reference
selection is disabled by default. When enabled, it supports only DIODE, which
you download yourself. See the [usage guide](usage.md) for details.

## Load a Custom Adapter Directly

After adding `my_baseline.py`, you can call it through a file entry point
without changing the YAML registry or the pipeline:

```python
from contextdepth import ContextDepthPipeline

with ContextDepthPipeline(
    depth_model="./my_baseline.py:MyBaseline",
    depth_options={
        "source": "/path/to/upstream",
        "weights": "/path/to/checkpoint",
        "options": {"resolution": 512},
    },
) as pipeline:
    prediction = pipeline.estimate_depth("query.png")
```

`resolution` is an option defined by this example adapter itself, not a public
parameter. You can also pass `package.module:Class`, an implementation class,
or an initialized instance. A file path passed directly is resolved relative to
the current working directory. Absolute paths are recommended for resources.
Use a package entry point when the adapter needs relative imports within a
package.

The pipeline loads the model on the first call and reuses it afterward.
Instances constructed from a class or a string are managed by the pipeline. For
an instance passed in directly, the caller manages its lifetime, and the
instance keeps its existing batch limit. `depth_options` is the dictionary of
resources and configuration passed to the adapter. Model options go under its
`options` key.

## Constructor and Three-Stage Interface

Subclass [`DepthBaseline`](../contextdepth/baseline.py) and implement the three
methods below. Normal inference and evaluation both use this contract:

```python
from contextdepth.baseline import DepthBaseline
from contextdepth.types import CameraEstimate, InferenceInput, Prediction


class MyBaseline(DepthBaseline):
    supports_batch = False
    supports_setting_reuse = False

    def __init__(self, name, paths, device):
        super().__init__(name, paths, device)
        # Load weights from paths["weights"] onto self.device.

    def preprocess(self, requests: list[InferenceInput]):
        # Prepare a compatible group of RGB images, optional cameras, and the
        # model's own preprocessing information.
        raise NotImplementedError

    def infer_model(self, prepared):
        # Call the model once on the prepared batch of independent images.
        raise NotImplementedError

    def postprocess(self, raw, prepared) -> list[Prediction]:
        # Return float32 metric camera-Z depth on the complete RGB grid, in input order.
        # An optional camera should also be mapped back to that complete RGB's pixel coordinates.
        # Prediction(depth, CameraEstimate(focal_px=...))
        raise NotImplementedError
```

`name` is only a result label. `paths` contains resource paths, `setting`
(`known` or `unknown`), `options`, and `batch_size`. Load weights in the
constructor, and do not initialize the GPU at module import. `prepared` is the
adapter's own intermediate structure and `raw` is the native model output. Both
are defined by the adapter. Query GT depth and evaluation masks do not enter
this interface.

The base class exposes `infer(rgb, camera=None)`, `infer(request)`, and
`predict(requests) -> list[Prediction]`. It validates inputs, splits them into
compatible batches, restores the RNG, runs the three stages under
`torch.inference_mode()`, and checks the number, shape, and dtype of the
results. Results must be in the same order as the inputs, and a single model
instance must be called sequentially.

When you call an adapter's `infer` directly, whether you pass K must match the
instance's `setting`. The normal-use pipeline chooses the setting based on the
optional K. A model that supports only a known camera can declare
`allowed_settings = ("known",)`.

## Input RGB, K, and Experiment Geometry

[`InferenceInput`](../contextdepth/types.py) contains an identity, RGB, an
optional K, the camera setting, and an optional geometry and RNG state used for
experiments.

Normal calls need no explicit geometry. The automatically created geometry
covers the whole RGB, and the input K uses that RGB's pixel coordinates. An
outpainted image is also passed in as one complete image. The public pipeline
has already translated the principal point of the original image's K into the
outpainted image's coordinates, so the adapter must not add the generated
border's offset again.

Only evaluation callers need to provide the historical experiment geometry
explicitly:

- RGB is `uint8` HWC, with a size equal to `canvas_width/height`.
- `roi_x/y` and `roi_width/height` describe the visible query region inside the
  canvas. `fixed_w/h` describes the original evaluation grid. A historical ROI
  may have been resized, so its size can differ from the fixed grid.
- In this case K is on the **fixed grid**. An adapter can map it to the model
  input with [`camera_on_canvas`](../contextdepth/model_compat.py). The
  geometry is also used to preserve the experiment's token or resolution budget.
- The `direct` and `reference` arms use the borderless direct geometry, and
  candidates use their corresponding canvas. Unknown-camera requests must use
  `camera=None`.

These experiment hints do not change the full-image output convention.
`preprocess` is responsible for converting K into the form the model actually
accepts and for running the model's own preprocessing, such as resizing. A
model whose upstream accepts only FOV does not automatically gain full K
support this way. Document the limitation in the corresponding adapter.

## Complete Prediction and Camera Output

Return a [`Prediction`](../contextdepth/types.py):

| Field | Convention |
| --- | --- |
| `depth` | NumPy `float32` with shape `request.rgb.shape[:2]`. Metric camera-Z depth on the complete input RGB grid. |
| `camera` | Optional `CameraEstimate` describing the camera of the complete input RGB. When unknown-camera reference selection is enabled, every candidate must provide a usable camera estimate. |
| `metadata` | Dictionary of model observations and inference settings. |

Euclidean ray distance, disparity, or scale-ambiguous depth must first be
converted to metric camera-Z. An adapter does not crop the experiment ROI, does
not restore the fixed grid, does not align scale with the query GT, and does
not return the selector-specific `camera_log_focal`.

The common representations in `CameraEstimate` are:

| Field | Convention |
| --- | --- |
| `focal_px` | `[fx]` or `[fx, fy]`, finite and positive, in pixels of the complete input RGB. |
| `intrinsics` | Finite 3×3 K in the coordinates of the complete input RGB. |
| `rays` | `(3, H, W)` ray field over the complete input, for non-pinhole models. |
| `source_intrinsics`, `resize_factors` | Optional native K and per-stage `(sx, sy)`. Needed only to preserve the numerical order of the historical coordinate transforms. |

Prefer returning a focal length or K that is already mapped to the input RGB.
For the normal-use selector, [`contextdepth/camera.py`](../contextdepth/camera.py)
removes the generated border and computes the camera features. Cropping the
border does not rescale depth or change the pixel focal length.

All evaluation projection lives in
[`evaluation/geometry.py`](../evaluation/geometry.py). The ROI is cropped only
after the model returns, the fixed grid is restored with OpenCV bilinear
interpolation, and a one-axis or two-axis log focal is computed. Camera
representations are used in this order of preference: `focal_px`, `rays`,
native K plus resize records, `intrinsics`. The focal observation is
`log(fx / roi_width)` and optionally `log(fy / roi_height)`. Rays are fitted to
an equivalent camera within the experiment ROI. Candidates of the same query
must have the same observation dimension, and the query GT camera is not used
in place of an unknown-camera estimate.

## Batch Inference and Instance Reuse

The default is `batch_size=1`, which preserves per-image RNG behavior. Declare
`supports_batch = True` only once the adapter supports true multi-image
inference: collect compatible RGB images and per-image K on the CPU and move
them to the device in one transfer, have `infer_model` call native batch
inference once, then have `postprocess` split the output back into results in
the same order. Calling a single-image model in a loop does not count as native
batch support.

The base class groups requests by `batch_key(request)` and then splits each
group by `max_batch_size`. The default key contains the camera setting and the
RGB shape. If a model has budget, geometry, or precision constraints, add them
to the key. An actual batch can be smaller than the limit. An adapter that has
not declared batch support raises an error when configured with a batch size
greater than 1.

The normal-use pipeline sets the limit through `depth_batch_size=N` or the CLI
option `--depth-batch-size N`, and accepts multiple images through
`estimate_depths(images, cameras=...)`. Each batch uses the RNG state or seed
policy of its first request, and an adapter may also keep its own handling. A
batch larger than 1 can change floating-point computation and random number
consumption, so results are not guaranteed to match per-image results value for
value. The metadata records the actual batch size.

An adapter that can switch camera settings safely can declare
`supports_setting_reuse = True`. `configure(name, paths)` checks
`allowed_settings` and the loading configuration, and updates the setting, the
label, and numerical options such as TF32. If the adapter has additional state,
override this method, call the base class first, then update the adapter's own
state. Do not reload weights in `configure`, and do not switch settings during
inference.

[`model_load_config`](../contextdepth/config.py) excludes only `setting`. All
other resource, options, batch, and RNG configuration must be identical. By
default, a plugin adapter that has not declared support is not merged into a
shared load. When the normal-use pipeline switches settings, it rebuilds the
incompatible instances that it owns. An externally supplied instance that
cannot switch safely raises an error. The lifetime and setting-reuse rules do
not depend on model names.

## Add an Adapter to the Paper Evaluation

Configure `experiments` only when you need manifest-driven, multi-configuration
evaluation. The entry point is `python -m evaluation`, and the default
configuration is [`evaluation/models.yaml`](../evaluation/models.yaml). The
standalone configuration below uses already generated candidates, so it omits
outpainting:

```yaml
default_experiment: my_depth_unknown
depth_batch_size: 1
retrieval:
  checkpoint: /path/to/siglip2-so400m-patch16-384-vision
models:
  my_depth:
    implementation: ./my_baseline.py:MyBaseline
    source: /path/to/upstream
    weights: /path/to/checkpoint
experiments:
  my_depth_unknown:
    baseline: my_depth
    camera: unknown
    context_fraction: 0.25
    reference: my_depth_reference
  my_depth_reference:
    baseline: my_depth
    camera: known
    context_fraction: 0.25
    reference: my_depth_reference
    reference_only: true
```

```bash
python -m evaluation run --manifest data/manifest.json --config custom_models.yaml \
  --models my_depth_unknown --skip-generation --gpus 0 --output outputs/custom
```

The manifest must provide three compatible candidates for each query, along
with the reference RGB, metric depth, and pixel K. The reference configuration
must be `known`. `reference_only: true` excludes it from the query
configurations selected by `--models all`. A model that supports only a known
camera can define just the known query and reference configurations.

Relative file entry points and resource paths in the YAML are resolved relative
to the directory containing the YAML. Package entry points must be importable
in every worker environment. Experiment options override backend options of the
same name. A YAML that contains `experiments` is a complete configuration. A
file that contains only resource overrides inherits the built-in evaluation
settings. To generate candidates, add the outpainting configuration from
`evaluation/models.yaml`.

### Reusing the Query's Camera for the Candidates

An unknown-camera experiment can set `camera_source: query`. The camera is then
estimated once, on the unexpanded query, and reused when the outpainted
candidates are predicted, instead of estimating a new camera on every
candidate. The built-in configuration does this for UniDepth V2 and UniK3D.

```yaml
experiments:
  my_depth_unknown:
    {baseline: my_depth, camera: unknown, camera_source: query, context_fraction: 0.25, reference: my_depth_reference}
```

An adapter opts in with `supports_query_camera = True` and two small pieces:

- For an unknown-camera call on the query (`request.baseline` is true), return
  `Prediction(..., camera_state=state)`. `state` is whatever the adapter needs to
  apply this camera to a larger canvas later: an intrinsic matrix, a focal
  length, the parameters of a non-pinhole camera. It is opaque to the pipeline
  and is never saved.
- When `request.query_camera` is set, the request is a candidate and that value
  is the query's `state`. Predict depth with that camera, placed at the query's
  position on the canvas (`request.geometry.roi_x`, `roi_y`). Such requests
  always arrive one image per call.

The pipeline first predicts the query and the three candidates in the ordinary
way, then predicts the candidates again with the query's camera. The saved
candidate depths are the second ones. Unknown-camera selection still uses the
camera each candidate estimated in the first pass, so an adapter that supports
this option must still return a camera estimate for ordinary calls. The
normal-use `ContextDepthPipeline` does not use this option. See
[`baselines/unidepth.py`](../baselines/unidepth.py) for a pinhole example and
[`baselines/unik3d.py`](../baselines/unik3d.py) for a ray-field example.

### Evaluation Caching and Scheduling
The following behavior belongs to
[`evaluation/pipeline.py`](../evaluation/pipeline.py). The normal-use
`ContextDepthPipeline` does not create these experiment caches or execution
reports.

- The CLI option `--depth-batch-size` overrides the global configuration.
  Without a global configuration, the backend's `batch_size` can be used, and
  the final default is 1. `predict`, `select`, and `evaluate` must use the same
  setting. The batch configuration and the fixed input grouping are part of the
  cache identity. If part of the cache is missing, the whole corresponding
  window is recomputed.
- [`evaluation/cache.py`](../evaluation/cache.py) records the adapter, `_*.py`
  helpers in the same directory, model resources, the upstream source, and
  evaluation preprocessing dependencies. Declare additional helpers with the
  backend's `cache_dependencies: [./helper.py]`. Changing only the selection
  rule or the scoring code allows the depth arrays to be reused. Changing a
  computation dependency invalidates the cache.
- Workers merge settings that can be reused safely according to
  [`model_reuse_key`](../contextdepth/models.py), so that one instance
  processes the reference images and the queries of different camera settings
  in turn. Prediction caches are still saved separately for each experiment.
  With a single GPU, a single worker, and no cache hits for any task, the 14
  built-in query settings share 8 checkpoints. DA3 Nested and DA3 Metric are
  loaded separately, and UniDepth is loaded once more for its reference
  calibration, which uses a different option. `execution.json` records the grouping and
  the number of model instances that actually ran tasks. Multiple workers and
  cache hits affect the actual number of loads.

Batching reduces the number of forward passes, and instance reuse reduces the
number of weight loads. Neither gives a fixed speedup factor.

### Evaluation Protocol

The built-in evaluation is the crop protocol of the paper's Table 1: aligned
candidate canvases with 25% per side, used in full by every built-in experiment
(`context_fraction: 0.25`; a smaller value crops the candidates). This differs
from the normal-use interface, which generates 10% per side directly.
The adapters preserve the recorded preprocessing, interpolation, budgets,
numerical options, MoGe-3 refinement/full-K recovery, and DA3 camera handling.
Sharing an interface does not mean these behaviors have been replaced with
upstream defaults.

The default evaluation needs no external random state files. If the manifest
provides `rng_states` or `reference_rng_states`, the adapter restores the
random state for the corresponding input. For a `paper-roi` manifest, a custom
experiment can set `requires_rng: true` to require that queries provide a
state. See the [reproduction guide](reproduction.md) for the full data
preparation and run commands.

## Integrate a New Model with AI Assistance

You can have your own coding tool read the
[depth-model guide](../skills/contextdepth/references/add-depth-model.md) of the repository's skill,
add a single adapter that follows the interface, and verify its complete
output, camera, and batch behavior. For generation models, use the
[outpainting-model guide](../skills/contextdepth/references/add-outpainting-model.md).
Verify the adapter through the normal-use file entry point first, and add an
experiment configuration only when you need the paper evaluation.
