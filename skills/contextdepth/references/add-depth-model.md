# Integrate a Depth Model

The goal is to deliver a callable `baselines/xxx.py` that exposes a `Baseline`
class. The user can pass `./baselines/xxx.py:Baseline` directly to the public
pipeline. No changes are needed to the model registry, the pipeline, the CLI,
or the paper evaluation configuration. Unless the user asks otherwise, use the
upstream installation and paths the user provides for dependencies and weights.

## Confirm the Model's Actual Capabilities First

Read the official loading and `infer` implementation for the version the user
specifies, then read this repository's
[DepthBaseline](../../../contextdepth/baseline.py),
[input and output types](../../../contextdepth/types.py), and
[three-stage interface description](../../../docs/baselines.md#constructor-and-three-stage-interface).

Establish the following facts, which affect the adapter:

- Whether the upstream already handles resize, normalize, pad, and size
  restoration. Prefer calling the complete official interface to avoid
  duplicating preprocessing.
- Whether the output is metric camera-Z, ray distance, disparity, or
  scale-ambiguous depth. Convert only with a well-founded formula. If the true
  metric scale cannot be recovered, state clearly that the model does not yet
  satisfy the interface. Do not align with the query GT or invent a scale
  factor.
- Whether the camera input actually accepts a full K, a focal length, an FOV,
  or rays. Use the officially supported conversion, and describe in the
  delivery notes any camera degrees of freedom that cannot be preserved. If the
  model does not support a known camera, declare
  `allowed_settings = ("unknown",)`. Do not silently ignore a K that is passed
  in.
- Whether the camera output is a genuine estimate of focal length, K, or rays.
  A model with no camera output may return `camera=None`, with a note that it
  cannot be used for unknown-camera reference selection. Do not fabricate a
  camera to get past the checks.

## Implement in a Single File

Do not load weights or initialize the GPU at module import. In the constructor,
call the base class and load the model. `paths["source"]` and
`paths["weights"]` are resources, and `paths["options"]` holds the model
options. `name` is only a result label and should not control model branching.
Use `eval()` and the inference precision the upstream requires.

The following is an interface skeleton. Each stage must be replaced with the
verified official implementation before delivery:

```python
from contextdepth.baseline import DepthBaseline
from contextdepth.types import Prediction


class Baseline(DepthBaseline):
    supports_batch = False
    supports_setting_reuse = False

    def __init__(self, name, paths, device):
        super().__init__(name, paths, device)
        # Load the official model here and store it as self.model.

    def preprocess(self, requests):
        # requests is a compatible group of InferenceInput; by default it is called per image.
        # Keep the original image size, the optional K, and what official preprocessing needs.
        raise NotImplementedError

    def infer_model(self, prepared):
        # Call the official inference interface. Do not compute evaluation metrics or select candidates here.
        raise NotImplementedError

    def postprocess(self, raw, prepared):
        # Return list[Prediction] in the original order, each covering its complete input RGB.
        raise NotImplementedError
```

For a normal request, `request.rgb` is `uint8`, HWC, RGB. `request.camera` is a
3×3 K in the pixel coordinates of that complete RGB, or `None` for an unknown
camera. Map K to the image grid the model actually processes, and map it back
to the complete input RGB on return. Do not write dataset ROIs, historical
resolution budgets, query GT, evaluation masks, or candidate selection into a
new adapter. The historical protocol belongs in `evaluation/`.

Each `Prediction.depth` must be NumPy `float32` with the input's shape
`(H, W)`, in meters, with the semantics of depth along the camera Z axis.
Official internal preprocessing may resize or pad, but the output must be
restored to the complete input. Do not return only the central region, and do
not add depth clipping or filling that has no basis in the upstream.

When there is a camera estimate, use `CameraEstimate`: `focal_px` is `[fx]` or
`[fx, fy]`, `intrinsics` is the pixel K on the input RGB, and non-pinhole
models may return `rays` of shape `(3,H,W)`. These representations must
describe the same complete input grid, and focal lengths should be finite and
positive. Do not treat a normalized K as a pixel K, and do not return the
selector-internal `camera_log_focal`. See the
[CameraEstimate conventions](../../../docs/baselines.md#complete-prediction-and-camera-output)
for details.

Declare `supports_batch=True` only after confirming that a single official
forward pass can process mutually independent images. Check whether attention,
and the reductions used in scale and camera recovery, span across images.
Multi-view input is not the same as an independent batch. Extend
`batch_key(request)` where necessary, and postprocess per item. If the adapter
would only loop over single images, keep the default configuration.

Declare `supports_setting_reuse=True` only when the same weights and loading
parameters can switch safely between known and unknown camera. `configure()`
must not reload weights. If there is state that depends on the camera setting,
refresh that state and the metadata after calling the base class. Otherwise
keep the default, which is off, and let the pipeline manage separate instances.
If the adapter holds a GPU model, implement `close()` to release the weight
references it holds.

## Verify Through the Public Entry Point

Run from the repository root, replacing the resource paths with the user's
actual paths:

```python
from contextdepth import ContextDepthPipeline as Pipeline

with Pipeline(
    depth_model="./baselines/xxx.py:Baseline",
    depth_options={"source": "/path/to/upstream", "weights": "/path/to/checkpoint"},
    device="cuda:0",
) as pipeline:
    prediction = pipeline.estimate_depth("query.png")
    print(prediction.depth.shape, prediction.depth.dtype)
```

If the model supports a known camera, also verify argument passing and
coordinate transforms with a real `camera=K`. First use a lightweight CPU
stand-in to check the complete output size, input order, camera mapping, and
the failure path when K is missing. Then check the real model under the user's
runtime conditions. If only the stand-in was run, state clearly that the real
weights have not been verified. When declaring batch or setting reuse, add the
corresponding independence and state-switching checks, and do not claim that
batch results match single-image results value for value.

Deliver the file path, the shortest way to call it, the dependencies, and the
scope of what was verified. If the official model has capability gaps, say so
directly.

## A Request Users Can Copy to an AI

```text
Follow skills/contextdepth/references/add-depth-model.md and integrate the depth model I provide into ContextDepth.
Official source location or link: [fill in]. Version: [fill in]. Weights path: [fill in].
Add only baselines/xxx.py, expose Baseline, and use the model's own official infer API.
Return float32 metric camera-Z depth on the complete input RGB. Handle camera input and output
according to the model's actual capabilities, and do not fake an unsupported scale or camera.
No changes are needed to the registry, the pipeline, or the paper evaluation logic.
Give me a calling example for Pipeline(depth_model='./baselines/xxx.py:Baseline', ...),
and state which verifications were actually run.
```
