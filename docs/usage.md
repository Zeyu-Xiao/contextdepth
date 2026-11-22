# Using ContextDepth

Normal use goes through `ContextDepthPipeline`, which takes RGB and optional camera intrinsics as input. Dataset ROIs and metric computation use a separate entry point. See [Reproducing Table 1](reproduction.md).

## Inputs and Outputs

```python
from contextdepth import ContextDepthPipeline

with ContextDepthPipeline() as pipeline:
    result = pipeline("query.png")
    result.save("outputs/example")
```

RGB can be a path, a PIL image, or a NumPy `uint8` array of shape `(H, W, 3)` in RGB channel order. The model adapter handles the model's own preprocessing and maps the output back to the complete input grid.

| Return value | Contents |
| --- | --- |
| `result.direct` | `Prediction` for the original image |
| `result.candidates[i].outpainting` | The i-th outpainted image, with `rgb`, the original image's position `bbox`, and `seed` |
| `result.candidates[i].prediction` | `Prediction` for that complete outpainted image |
| `result.selected` | The selected prediction over the original image extent when DIODE is enabled. `None` by default |
| `result.reference_uid` | Identifier of the DIODE reference used. `None` by default |
| `result.selection` | Selection basis and candidate index. `None` by default |

Each `Prediction.depth` is a 2D `float32` array of camera-Z depth in meters. It has the same size as the corresponding input RGB. `Prediction.camera` contains whichever of the pixel intrinsics `intrinsics`, the pixel focal length `focal_px`, or the full-image rays `rays` the model can provide. A model or camera mode that provides no camera estimate may return `None`.

By default, no candidate is selected. With DIODE enabled, the selector compares depth only over the original image extent, removing the generated border and adjusting the camera coordinates accordingly. The complete outpainted images and complete depth maps remain in `candidates`. This step does not crop the content of the input RGB, and it does not apply the paper's dataset ROI.

## Camera Input

`camera=K` accepts a finite 3×3 intrinsic matrix with positive focal lengths, in coordinates of the input RGB's pixel grid:

```python
import numpy as np
from contextdepth import ContextDepthPipeline

K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
with ContextDepthPipeline() as pipeline:
    prediction = pipeline.estimate_depth("query.png", camera=K)
```

Use the image's actual K. Omitting it selects unknown-camera mode, where the camera output depends on the model's capabilities. A model that requires a known camera reports that it does not support this mode. After context is generated, the pipeline translates the principal point according to the original image's position in the canvas, and each adapter then converts K into the FOV, focal length, or camera object its upstream requires.

## Generation Parameters

```python
with ContextDepthPipeline(expansion=0.1, seeds=(0, 1, 2)) as pipeline:
    contexts = pipeline.generate("query.png")
```

`expansion` is the fraction by which each side is extended, relative to the corresponding dimension of the original image. The default is 10%. For example, a 640×480 input produces a 768×576 canvas. Pixel counts are rounded half up to integers, and the original image is not resized. The seeds must all be different from each other.

The Python interface also accepts a tuple ordered top, bottom, left, right, or a `top/bottom/left/right` dictionary:

```python
contexts = pipeline.generate(
    "query.png",
    expansion={"top": 0.1, "bottom": 0.2, "left": 0.1, "right": 0.1},
    seeds=(10, 20, 30),
)
```

`bbox=(x0, y0, x1, y1)` uses half-open coordinates: the start is inclusive and the end is exclusive. Every generation adapter must output the agreed canvas size, and the shared generation base class restores the original pixels inside the bbox. If OneReward needs to align to the model's size requirements, it pads only temporarily, on the right and bottom, and removes the padding before returning.

## Run Stages Separately and Reuse Outpaintings

`generate()` loads only the generation model, and `estimate_depth()` loads only the depth model. When you need the full workflow but GPU memory cannot hold both models at once:

```python
from contextdepth import ContextDepthPipeline

with ContextDepthPipeline(generation_options={"offload": "model"}) as pipeline:
    contexts = pipeline.generate("query.png")
    pipeline.release_generation()
    result = pipeline("query.png", outpaintings=contexts)
```

`outpaintings` must be a list of `Outpainting` objects, and each image must preserve the same complete input RGB. See the [outpainting-model guide](../skills/contextdepth/references/add-outpainting-model.md) for how to integrate an external generator.

Outpaintings generated from the command line can be reloaded and passed to any depth model:

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image
from contextdepth import ContextDepthPipeline, Outpainting

folder = Path("outputs/context")
records = json.loads((folder / "outpaintings.json").read_text())
contexts = [
    Outpainting(
        rgb=np.array(Image.open(folder / item["image"]).convert("RGB")),
        bbox=tuple(item["bbox"]), seed=item["seed"],
    )
    for item in records
]
with ContextDepthPipeline(depth_model="moge2") as pipeline:
    result = pipeline("query.png", outpaintings=contexts)
```

## Depth Models and Batching

The normal-use API takes model names. There is no need to choose a paper experiment name:

| `depth_model` | Camera mode |
| --- | --- |
| `moge2`, `moge3`, `unidepth`, `unik3d`, `depth_pro`, `depth_pro2` | Known or unknown |
| `da3nested` | Known or unknown, uses the Nested weights |
| `da3metric` | Requires a known camera, uses the Metric weights |

See [optional depth dependencies](installation.md#5-optional-other-built-in-depth-models) for installing the other models. Normal inference and evaluation call the same adapters. The context fractions and ROI budgets of the different experiments are configured only through the evaluation entry point.

Submit several independent RGB images at once:

```python
with ContextDepthPipeline(depth_model="moge2", depth_batch_size=3) as pipeline:
    predictions = pipeline.estimate_depths(["a.png", "b.png", "c.png"])
```

Use `cameras=[K_a, None, K_c]` to provide intrinsics per image. The pipeline groups images by camera mode, the adapter then forms batches by size and model requirements, and the input order is restored at the end. The default batch size is 1. Raising it only sets an upper limit on the number of images per model call. It does not guarantee a proportional speedup or floating-point results identical to per-image inference. DA3 places independent images along the batch dimension with a single view per item, and does not treat them as a multi-view sequence in which views reference each other.

A pipeline instance keeps its loaded depth model. Adapters that support switching camera modes reuse their weights. Custom adapters that do not support reuse are constructed again. `close()` or the end of the `with` block releases the models the pipeline created itself. A model object that was already constructed when passed in remains managed by the caller.

## DIODE Reference Bank

The reference bank is disabled by default. Enabling it requires setting both `reference_bank="diode"` and `diode_root`. See the [README](../README.md#optional-diode-reference-selection) for the download and a minimal example. The normal-use interface does not accept custom reference banks.

Retrieval reads only RGB. Only after a reference is chosen does the pipeline read its metric depth and K, run known-camera inference on it with the current depth model, compute the reference bias, and then select the query prediction. The query image needs no GT depth, and unknown-camera mode does not use the query K either.

Enabling DIODE requires a depth model that supports known-camera inference on the reference image. When the query camera is unknown, the three candidates must also provide usable camera estimates. A custom model that lacks these capabilities can still estimate depth on its own, but cannot run the corresponding reference selection.

Each pipeline computes and keeps the DIODE image features the first time it uses the reference bank, and afterward encodes only the query image. The calibration result for the same reference is also reused. When you pass the path of an image inside DIODE, its scene is excluded. When the input pixels are identical to an image in the bank, that image's scene is excluded as well. Resized, cropped, or re-encoded images from other sources carry no scene identifier, so the same scene cannot be recognized automatically from them.

## Command Line and Output Formats

```bash
python run.py generate --image query.png --seeds 0 1 2 --expansion 0.1 \
  --offload model --output outputs/context
python run.py depth --image query.png --depth-model moge2 --output outputs/depth
python run.py infer --image query.png --offload model --output outputs/example
python run.py infer --image query.png --offload model \
  --reference-bank diode --diode-root data/moge_eval/DIODE --output outputs/selected
```

Common options include `--config models.local.yaml` and `--device cuda:0`. Depth commands support `--camera K.json` (a file that stores the 3×3 array directly), `--depth-model`, and `--depth-batch-size`. Generation commands support `--outpainter`, `--expansion`, and `--seeds`.

| Command | Files in the output directory |
| --- | --- |
| `generate` | Outpainted images such as `candidate0.png`, plus `outpaintings.json` recording bbox / seed |
| `depth` | `depth.npz` |
| `infer` / `result.save()` | `direct.npz`, a PNG and an NPZ for each candidate, and `result.json`. Also `selected.npz` when the reference bank is enabled |

An NPZ always contains `depth`, and also stores whichever of `intrinsics`, `focal_px`, and `rays` the model provides. The normal-use interface does not write metrics or the paper's prediction caches. To reuse generation results across runs, save the outpaintings and reload them.
