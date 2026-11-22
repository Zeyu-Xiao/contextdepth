# Integrate an Outpainting Model

The goal is to deliver a callable `generators/xxx.py` that exposes an
`Outpainter` class. The public pipeline loads it through a file entry point,
with no model registry and no need to install the module into the global
environment. Outpainting and depth are independent interfaces. The adapter does
not need to load a depth model, the reference bank, or evaluation data.

## Confirm the Official Generation Entry Point First

Read the official inpainting/outpainting interface for the version the user
specifies, and read this repository's
[OutpaintingBaseline](../../../contextdepth/generation.py) and the
[OneRewardBaseline example](../../../contextdepth/outpainting.py). Establish the
meaning of the mask, the RGB range, native canvas size constraints, the prompt,
the random number entry point, and whether preprocessing is already done. Do
not infer these behaviors from the model name alone.

If the model needs an extra prompt or generation parameters, put them in
`paths["options"]`, and document the defaults and required resources clearly.
If the model cannot outpaint with a protected region, state the gap clearly,
and do not pass off a blank border as generated content. Do not load weights or
allocate GPU memory at module import. The constructor does the loading, and
`close()` releases the model it holds.

## Implement in a Single File

Subclass `OutpaintingBaseline` and implement the interface skeleton below,
replacing the method bodies with the actual official calls:

```python
from contextdepth.generation import OutpaintingBaseline


class Outpainter(OutpaintingBaseline):
    def __init__(self, paths, device="cuda:0"):
        super().__init__(paths, device)
        # Load the official generation model from the resource paths here.

    def generate_images(self, rgb, *, bbox, canvas_size, seeds):
        # Build the canvas and the protection mask, then call the official model.
        # Return a list[np.ndarray] with the same length and order as seeds.
        raise NotImplementedError

    def close(self):
        super().close()
        # Release the model references and resources held by this instance.
```

`generate_images` receives the following, already determined:

| Parameter | Meaning |
| --- | --- |
| `rgb` | The original, nonempty, `uint8` HWC RGB. Its size and geometry must not be changed. |
| `bbox` | `(x0,y0,x1,y1)` of the original image in the returned canvas. The bottom-right corner is an exclusive upper bound. |
| `canvas_size` | `(width,height)` of the returned images. |
| `seeds` | Independent integer seeds, in output order. |

Return each complete canvas as `uint8` HWC RGB with a size that matches
`canvas_size` exactly. The base class checks the count and sizes, and pastes
the original RGB back into the bbox pixel by pixel. Do not override
`generate()` to bypass these checks. A mask that marks the query region as
protected should still be passed to the official model. Do not rely on the
paste-back after generation as a substitute for correct conditioning input.

By default, the public `generate(rgb, expansion=0.1, seeds=(0,1,2))` extends
each of the four sides by 10% of the corresponding axis of the original image
and returns three `Outpainting` items. `expansion` can also be a complete
`top/bottom/left/right` dictionary or a 4-tuple in that order. Margins are
rounded to integers with decimal half-up rounding. Each result contains the
complete `rgb`, `bbox`, `seed`, and optional metadata. `map_intrinsics(K)`
translates the original image's K onto the new canvas. The base class performs
these geometry computations. The adapter uses only the bbox and canvas size
passed in.

If the official model accepts only aligned sizes, the adapter may add minimal
temporary padding and remove the extra padding before returning. Document the
handling, and ensure the original image is not resized and the bbox does not
shift. Do not swap the default 10% for the historical 25% generation followed
by cropping, and do not reuse the paper's UID-derived seeds. A model that must
resize natively may be used only if it can meet the external geometry and pixel
conventions above, with its internal handling documented. If it cannot, report
the limitation.

Each item must use its own seed. Avoid resetting to the same seed repeatedly
inside a loop. Generating serially, one seed at a time, is fine. If you use a
native batch, confirm that each item's random stream and output order are
independent. Different seeds do not guarantee visual differences.

## Verify Through the Public Entry Point

Call from the repository root. The
[pipeline](../../../contextdepth/pipeline.py)'s `outpainter_paths` is passed
directly as the adapter's resource `paths`, while `generation_options` is
merged into `paths["options"]`. Replace the entry point and paths in the
example below. The default model configuration does not need to change:

```python
from contextdepth import ContextDepthPipeline as Pipeline

with Pipeline(
    outpainter="./generators/xxx.py:Outpainter",
    outpainter_paths={"source": "/path/to/upstream", "weights": "/path/to/checkpoint"},
    generation_options={"prompt": "Extend the surrounding scene naturally"},
    device="cuda:0",
) as pipeline:
    images = pipeline.generate("query.png")
    print([(item.rgb.shape, item.bbox, item.seed) for item in images])
```

For asymmetric expansion, call `pipeline.generate(rgb, expansion={"top": .1, "bottom": .2, "left": 0, "right": .3})`.
An adapter instance the user has already loaded can also be passed directly to
`outpainter`. The user manages the lifetime of such an instance.

First use a CPU stand-in to verify the default three seeds, asymmetric
expansion, exact sizes, strict equality between the pixels inside the bbox and
the original image, and the K translation. Then call the real model under the
user's runtime conditions. If only the stand-in tests passed, state that real
generation has not been verified. Deliver the single file, a calling example,
the official dependencies and resource requirements, any size or GPU memory
limits, and the actual verification results. Do not modify the historical
canvas, random seed, or cache protocols in `evaluation/`.

## A Request Users Can Copy to an AI

```text
Follow skills/contextdepth/references/add-outpainting-model.md and integrate the outpainting model I provide into ContextDepth.
Official source location or link: [fill in]. Version: [fill in]. Weights path: [fill in].
Add only generators/xxx.py, expose Outpainter, and call the official generation API.
Use the OutpaintingBaseline interface. Default to 10% expansion on each of the four sides and
three independent seeds, and support asymmetric expansion.
Return the complete RGB. The original image size and every pixel inside the bbox must stay unchanged.
No changes are needed to the registry, the pipeline, or the paper evaluation logic. Give me a calling
example for the public pipeline, and state which verifications were actually run and the model's own limitations.
```
