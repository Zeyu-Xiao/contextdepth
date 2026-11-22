# ContextDepth: Boosting Metric Depth Estimation via Richer Context

<!--links-->
[Project page](https://zeyu-xiao.github.io/contextdepth/) · [Installation](#installation) · [Usage](#usage) · [Agent skill](#agent-skill) · [Reproducing Table 1](#reproducing-table-1) · [Your own models](#your-own-models)

ContextDepth is a training-free way to improve a frozen metric depth model. It first outpaints the surroundings of the input image, runs the depth model on the original image and on each outpainted candidate, and then uses a labelled reference image to select one prediction. The outpainting model and the depth model connect through independent adapters, so either can be replaced.

The defaults are **OneReward** for outpainting and **MoGe-2** for depth. Each side is extended by **10%** of the image size, producing **3 candidates with different seeds** while preserving the original pixels. The reference bank is disabled by default; enable DIODE to select a prediction.

Built-in depth models: MoGe-3, DA3, MoGe-2, UniDepth V2, UniK3D, Depth Pro and Depth Pro 2.
## Installation

```bash
git clone https://github.com/Zeyu-Xiao/contextdepth.git
cd contextdepth
conda create -n contextdepth python=3.12 pip -y
conda activate contextdepth
python -m pip install setuptools==80.10.2 wheel
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r install-contextdepth/requirements.txt
python install-contextdepth/setup_upstreams.py --install --group core
python -m pip install --no-deps -e third_party/utils3d
python install-contextdepth/check_installation.py
```

Generation and depth estimation run in **one environment** for normal use. These commands target Linux with an NVIDIA GPU. A Python 3.12 virtual environment can be used instead of Conda.

Prepare the default model weights below. The two optional resources are needed only when enabling the DIODE reference bank.

| Resource | Download | Path after preparation |
| --- | --- | --- |
| OneReward ([paper](https://arxiv.org/abs/2508.21066)) | [Weights](https://huggingface.co/bytedance-research/OneReward) | `weights/OneReward/diffusers-bf16/` |
| FLUX.1-Fill-dev base components | [Weights](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) | `weights/FLUX.1-Fill-dev/` |
| MoGe2 ([paper](https://arxiv.org/abs/2507.02546)) | [Weights](https://huggingface.co/Ruicheng/moge-2-vitl-normal) | `weights/moge-2-vitl-normal/model.pt` |
| DIODE reference bank (optional) | [DIODE.zip](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation/blob/main/DIODE.zip) | `data/moge_eval/DIODE/` |
| SigLIP2 retrieval model (optional) | [Weights](https://huggingface.co/google/siglip2-so400m-patch16-384) | `weights/siglip2-so400m-patch16-384-vision/` |

Obtain access to FLUX on its model page before downloading. Convert OneReward to BF16 and export the SigLIP2 vision tower using the [weight preparation](docs/installation.md#3-prepare-the-default-model-weights) and [reference bank setup](docs/installation.md#4-optional-diode-reference-bank) instructions. The extracted DIODE directory should contain `.index.txt` directly.

Paths for normal use are configured in [`models.yaml`](models.yaml). To use existing local resources, pass `ContextDepthPipeline(config="models.local.yaml")` or `--config models.local.yaml` on the command line. See the [installation guide](docs/installation.md) for setup and GPU memory options.

## Usage

### Minimal Code Example

```python
from contextdepth import ContextDepthPipeline

with ContextDepthPipeline() as pipeline:
    result = pipeline("query.png")
    result.save("outputs/example")

candidate = result.candidates[0]
rgb = candidate.outpainting.rgb          # First complete outpainted image
depth = candidate.prediction.depth       # Depth in meters, with the same height and width as rgb
camera = candidate.prediction.camera     # Camera estimate for this outpainted image

direct_depth = result.direct.depth       # Baseline prediction from the original image alone
```

The input can be an image path, a PIL image, or a `uint8` RGB array. For a known camera, call `pipeline("query.png", camera=K)`, where `K` is the 3×3 intrinsic matrix in the original image's pixel coordinates. The pipeline adjusts the principal point for each outpainted image.

By default, the pipeline returns predictions for the original image and all three complete outpainted images. It does not select a candidate, so `result.selected` is `None`. With DIODE reference selection enabled below, `result.selected` contains the chosen prediction at the original image size. The default workflow requires no dataset manifest, ground-truth depth, or metric evaluation.

The full OneReward model requires substantial GPU memory. To offload inactive generation components to the CPU, use `ContextDepthPipeline(generation_options={"offload": "model"})`. This requires more system RAM and adds data transfer overhead.

### Separate Generation and Depth Estimation

```python
from contextdepth import ContextDepthPipeline

with ContextDepthPipeline() as pipeline:
    images = pipeline.generate("query.png")        # 10% per side; seeds 0, 1, 2
    pipeline.release_generation()                  # Release weights when generation is finished
    prediction = pipeline.estimate_depth(images[0].rgb)
    depth, camera = prediction.depth, prediction.camera
```

`estimate_depth()` loads only the depth model, and `generate()` loads only the generation model. Models are loaded on first use and reused by subsequent calls within the same pipeline instance.

Command-line usage:

```bash
python run.py generate --image query.png --output outputs/context
python run.py depth --image query.png --output outputs/depth
python run.py infer --image query.png --output outputs/example
```

Add `--offload model` to commands that generate images to enable CPU offloading. See the [usage guide](docs/usage.md) for more options, reusing existing outpaintings, batching, and output formats.

### Optional: DIODE Reference Selection

After preparing DIODE and SigLIP2 as described in [Installation](#installation):

```python
from contextdepth import ContextDepthPipeline

with ContextDepthPipeline(reference_bank="diode", diode_root="data/moge_eval/DIODE") as pipeline:
    result = pipeline("query.png")
    depth = result.selected.depth                  # Same height and width as the original query.png
    result.save("outputs/with_reference")
```

The pipeline retrieves a DIODE reference using RGB features, calibrates the selection process with that reference's depth and camera, and returns the selected prediction. The query needs no ground-truth depth. Reference features and completed calibrations are reused within the pipeline instance. The public inference API currently supports DIODE as its only reference bank.

## Agent skill

The repository ships one skill for AI coding agents: [`skills/contextdepth/SKILL.md`](skills/contextdepth/SKILL.md). It tells an agent how to clone the code and build the environment, run ContextDepth on images, reproduce Table 1 when you ask for it, and plug in your own outpainting or depth model. Give your agent this prompt:

> Read https://raw.githubusercontent.com/Zeyu-Xiao/contextdepth/main/skills/contextdepth/SKILL.md and follow it to set up ContextDepth on this machine.

Inside a checkout, point the agent at the file instead, or copy the `skills/contextdepth` folder into your agent's skills directory (for Claude Code: `~/.claude/skills/`).

## Reproducing Table 1

The evaluation reproduces the crop block of the paper's Table 1: δ₁ of Query-only, Outpaint and Outpaint + Select for seven depth models, without and with the ground-truth camera, on **3,406 images** from six benchmarks. [docs/reproduction.md](docs/reproduction.md) describes the protocol (cropping, context, camera settings, selection), the data, and the expected numbers.

Quick check on six images (one 41 MB download, default models only):

```bash
hf download Ruicheng/monocular-geometry-evaluation iBims-1.zip --repo-type dataset \
  --revision 574610d77c0cc14d911a1ff46ec529ba934c8960 --local-dir data/downloads
mkdir -p data/moge_eval && unzip -q data/downloads/iBims-1.zip -d data/moge_eval
python -m evaluation prepare --eval-root data/moge_eval --output data/quick --subset quick
python -m evaluation run --manifest data/quick/manifest.json --models moge2 moge2_gt_fov --gpus 0 --output outputs/quick
python -m evaluation report --output outputs/quick
```

Full run, after installing the [other depth models](docs/installation.md#5-optional-other-built-in-depth-models) and downloading the [six evaluation archives](docs/reproduction.md#data):

```bash
python -m evaluation prepare --eval-root data/moge_eval --output data/paper
python -m evaluation run --manifest data/paper/manifest.json --models all --gpus 0 --output outputs/evaluation
python -m evaluation report --output outputs/evaluation
```

`prepare` crops every image with the paper's rule and writes queries, ground truth and intrinsics. `run` outpaints three candidates per query with OneReward, predicts, selects and scores; it is resumable. `report` prints the results in the layout of the table, next to the paper's values. Outpainted images are not distributed; they are generated by the run.

| Depth model | Weights | Location |
| --- | --- | --- |
| MoGe-2 | [moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) | `weights/moge-2-vitl-normal/model.pt` |
| MoGe-3 | [moge-3-vitg](https://huggingface.co/Ruicheng/moge-3-vitg) | `weights/moge-3-vitg/model.pt` |
| DA3 (no GT camera) | [DA3NESTED-GIANT-LARGE-1.1](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1) | `weights/DA3NESTED-GIANT-LARGE-1.1/` |
| DA3 (GT camera) | [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE) | `weights/DA3METRIC-LARGE/` |
| UniDepth V2 | [unidepth-v2-vitl14](https://huggingface.co/lpiccinelli/unidepth-v2-vitl14) | `weights/unidepth-v2-vitl14/` |
| UniK3D | [unik3d-vitl](https://huggingface.co/lpiccinelli/unik3d-vitl) | `weights/unik3d-vitl/` |
| Depth Pro | [depth_pro.pt](https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt) | `weights/depth_pro.pt` |
| Depth Pro 2 | [DepthPro2-H](https://huggingface.co/apple/DepthPro2-H), [DepthPro2-focal](https://huggingface.co/apple/DepthPro2-focal) | `weights/depth_pro2/` |

`python install-contextdepth/prepare_weights.py depth` downloads and hash-checks all of them. Paths are configured in [`evaluation/models.yaml`](evaluation/models.yaml); to use resources you already have, pass `--config models.local.yaml`.

## Your own models

Each side of the method is one adapter file, loaded by path. No registry and no change to the pipeline is needed.

**Depth model.** Add `baselines/xxx.py` with a `Baseline` class that wraps the model's own inference API and returns metric depth (and, if the model has one, a camera estimate) on the complete input image. Guide: [add-depth-model](skills/contextdepth/references/add-depth-model.md).

```python
from contextdepth import ContextDepthPipeline

with ContextDepthPipeline(
    depth_model="./baselines/xxx.py:Baseline",
    depth_options={"source": "/path/to/source", "weights": "/path/to/weights"},
) as pipeline:
    prediction = pipeline.estimate_depth("query.png")
```

**Outpainting model.** Add `generators/xxx.py` with an `Outpainter` class that receives the image, its position on the canvas and one seed per candidate. The shared base class restores the original pixels. Guide: [add-outpainting-model](skills/contextdepth/references/add-outpainting-model.md).

```python
from contextdepth import ContextDepthPipeline

with ContextDepthPipeline(
    outpainter="./generators/xxx.py:Outpainter",
    outpainter_paths={"source": "/path/to/source", "weights": "/path/to/weights"},
) as pipeline:
    images = pipeline.generate("query.png")
```

With the [agent skill](#agent-skill), you only describe how your model is called; the agent writes the adapter and verifies it. The shared base classes are in [`contextdepth/baseline.py`](contextdepth/baseline.py) and [`contextdepth/generation.py`](contextdepth/generation.py); the [adapter guide](docs/baselines.md) documents the full depth interface. The same adapters can be evaluated on the benchmark: see [Using your own models](docs/reproduction.md#using-your-own-models).

## Citation

```bibtex
@inproceedings{contextdepth,
  title     = {ContextDepth: Boosting Metric Depth Estimation via Richer Context},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027}
}
```
## License

ContextDepth's own code is released under the [MIT License](LICENSE). Upstream model code, weights, and datasets remain subject to their respective licenses. Bundled compatibility patches retain the original projects' terms.
