---
name: contextdepth
description: Install and use ContextDepth, a training-free method that improves a frozen metric depth model by outpainting the image and selecting among the resulting predictions. Use it to clone the repository and set up its environment, run it on images, reproduce the crop block of the paper's Table 1 when the user asks for a test, or plug in the user's own outpainting or metric depth model.
---

# ContextDepth

ContextDepth takes one RGB image, outpaints its surroundings three times, runs a frozen metric depth model on the original image and on each outpainted candidate, and (with a labelled reference image) selects one prediction. Nothing is trained. The outpainting model and the depth model are separate adapters, so either can be replaced.

Repository: `https://github.com/Zeyu-Xiao/contextdepth`

This skill covers four jobs. Do only the ones the user asked for.

| Job | When | Section |
| --- | --- | --- |
| Set up | Always first | [1](#1-set-up) |
| Run on images | The user wants depth for their images | [2](#2-run-on-images) |
| Test | Only when the user asks to test or reproduce results | [3](#3-test-only-on-request) |
| Use the user's own model | The user names an outpainting or depth model of their own | [4](#4-use-your-own-models) |

## Ground rules

- Work inside the repository checkout, with the environment's own `python`. Do not install into, upgrade or modify an environment the user already has.
- Ask before any download above a few gigabytes, and before any long GPU job. State the size or duration first.
- Never type or store credentials. FLUX.1-Fill-dev is a gated model: the user accepts its terms on its Hugging Face page and runs `hf auth login` themselves.
- The scripts stop on a revision or SHA-256 mismatch. Report the mismatch; do not edit expected hashes or skip the check.
- Report only what was actually run. A passed import check is not a passed inference; a quick check on six images is not a reproduction of the table.
- Do not change precision, sampling steps, seeds, or the evaluation protocol to make numbers agree.

## 1. Set up

Requirements: Linux x86_64, an NVIDIA GPU with a driver that supports CUDA 12.8, Python 3.12 (Conda or `venv`), `git`.

```bash
git clone https://github.com/Zeyu-Xiao/contextdepth.git
cd contextdepth
conda create -n contextdepth python=3.12 pip -y
conda activate contextdepth
python -m pip install setuptools==80.10.2 wheel
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r install-contextdepth/requirements.txt
python install-contextdepth/setup_upstreams.py --install --group core
python -m pip install --no-deps -e third_party/utils3d
python install-contextdepth/check_installation.py
```

Without Conda: `python3.12 -m venv .venv && source .venv/bin/activate`, then the same `pip` commands.

`check_installation.py` imports the real model classes and runs small CPU operations. It loads no weights. Next, the default weights: MoGe-2 for depth (1.2 GB), and OneReward on FLUX.1-Fill-dev for outpainting (about 10 GB of FLUX components and a 24 GB transformer after conversion, plus the original OneReward download). Tell the user these sizes before downloading:

```bash
python install-contextdepth/prepare_weights.py depth --model moge2
hf download black-forest-labs/FLUX.1-Fill-dev --local-dir weights/FLUX.1-Fill-dev \
  --include model_index.json 'scheduler/*' 'tokenizer/*' 'tokenizer_2/*' \
  'text_encoder/*' 'text_encoder_2/*' 'vae/*' 'LICENSE*' POLICY.md README.md
hf download bytedance-research/OneReward --local-dir weights/OneReward-original \
  --include 'flux.1-fill-dev-OneReward-transformer/*' 'LICENSE*' README.md
python install-contextdepth/prepare_weights.py flux-base
python install-contextdepth/prepare_weights.py onereward \
  --source weights/OneReward-original/flux.1-fill-dev-OneReward-transformer
```

If the FLUX download is refused, stop and ask the user to accept the model's terms and log in. If the user already has checkpoints or source checkouts, do not download again: write a `models.local.yaml` with their paths and pass `--config models.local.yaml` (see "Local Path Configuration" in [docs/installation.md](../../docs/installation.md)).

Then confirm that inference works on one image:

```bash
python run.py infer --image path/to/image.png --output outputs/example
```

OneReward with every component on the GPU peaks at about 35 GB of GPU memory. On a smaller GPU add `--offload model`. Setup is done when this command writes `outputs/example/result.json`. Details, optional models and the reference bank: [docs/installation.md](../../docs/installation.md).

## 2. Run on images

```python
from contextdepth import ContextDepthPipeline

with ContextDepthPipeline() as pipeline:
    result = pipeline("query.png")           # pass camera=K (3x3, pixels) if intrinsics are known
    result.save("outputs/example")

direct = result.direct.depth                  # metres, same size as the input
candidates = [c.prediction.depth for c in result.candidates]   # one per outpainted image
```

By default each side grows by 10% and the three candidates use seeds 0, 1, 2. Without a reference bank the pipeline returns all predictions and selects none. With `reference_bank="diode", diode_root="data/moge_eval/DIODE"` it retrieves a labelled reference and fills `result.selected`. Command line: `python run.py generate | depth | infer`. Everything else (camera input, batching, reusing outpaintings, output files, other built-in depth models): [docs/usage.md](../../docs/usage.md).

## 3. Test (only on request)

Do not start a test unless the user asks to test, verify or reproduce. The test reproduces the crop block of the paper's Table 1: δ₁ of Query-only, Outpaint and Outpaint + Select for seven depth models, without and with the ground-truth camera. Full description: [docs/reproduction.md](../../docs/reproduction.md). Offer the user the two levels and let them choose.

**Quick check** (six iBims-1 queries, one 41 MB download, MoGe-2 only by default, about 20 minutes on one RTX 4090). It verifies the installation end to end:

```bash
hf download Ruicheng/monocular-geometry-evaluation iBims-1.zip --repo-type dataset \
  --revision 574610d77c0cc14d911a1ff46ec529ba934c8960 --local-dir data/downloads
mkdir -p data/moge_eval && unzip -q data/downloads/iBims-1.zip -d data/moge_eval
python -m evaluation prepare --eval-root data/moge_eval --output data/quick --subset quick
python -m evaluation run --manifest data/quick/manifest.json --models moge2 moge2_gt_fov --gpus 0 --output outputs/quick
python -m evaluation report --output outputs/quick
```

`prepare` fails if the decoded pixels differ from the paper's checksums. `report` exits non-zero if an experiment's Query-only δ₁ is more than 2 points from its stored value. Tell the user that result, and that the Outpaint columns of a six-image run are not comparable with the paper.

**Full run** (3,406 queries, 2.9 GB of evaluation data, eight depth checkpoints totalling about 23 GB, 10,218 generated images). Before starting, tell the user what it involves and get a go-ahead:

1. Install the other depth models and their weights: section 5 of [docs/installation.md](../../docs/installation.md). MoGe-3 needs the `ptxas` step described there.
2. Download and extract the six archives listed in [docs/reproduction.md](../../docs/reproduction.md#data).
3. Run:

```bash
python -m evaluation prepare --eval-root data/moge_eval --output data/paper
python -m evaluation run --manifest data/paper/manifest.json --models all --gpus 0 --output outputs/evaluation
python -m evaluation report --output outputs/evaluation
```

Generation alone takes about 47 seconds per image on an RTX 4090, roughly 5.5 GPU-days for the full cohort on one such card; say so before starting, and offer `--gpus 0 1 ...` or a subset of models. On a GPU with less than about 40 GB, use the `offload: model` option described in the reproduction guide. The run is resumable; rerun the same command after an interruption. To test a subset, list experiment names after `--models` (the table of names is in the reproduction guide). Outpainted images are not shipped: the run generates them. Give the user the table that `report` prints, with the differences from the paper as printed. Query-only should agree closely; Outpaint and Outpaint + Select are regenerated and are not expected to match to the last digit.

## 4. Use your own models

The user supplies how their model is called (its repository or import path, weights, and its inference function). You write one adapter file; nothing else in the repository changes.

**A metric depth model** becomes `baselines/<name>.py` with a class `Baseline(DepthBaseline)`. Follow [references/add-depth-model.md](references/add-depth-model.md). Then:

```python
with ContextDepthPipeline(
    depth_model="./baselines/<name>.py:Baseline",
    depth_options={"source": "/path/to/upstream", "weights": "/path/to/checkpoint"},
) as pipeline:
    result = pipeline("query.png")
```

**An outpainting model** becomes `generators/<name>.py` with a class `Outpainter(OutpaintingBaseline)`. Follow [references/add-outpainting-model.md](references/add-outpainting-model.md). Then:

```python
with ContextDepthPipeline(
    outpainter="./generators/<name>.py:Outpainter",
    outpainter_paths={"source": "/path/to/upstream", "weights": "/path/to/checkpoint"},
) as pipeline:
    result = pipeline("query.png")
```

Both can be combined in one pipeline. What to ask the user if they have not said:

- Depth model: where the code and weights are; whether its output is metric depth along the camera Z axis; whether it accepts intrinsics, and in what form; whether it returns a camera estimate.
- Outpainting model: where the code and weights are; how it takes the image, the mask and the seed; any size constraints.

Do not invent a metric scale or a camera for a model that does not provide one; say what is missing instead. Verify with the public pipeline on a real image before reporting success, and say which checks ran.

To run the test of section 3 with the user's models, write a config that names the adapters and their experiments: see "Using your own models" in [docs/reproduction.md](../../docs/reproduction.md#using-your-own-models).

## Reporting back

End with: what was installed and where; which commands ran and whether they passed; for a test, the printed table and whether it was the quick check or the full run; anything skipped or not verified.
