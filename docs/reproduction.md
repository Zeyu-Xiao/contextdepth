# Reproducing Table 1 (crop block)

This guide reproduces the crop block of the paper's Table 1: δ₁ of **Query-only**, **Outpaint** and **Outpaint + Select** for every depth model, without and with the ground-truth camera, on 3,406 images from six benchmarks. It is the only evaluation shipped with the code. Run everything from the repository root, in the environment from [installation.md](installation.md).

There are two ways to run it:

| | Images | Data to download | Purpose |
| --- | --- | --- | --- |
| [Quick check](#quick-check) | 6 queries from iBims-1 | `iBims-1.zip` (41 MB) | Confirm that an installation works end to end |
| [Full run](#full-run) | 3,406 queries | six archives (2.9 GB) | Reproduce the table |

Outpainted images are not distributed. Both runs generate them with OneReward, three per query.

## What is evaluated

**The crop.** Each benchmark image is cropped to its centre: one sixth of the width is removed on the left and on the right, one sixth of the height at the top and at the bottom ([`paper_roi_geometry`](../evaluation/data.py)). The crop is the *query*. Ground-truth depth and intrinsics are cropped with it. The crop is resized to a width of 640 pixels for generation and inference; scores are computed on the crop's original pixel grid.

**The context.** OneReward outpaints the query by 25% of its size on each of the four sides. Three candidates are generated per query, each with its own seed; the query pixels are pasted back unchanged. `evaluation/protocol/roi_samples.jsonl.gz` lists the 3,406 queries, their seeds, and a checksum of every query's pixels.

**The three columns.**

| Column | Key in the results | What is scored |
| --- | --- | --- |
| Query-only | `direct` | The frozen depth model on the query alone |
| Outpaint | `seed0` | The same model on the first candidate, cropped back to the query |
| Outpaint + Select | `selector` | The prediction chosen among the three candidates (and, with a known camera, the query) |

**The reference bank.** Selection needs one labelled reference image per query. The bank is the evaluation cohort itself: each query is bound to its nearest SigLIP2 neighbour from a *different scene*. These bindings are shipped in the protocol file, so the retrieval model is not needed. Only the reference's own depth and camera are read; the query's ground truth is used for scoring alone. Pass `--recompute-retrieval` to recompute the bindings with SigLIP2 instead.

**Camera settings.** Every model is run twice, once without and once with the ground-truth camera:

| Model | Without GT camera | With GT camera |
| --- | --- | --- |
| MoGe-3 | `moge3_unknown` | `moge3_known` |
| DA3 | `da3nested` (Nested checkpoint) | `da3metric_gt_focal` (Metric checkpoint) |
| MoGe-2 | `moge2` | `moge2_gt_fov` |
| UniDepth V2 | `unidepth_v2_vitl14` | `unidepth_gt_k` |
| UniK3D | `unik3d_unknown` | `unik3d_known` |
| Depth Pro | `depth_pro` | `depth_pro_gt_focal` |
| Depth Pro 2 | `depth_pro2` | `depth_pro2_gt_focal` |

Without the GT camera, UniDepth V2 and UniK3D estimate the camera once on the query and reuse it for the outpainted candidates (`camera_source: query` in [`evaluation/models.yaml`](../evaluation/models.yaml)). The other models estimate a camera on every input. In all cases the selector observes the camera each candidate estimates on its own. With the GT camera, the calibration is moved onto the candidate canvas and passed through each model's own camera interface.

**The score.** δ₁ is the fraction of valid ground-truth pixels with `max(d̂/d, d/d̂) < 1.25`, with no scale or shift alignment, averaged over images. Depth Pro 2 can predict infinite depth for sky; such pixels stay in the evaluation and count as δ₁ failures. `metrics.json` also holds AbsRel, which is undefined for an experiment when any of its images has an infinite prediction on a ground-truth pixel.

## Quick check

The quick check prepares six iBims-1 queries whose references are also iBims-1 images, generates their 18 candidates, and runs the complete pipeline. By default it uses MoGe-2, which needs nothing beyond the default installation. On one RTX 4090 it takes about 20 minutes, most of it generation.

```bash
hf download Ruicheng/monocular-geometry-evaluation iBims-1.zip --repo-type dataset \
  --revision 574610d77c0cc14d911a1ff46ec529ba934c8960 --local-dir data/downloads
mkdir -p data/moge_eval && unzip -q data/downloads/iBims-1.zip -d data/moge_eval

python -m evaluation prepare --eval-root data/moge_eval --output data/quick --subset quick
python -m evaluation run --manifest data/quick/manifest.json \
  --models moge2 moge2_gt_fov --gpus 0 --output outputs/quick
python -m evaluation report --output outputs/quick
```

`prepare` fails if a decoded query differs from the stored pixel checksum, so a passing `prepare` means the data and the cropping match the paper's. `report` prints the measured δ₁ and compares each experiment's mean Query-only δ₁ over the six images with the value stored in `evaluation/protocol/quick.json`. Query-only does not depend on generation, so it should match; the command exits with a non-zero status if an experiment is off by more than `--tolerance` (0.02 by default, that is, 2 points). The Outpaint and Outpaint + Select numbers on six images only show that generation, depth, selection and scoring run; they are not comparable with the table.

To include more models, install them first ([optional depth models](installation.md#5-optional-other-built-in-depth-models)) and name their experiments, or pass `--models all`.

## Full run

### Data

Download the six archives of the [MoGe evaluation release](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation) at the pinned revision and extract them side by side:

```bash
hf download Ruicheng/monocular-geometry-evaluation \
  DIODE.zip ETH3D.zip HAMMER.zip KITTI.zip NYUv2.zip iBims-1.zip --repo-type dataset \
  --revision 574610d77c0cc14d911a1ff46ec529ba934c8960 --local-dir data/downloads
mkdir -p data/moge_eval
for name in DIODE ETH3D HAMMER KITTI NYUv2 iBims-1; do
  unzip -q data/downloads/$name.zip -d data/moge_eval
done
```

| Archive | Size |
| --- | ---: |
| `DIODE.zip` | 573 MB |
| `ETH3D.zip` | 1.36 GB |
| `HAMMER.zip` | 403 MB |
| `KITTI.zip` | 249 MB |
| `NYUv2.zip` | 246 MB |
| `iBims-1.zip` | 41 MB |

Sizes and SHA-256 checksums are in [`evaluation/protocol/full.sources.json`](../evaluation/protocol/full.sources.json). The datasets keep their own licences.

### Models

Install the optional depth models and download their weights as described in [installation.md](installation.md#5-optional-other-built-in-depth-models). The seven models use eight depth checkpoints; MoGe-3 also needs the `ptxas` compiler step described there.

### Run

```bash
# Crop every image, write queries, ground truth and intrinsics
python -m evaluation prepare --eval-root data/moge_eval --output data/paper

# Outpaint (3 candidates per query), predict, select and score
python -m evaluation run --manifest data/paper/manifest.json --models all \
  --gpus 0 --output outputs/evaluation

# Lay the results out like the table and compare with the paper
python -m evaluation report --output outputs/evaluation
```

The run is resumable: existing candidates and predictions are reused when their inputs and configuration are unchanged, so an interrupted run continues where it stopped. The stages can also be run one by one (`generate`, `predict`, `select`, `evaluate`), for instance to generate on one machine and predict on another. `--gpus 0 1 2 3` spreads the work over several GPUs. To evaluate a subset of models, list their experiment names after `--models`.

Generation dominates the cost: 10,218 images at 50 diffusion steps each. Depth inference then covers 3,406 queries and their candidates for 14 experiments, plus the reference images.

Measured on one RTX 4090 with 48 GB of memory: generating one candidate at the evaluation canvas size takes about 47 seconds, so the full cohort needs roughly 5.5 GPU-days of generation on such a card, and the pipeline peaks at 35 GB of GPU memory. Use several GPUs to shorten the run. On a GPU with less memory, let OneReward keep its inactive components on the CPU by adding this to a local config and passing it with `--config`:

```yaml
# models.local.yaml
outpainting:
  options:
    offload: model
```

This uses more system memory and more time per image. It does not change the sampling. In our single-image test it lowered the peak from 35 GB to 24.4 GB, so a 24 GB card is at the limit; we have not run on one.

### Expected results

`report` prints the measured values next to these (δ₁ in percent, 3,406 images):

| Model | Query-only | Outpaint | Outpaint + Select | Query-only | Outpaint | Outpaint + Select |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | *no GT camera* | | | *GT camera* | | |
| MoGe-3 | 62.67 | 76.73 | 78.99 | 78.36 | 78.84 | 81.21 |
| DA3 | 51.17 | 68.47 | 73.82 | 73.99 | 79.64 | 81.05 |
| MoGe-2 | 60.32 | 77.06 | 78.28 | 80.39 | 82.02 | 82.83 |
| UniDepth V2 | 62.06 | 69.91 | 72.04 | 73.01 | 72.85 | 75.56 |
| UniK3D | 62.80 | 74.68 | 76.37 | 76.26 | 76.74 | 78.62 |
| Depth Pro | 55.62 | 57.40 | 60.00 | 68.12 | 67.75 | 70.56 |
| Depth Pro 2 | 62.80 | 71.49 | 76.21 | 84.08 | 85.59 | 86.54 |

Depth Pro 2 was evaluated after the paper was written, with the same protocol, and is not in the paper's table.

How closely to expect a match:

- **Query-only** does not involve generation. It should agree with the table up to small numerical differences between GPUs and library builds.
- **Outpaint** and **Outpaint + Select** depend on the generated images. The paper's candidates were generated once and archived; a new run regenerates them with the same seeds, but diffusion sampling is not bit-identical across GPUs and library versions, so these columns will not match to the last digit. As an indication: for the six quick-check queries, the 18 candidates we regenerated on another GPU differed from the archived ones by 0.2 to 2.9 grey levels (of 255) on average per image, none was identical, and with MoGe-2 the six-image mean of Outpaint moved by at most 0.4 points and that of Outpaint + Select by at most 1.5 points (one query selected a different candidate). We have not measured the difference on the full cohort.
- The paper's DA3 runs restored a recorded random-number state before each input. A new run seeds each input instead, which can move DA3's numbers slightly.

## Using your own models

The same commands evaluate a model you added. Write a config that defines the model and its experiments; a file that contains an `experiments` section replaces the built-in ones:

```yaml
# my_models.yaml
outpainting:
  implementation: contextdepth.outpainting:OneRewardOutpainter
  source: third_party/OneReward
  base_model: weights/FLUX.1-Fill-dev
  weights: weights/OneReward/diffusers-bf16
models:
  mine:
    implementation: baselines/my_model.py:Baseline
    source: /path/to/upstream
    weights: /path/to/checkpoint
experiments:
  mine_unknown: {baseline: mine, camera: unknown, context_fraction: 0.25, reference: mine_known}
  mine_known:   {baseline: mine, camera: known,   context_fraction: 0.25, reference: mine_known}
```

```bash
python -m evaluation run --manifest data/paper/manifest.json --config my_models.yaml \
  --models all --recompute-retrieval --gpus 0 --output outputs/mine
python -m evaluation report --output outputs/mine --reference my_table.json
```

New experiment names have no stored reference bindings. Either pass `--recompute-retrieval`, which needs the [SigLIP2 weights](installation.md#4-optional-diode-reference-bank), or pass `--retrieval bindings.json`, a JSON object that maps each query uid to a reference uid (the bindings of any built-in experiment in an existing run's `reference_ids.json` can be reused this way). Reference calibration needs a known-camera experiment, and unknown-camera selection needs a camera estimate from the model; see the [adapter guide](baselines.md). `report --reference` takes a file shaped like [`table1_crop.json`](../evaluation/protocol/table1_crop.json) that names your experiments.

To evaluate with your own outpainting model, point `outpainting.implementation` at your adapter, for example `generators/my_model.py:Outpainter`. An adapter written for normal use (a subclass of [`OutpaintingBaseline`](../contextdepth/generation.py)) works unchanged: the evaluation calls its `generate_images` with each query, its position on the 25% canvas and the query's seed, then restores the query pixels. Candidates live next to the manifest and are regenerated whenever the outpainting configuration changes, so run `prepare` into a second folder if you want to keep the OneReward candidates. Candidates produced elsewhere can also be placed at the paths listed in the manifest and used with `--skip-generation`.
