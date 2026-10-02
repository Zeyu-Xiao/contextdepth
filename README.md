# ContextDepth

ContextDepth is a small, file-based reference-selection pipeline for frozen
metric-depth models. Given a query RGB image and a bank of reference RGB,
metric-depth, and camera examples, it can:

1. generate three independent outpaintings;
2. run depth estimation on the direct image and the three candidates;
3. retrieve one different-scene reference using RGB only;
4. select one complete depth prediction using the reference residual.

No training and no query ground-truth depth are used during inference. The
package contains only the runtime wrapper and installation helpers. Checkpoints,
upstream repositories, datasets, and generated images are downloaded or
supplied separately.

The six supported backbones are MoGe-3 ViT-G, Depth Anything 3, MoGe-2,
UniK3D ViT-L, Depth Pro, and UniDepth V2 ViT-L14. The twelve configuration
names are:

```text
moge3_unknown       moge3_known
da3nested           da3metric_gt_focal
moge2               moge2_gt_fov
unik3d_unknown      unik3d_known
depth_pro           depth_pro_gt_focal
unidepth_v2_vitl14  unidepth_gt_k
```

The unknown-camera settings do not receive query intrinsics. Known-camera
settings receive a 3x3 query K. Each run has exactly three candidates. The
unknown selector uses the nearest predicted-camera candidate as its anchor and
never falls back to Query-only; the known selector may return Query-only.

## Package contents

```text
run.py                         command-line entry point
models.yaml                    portable local-path configuration
contextdepth/                  inference, selection, evaluation, and I/O
install-contextdepth/          installation skill and pinned setup helpers
LICENSE                        license for the wrapper code
```

There are intentionally no benchmark, report, saved-metric, or internal
workflow files in this package.

The installation helpers include a measured runtime manifest and a read-only
runtime check. These describe the validated binary environment; they do not
bundle that environment, model weights, or the external benchmark assets.

## Install

Clone this private repository with your GitHub account, then run all setup
commands from its root on the Linux/NVIDIA machine used for inference:

```bash
git clone https://github.com/Zeyu-Xiao/contextdepth.git
cd contextdepth
```

Start with the complete environment recipe below before running the source
preparation commands. This checkout contains the code, not a preinstalled
environment or a self-contained paper benchmark. Exact historical replay also
requires the external frozen ROI specification, reference bank, archived
candidates and associated RNG state described below.

An agent performing installation should first read
[`install-contextdepth/SKILL.md`](install-contextdepth/SKILL.md). It creates two
isolated environments because the depth and diffusion dependencies are
different. The skill also prepares pinned upstream checkouts and verifies the
downloaded checkpoint bytes. It never stores credentials in this repository.

The source-preparation commands are:

```bash
python install-contextdepth/setup_upstreams.py --install
python install-contextdepth/setup_upstreams.py --check
python install-contextdepth/prepare_weights.py depth
```

This is not a substitute for the complete environment setup below; an agent
should follow the skill when starting from a clean machine.

For generation, download the user-authorized FLUX.1-Fill-dev and OneReward
assets, then run the `flux-base` and `onereward` preparation commands in the
skill. No model weights are bundled here.

## Exact ROI preparation and validation scope

The MoGe evaluation release's original JPEG files can reproduce the historical
iBims RGB inputs. Use `prepare --spec SPEC --eval-root ROOT --output OUT`
with the externally supplied frozen specification. `ROOT` is the directory
containing `iBims-1/<scene>/image.jpg` and the corresponding depth files.
The specification must preserve each UID, crop box, target size, camera, seeds,
and expected RGB hashes; do not approximate the crop from a fraction alone.

The validated iBims path in `contextdepth/data.py` is:

1. Decode JPEG bytes with OpenCV `imdecode(..., IMREAD_COLOR)` and convert BGR
   to RGB with `cvtColor(..., COLOR_BGR2RGB)`.
2. Crop using the recorded `source.crop_xywh` and check
   `source.retrieval_rgb_sha256` against the contiguous uint8 RGB bytes.
3. Resize that RGB crop using Pillow `Image.Resampling.LANCZOS` to the exact
   `geometry.roi_width` and `geometry.roi_height`.
4. Check `query_rgb_sha256` before saving the query PNG. These are decoded
   pixel hashes, not PNG file hashes. If neither supported decoding path
   produces the expected query hash, preparation stops.

For the 100-image frozen iBims cohort, the query decoder is OpenCV. Pillow
performs the resize, not the JPEG decode. For compatibility with other
historical cohorts, the general preparer tries Pillow when the specification
requests it or when the OpenCV-derived query hash does not match. It still
requires the recorded query hash and records the decoder in its output;
retrieval pixels must always pass the OpenCV hash check. All 100 validated
iBims queries matched through OpenCV without this fallback. Do not change an
expected hash to silence a mismatch. The June full-image PNG is not required
to rebuild these iBims inputs.

The completed input audit matched all 100 fixed ROIs and all 100 resized
queries. A separate depth replay using the restored binary environment,
archived candidates and associated RNG state matched 4,800 arrays
(100 queries x 12 configurations x direct/three candidates). The input audit
was performed separately from that inference replay; this was not a new
end-to-end diffusion run. All reported table entries match at paper precision;
19 of 24 raw aggregate floats match exactly and five differ by one ULP. The
aggregation difference has not been removed by altering metrics or baselines.
Full-cohort validation of the other datasets remains outstanding.

See `install-contextdepth/SKILL.md` for the observed Conda builds, decoder
backend and exact compiler. A matching package version list alone is not a
prediction-parity guarantee. A fresh installation from the updated recipe
still requires the same input, asset and prediction checks.

## Input manifest

Paths are relative to the manifest unless `data_root` or `--data-root` is
provided. A reference must contain an RGB image, a metric-depth NPZ, and its
pixel-space K. The NPZ must contain `depth`; an optional `mask` must be the
finite support (`depth > 1e-6`). A query may contain `depth` for evaluation and
`camera` for known-camera inference.

```json
{
  "data_root": ".",
  "samples": [
    {"uid": "query001", "dataset": "custom", "scene_group": "scene001",
     "image": "query/query001.png"}
  ],
  "references": [
    {"uid": "ref001", "dataset": "custom", "scene_group": "scene002",
     "image": "references/ref001.png", "depth": "references/ref001.npz",
     "camera": [[900, 0, 620], [0, 900, 185], [0, 0, 1]]}
  ]
}
```

The reference must be from a different scene/capture than the query. Relative
candidate paths default to
`outpaintings/<uid>/candidate0.png`, `candidate1.png`, and `candidate2.png`.
Existing candidates can instead be listed in a three-element `candidates`
field and used with `--skip-generation`. Their center ROI must equal the query
pixels exactly. Images are otherwise kept at their supplied resolution; there
is no hidden resize to 640 pixels for ordinary inference.

For the single-image command, a reference-bank-only file is also accepted:

```json
{
  "references": [
    {"uid": "ref001", "dataset": "custom", "scene_group": "scene002",
     "image": "references/ref001.png", "depth": "references/ref001.npz",
     "camera": [[900, 0, 620], [0, 900, 185], [0, 0, 1]]}
  ]
}
```

## Run one image

Prepare a manifest in the depth or generation environment:

```bash
python run.py infer \
  --image query.png \
  --reference-bank reference_bank.json \
  --models moge3_unknown \
  --output outputs/example \
  --prepare-only
```

Generate the three candidates in the generation environment:

```bash
python run.py generate --manifest outputs/example/input.json \
  --config models.yaml --gpus 0 --generation-batch-size 1
```

Run depth, retrieval, selection, and (when query GT is present) evaluation in
the depth environment:

```bash
python run.py run --manifest outputs/example/input.json \
  --config models.yaml --models moge3_unknown --skip-generation \
  --output outputs/example --gpus 0
```

For known-camera inference, select one of `moge3_known`, `moge2_gt_fov`,
`da3metric_gt_focal`, `unik3d_known`, `depth_pro_gt_focal`, or
`unidepth_gt_k`, and pass `--camera query_camera.json` to `infer`. Keep
`--prepare-only` when using the recommended separate generation and depth
environments. A single-environment end-to-end run is supported only when that
environment has both dependency sets installed.

## Run a dataset efficiently

Stages communicate through ordinary files and are resumable. Generate once,
then reuse the candidates for all depth configurations:

```bash
# generation environment; equal canvas sizes can be real batches
python run.py generate --manifest data/manifest.json --config models.yaml \
  --gpus 0 1 2 --generation-batch-size 1

# depth environment; persistent workers share the GPUs dynamically
python run.py predict --manifest data/manifest.json --config models.yaml \
  --models all --gpus 0 1 2 3 --workers-per-gpu 1 \
  --output outputs/all
python run.py select --manifest data/manifest.json --config models.yaml \
  --models all --output outputs/all
python run.py evaluate --manifest data/manifest.json --config models.yaml \
  --models all --output outputs/all
```

`evaluate` requires a finite metric `depth` entry for every query. Known-camera
settings additionally require the query `camera`; unknown-camera settings do
not use it for inference.

Start with generation batch size 1 for the recorded protocol; increase it only
after confirming that canvas sizes match and GPU memory remains sufficient.
The full OneReward/FLUX generator occupies roughly 35 GiB of resident GPU
memory, so a 24 GiB card may not fit the unoffloaded path. `--workers-per-gpu 2`
is available when two depth-model instances fit in memory.
Generation groups equal canvas sizes and keeps one pipeline resident per GPU.
Depth workers use bounded read-ahead and atomic result files; references are
calibrated once per backbone. Retrieval uses a batch of 16 by default. DA3 is
called through its independent-image path, not as an unrelated multiview list.

Useful stage commands are `generate`, `retrieve`, `predict`, `select`,
`evaluate`, and `run`. `prepare --spec SPEC --eval-root ROOT --output OUT`
materializes an externally supplied exact evaluation specification. The
`import-assets` command imports externally supplied candidate images and RNG
states without overwriting different existing files:

```bash
python run.py import-assets --manifest data/manifest.json \
  --index /path/to/assets.jsonl.gz --source /path/to/assets --mode copy
```

## Outputs and numerical contract

Predictions are stored under `predictions/<config>/`, reference calibration
under `references/<backend>/`, and selected metric depth under
`selected/<config>/`. Each NPZ contains depth arrays; the adjacent JSON records
the decision, camera observations, reference, cache identity, and timing.

The implementation preserves the paper protocol: three UID-derived candidate
seeds (candidate0 is not literal integer seed 0), fixed finite GT support,
strict `delta1 < 1.25`, no GT scale/shift alignment, no clipping, no depth
fusion, and no prediction-dependent mask. MoGe-3 uses three refinement steps.
Unknown MoGe-3/DA3/MoGe-2 use the aligned 25% canvas; unknown UniK3D, Depth
Pro, and UniDepth V2 crop the same candidates to 10%. Known-camera settings
use the aligned 25% canvas. These fractions are per side of the query ROI.

For exact paper-number reproduction, supply the same externally archived
candidate images, query/reference manifest, and DA3 before-input RNG metadata.
Fresh diffusion output or a different batch/runtime stack is not guaranteed to
produce identical pixels. Those external or dataset-specific assets are
deliberately not included in this blind source package.

## Attribution

The wrapper is MIT-licensed. The two bundled compatibility patches retain the
terms of their upstream projects: FlexGEMM (MIT) and UniK3D
(CC BY-NC-SA 4.0). Upstream model code, checkpoints, diffusion assets, and
datasets retain their own licenses and must be obtained under those terms.
