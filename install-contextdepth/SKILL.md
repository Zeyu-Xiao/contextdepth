---
name: contextdepth-install
description: Install the ContextDepth inference and outpainting runtime from this repository, including pinned upstream checkouts, depth checkpoints, retrieval weights, and the separate generation environment.
---

# ContextDepth installation

Use this skill when an agent needs to make the ContextDepth checkout runnable.
Work from the repository root. Read the root `README.md` for the input manifest
and run commands; this skill is only the environment and asset setup.

## Boundary and safety

- Do not put credentials, access tokens, local absolute paths, or downloaded
  weights into source control.
- Do not modify a user's existing environment or replace a checkout at a
  different revision. Prefer the two new environment names below.
- The scripts stop on revision, patch, or SHA-256 mismatches. Do not bypass a
  mismatch by deleting files or changing the expected hashes.
- The full FLUX/OneReward generation path is GPU-memory intensive. Do not
  silently quantize, offload, or change precision if the goal is numerical
  reproduction.

## 1. Pinned upstream sources

From the repository root, inspect and fetch the source trees:

```bash
python3 install-contextdepth/setup_upstreams.py
python3 install-contextdepth/setup_upstreams.py --install
python3 install-contextdepth/setup_upstreams.py --check
```

Use `--group depth` or `--group generation` when only one stage is needed.
Both MoGe versions remain separate source trees; do not editable-install both
under the same `moge` package name. DA3 and Depth Pro import from their `src/`
directories. Only the old `utils3d` helper is installed editable by the depth
environment. The setup script applies the adjacent FlexGEMM Triton-3.3 patch
and UniK3D optional-logging patch idempotently.

## 2. Depth and retrieval environment

The completed iBims depth replay used a restored Linux x86_64 environment on
an NVIDIA GeForce RTX 4090. Its observed runtime is recorded in
`validated-depth-runtime.json`. In particular, it used Anaconda builds of
PyTorch, torchvision and Triton, rather than the previously documented pip
wheel installation. The following core-build recipe is a reconstruction
target; a fresh environment created from it has not yet completed the full
prediction regression. Do not describe matching version strings as proof of
matching binaries or predictions.

Create a separate environment with the observed Conda core builds, then add
the pip dependencies:

```bash
conda create -n contextdepth-depth --override-channels -c defaults -y \
  python=3.12.13=h4d16e0c_1 \
  pytorch=2.7.0=gpu_cuda128_py312he3ce74d_303 \
  torchvision=0.22.1=cuda128py312h3a63a03_100 \
  triton=3.3.0=cuda128py312h87d881f_1 \
  pillow=12.1.1=py312h8263a33_0 \
  jpeg=9f=h5ce9db8_0 \
  cudnn=9.17.0.29=h1769b22_0
conda activate contextdepth-depth
python -m pip install setuptools==80.10.2 wheel
python -m pip install -r install-contextdepth/requirements-depth.txt
python -m pip install --no-deps -e third_party/utils3d
```

Do not substitute pip PyTorch/torchvision wheels for these Conda builds and
call the resulting environment the validated one. The active runtime used
NumPy 2.4.4, SciPy 1.17.1, Pillow 12.1.1, Transformers 4.52.3 and
`opencv-python==4.11.0.86`. It did not contain `opencv-python-headless`.
Install only one OpenCV distribution in this environment. A server missing
OpenCV's system shared libraries must resolve those dependencies and pass
the import/runtime check; silently switching OpenCV variants is not the
validated reproduction path.

Conda metadata may describe packages that were subsequently overlaid by pip.
For example, the restored environment had an old NumPy 2.4.3 Conda record
while the active imported NumPy was 2.4.4. The runtime check uses the imported
NumPy version and does not turn that stale Conda record into an installation
requirement. The recipe above is not a complete lock for every transitive
binary dependency.

For MoGe3, the validated compiler is `ptxas V12.8.93`, not any compiler with
the label CUDA 12.8. Download its pinned Linux x86_64 wheel, then use the
offline helper to verify both the archive and extracted executable:

```bash
python -m pip download --no-deps --only-binary=:all: \
  nvidia-cuda-nvcc-cu12==12.8.93 --dest tools/compiler
python install-contextdepth/prepare_compiler.py \
  --wheel tools/compiler/nvidia_cuda_nvcc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
```

Then explicitly point Triton at the selected compiler in every shell that runs
MoGe3 (or put the same absolute path in the local `models.yaml` `ptxas` field):

```bash
export TRITON_PTXAS_PATH="$PWD/tools/cuda-12.8/bin/ptxas"
"$TRITON_PTXAS_PATH" --version
python install-contextdepth/verify_runtime.py --ptxas "$TRITON_PTXAS_PATH"
```

The wheel SHA-256 is
`2d6dc36fb7cb5ac9c0b8825bc13d193c35487a315664007287d0126531238011`;
the executable SHA-256 is
`c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca`.
The extraction helper refuses to overwrite a different existing executable.
The runtime check inspects the compiler, imported versions, observed core
Conda build records, CUDA/cuDNN and decoder information. A passing check is
a necessary identity check for this reproduction target, not an inference
test or a claim of cross-GPU bitwise determinism.

Keep `FLEX_GEMM_AUTOTUNE_MODE=never`; the runtime enforces the same setting.
Do not add upstream training extras such as xFormers, flash-attn, or W&B.

Use a fresh process with `PYTHONNOUSERSITE=1` and the intended environment's
explicit `bin/python`. The replay launcher used `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2` and
`TOKENIZERS_PARALLELISM=false`. Model-specific numerical settings remain
owned by the adapters; do not impose a new global precision/determinism mode
to force agreement. Preserve the pinned source revisions, checkpoint hashes,
camera settings, candidate pixels and per-input DA3 RNG state.

### JPEG decoding and ROI identity

For the frozen iBims cohort, decode the official JPEG with OpenCV, convert
BGR to RGB, crop using the exact specification, and resize the RGB crop with
Pillow LANCZOS. OpenCV in the validated environment reports
`build-libjpeg-turbo (ver 3.0.3-70)`. Pillow reports JPEG API 9.0 without
libjpeg-turbo; using that Pillow build to decode the JPEG directly failed
the historical query hash for all 100 audited images. OpenCV decoding passed
for all 100. Pillow remains the resize implementation.

Use the existing `run.py prepare` entry point and preserve both RGB SHA-256
checks. A library version pin does not replace those pixel checks. The frozen
specification and reference/candidate/RNG assets remain external inputs; the
blind source package does not contain them. Rebuilding query pixels alone
does not regenerate the same diffusion candidates.

Prepare the seven depth checkpoints and the SigLIP2 vision tower:

```bash
python install-contextdepth/prepare_weights.py depth
python install-contextdepth/prepare_weights.py depth --check-only
hf download google/siglip2-so400m-patch16-384 \
  --local-dir weights/siglip2-so400m-patch16-384-original \
  --include '*.safetensors' '*.json'
python install-contextdepth/prepare_weights.py siglip2 \
  --source weights/siglip2-so400m-patch16-384-original
```

Hugging Face authentication and terms acceptance, when required, belong to the
user's local credential store. The helper verifies every expected model/tensor
hash. Use a fresh weights directory: a download can replace an existing
configuration file before validation. Preserve any existing custom files
outside the download destination before running the helper.

## 3. Generation environment

The outpainting stage uses the original OneReward/FLUX.1-Fill-dev pipeline in
a separate Python environment:

```bash
conda create -n contextdepth-generation python=3.10 pip -y
conda activate contextdepth-generation
python -m pip install setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r install-contextdepth/requirements-generation.txt
```

After accepting the upstream access terms and logging in locally, download the
required files and prepare them:

```bash
hf download black-forest-labs/FLUX.1-Fill-dev \
  --local-dir weights/FLUX.1-Fill-dev \
  --include model_index.json 'scheduler/*' 'tokenizer/*' 'tokenizer_2/*' \
  'text_encoder/*' 'text_encoder_2/*' 'vae/*' 'LICENSE*' POLICY.md README.md
hf download bytedance-research/OneReward \
  --local-dir weights/OneReward-original \
  --include 'flux.1-fill-dev-OneReward-transformer/*' 'LICENSE*' README.md
python install-contextdepth/prepare_weights.py flux-base
python install-contextdepth/prepare_weights.py onereward \
  --source weights/OneReward-original/flux.1-fill-dev-OneReward-transformer
```

The BF16 conversion is CPU-side and writes `weights/OneReward/diffusers-bf16`.
The full resident pipeline needs a large-memory CUDA device; do not silently
switch to an offloaded or quantized variant when comparing numbers.

## 4. Installation check

From each environment, run the relevant setup checks and then follow the root
README's `run.py` examples. Do not upgrade PyTorch or replace the observed
Conda build to resolve a pip metadata warning. For the depth environment:

```bash
python install-contextdepth/setup_upstreams.py --check
python install-contextdepth/prepare_weights.py depth --check-only
python install-contextdepth/verify_runtime.py --ptxas "$TRITON_PTXAS_PATH"
python run.py --help
```

After the required sources and weights are present, run one small manifest
through `infer --prepare-only`, then either `generate` the three candidates or
`import-assets` from an archived candidate bundle, and finally run
`run --skip-generation` on a single GPU before launching a full cohort. This
catches path, camera, candidate-shape, and CUDA-extension errors without
changing the numerical protocol.

The completed full iBims replay reused archived outpaintings with
`--skip-generation`: 100 queries x 12 configurations, with all 4,800 direct
and candidate depth arrays equal to the historical arrays. An independent
input audit reproduced all 100 fixed/query RGB pairs from official JPEGs.
All paper-display metrics agree, while five of 24 raw aggregate floats differ
by one ULP. Other datasets, newly generated diffusion candidates and a fresh
environment reconstructed from the recipe above have not passed this same
full-cohort gate. Keep those claims separate from the verified restored
environment result.

If a checkpoint, upstream revision, CUDA compiler, or required external asset
is unavailable, stop with the concrete missing item. Do not invent a fallback
model or claim paper-level numerical reproduction without the archived inputs.
