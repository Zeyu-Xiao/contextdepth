# Installation

By default, normal use needs only one `contextdepth` environment: OneReward
handles outpainting and MoGe2 handles depth prediction. The reference bank is
disabled by default, so neither DIODE nor the SigLIP weights are needed. Install
other depth models as needed. See [reproduction.md](reproduction.md) for the
evaluation commands. Run all commands below from the repository root. The
source and weight paths match the default `models.yaml`.

## 1. Create the Unified Environment

Linux x86_64, Python 3.12, and an NVIDIA driver that supports CUDA 12.8 are
recommended. Install the matching PyTorch / torchvision first, then the
inference dependencies:

```bash
conda create -n contextdepth python=3.12 pip -y
conda activate contextdepth
python -m pip install setuptools==80.10.2 wheel
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r install-contextdepth/requirements.txt
```

If you have Python 3.12 but not Conda, you can instead run
`python3.12 -m venv .venv` and `source .venv/bin/activate`, then run the same
pip commands. Use a new environment; do not install these dependencies over
an environment you already use for something else. The CUDA wheels bundle their runtime libraries, and a normal
OneReward + MoGe2 installation does not require compiling any CUDA extensions.

The unified combination uses `diffusers==0.35.1`, `transformers==4.52.3`, and
`numpy==1.26.4`. It satisfies the `diffusers>=0.35.0` and
`transformers>=4.51.3` requirements in the official OneReward documentation, as
well as the PyTorch requirement of the MoGe2 model. What is installed here is
the set of inference dependencies that ContextDepth calls. There is no need to
install each upstream project's full training, web demo, or data processing
environment.

## 2. Prepare the Default Model Sources and Check Imports

```bash
python install-contextdepth/setup_upstreams.py --install --group core
python -m pip install --no-deps -e third_party/utils3d
python install-contextdepth/setup_upstreams.py --check --group core
python install-contextdepth/check_installation.py
```

`core` prepares only the pinned versions of OneReward, MoGe2, and utils3d. The
adapters load MoGe2 and OneReward from the source paths in `models.yaml`, so
the full upstream packages do not need to be installed separately. The check
script imports the real `FluxFillCFGPipeline`, `FluxTransformer2DModel`, and
`MoGeModel` in a single Python process and runs small CPU operations. It does
not download or load weights, and it does not use the GPU.

## 3. Prepare the Default Model Weights

MoGe2:

```bash
python install-contextdepth/prepare_weights.py depth --model moge2
```

OneReward uses the base components of FLUX.1-Fill-dev and a separate OneReward
transformer. First accept the access terms of the corresponding models on
Hugging Face and log in with `hf auth login`, then download:

```bash
hf download black-forest-labs/FLUX.1-Fill-dev --local-dir weights/FLUX.1-Fill-dev \
  --include model_index.json 'scheduler/*' 'tokenizer/*' 'tokenizer_2/*' \
  'text_encoder/*' 'text_encoder_2/*' 'vae/*' 'LICENSE*' POLICY.md README.md
hf download bytedance-research/OneReward --local-dir weights/OneReward-original \
  --include 'flux.1-fill-dev-OneReward-transformer/*' 'LICENSE*' README.md
python install-contextdepth/prepare_weights.py flux-base
python install-contextdepth/prepare_weights.py onereward \
  --source weights/OneReward-original/flux.1-fill-dev-OneReward-transformer
```

The conversion script saves the original OneReward weights as BF16 on the CPU
and verifies the tensor contents. The output is
`weights/OneReward/diffusers-bf16`. The script does not overwrite an existing
output directory. Downloading and converting require substantial disk space and
memory, and the full BF16 generation model also requires substantial GPU
memory. If GPU memory is limited, you can configure the following explicitly in
a local YAML:

```yaml
outpainting:
  source: third_party/OneReward
  base_model: weights/FLUX.1-Fill-dev
  weights: weights/OneReward/diffusers-bf16
  options:
    offload: model
```

The offload configuration above applies to the normal-use
`ContextDepthPipeline`. It uses Accelerate's model CPU offload, which requires
ample host memory and adds transfer time. It does not guarantee that every
image size fits in GPU memory. See the [README](../README.md) for the normal
inference commands.

## 4. Optional: DIODE Reference Bank

The normal-use entry point supports only the optional DIODE reference bank.
Download the data yourself from the
[official MoGe release](https://huggingface.co/datasets/Ruicheng/monocular-geometry-evaluation):

```bash
hf download Ruicheng/monocular-geometry-evaluation DIODE.zip \
  --repo-type dataset --local-dir data/downloads
unzip -q data/downloads/DIODE.zip -d data/moge_eval
```

Enabling the reference bank also requires the
[SigLIP2 retrieval weights](https://huggingface.co/google/siglip2-so400m-patch16-384):

```bash
hf download google/siglip2-so400m-patch16-384 \
  --local-dir weights/siglip2-so400m-patch16-384-original \
  --include '*.safetensors' '*.json'
python install-contextdepth/prepare_weights.py siglip2 \
  --source weights/siglip2-so400m-patch16-384-original
python install-contextdepth/check_installation.py --with-bank
```

SigLIP uses the same environment, so no separate retrieval environment is
needed. The output path of the vision tower is
`weights/siglip2-so400m-patch16-384-vision`. See the [README](../README.md) for
the DIODE directory layout and how to enable it.

## 5. Optional: Other Built-in Depth Models

Add the inference dependencies and sources, still in the same environment:

```bash
python -m pip install -r install-contextdepth/requirements-extra-depth.txt
python install-contextdepth/setup_upstreams.py --install --group depth
```

Prepare the weights you need. `--model` can be repeated:

```bash
python install-contextdepth/prepare_weights.py depth --model unidepth --model depth_pro
```

The available choices are `moge2`, `moge3`, `da3`, `da3_metric`, `unik3d`,
`unidepth`, `depth_pro`, and `depth_pro2`. Omitting `--model` downloads the
checkpoints of every built-in model (about 23 GB in total). Add `--check-only` to only check local
files. MoGe3 also requires `ptxas 12.8.93`. Run the following from the
repository root:

```bash
python -m pip download --no-deps --only-binary=:all: \
  nvidia-cuda-nvcc-cu12==12.8.93 --dest tools/compiler
python install-contextdepth/prepare_compiler.py \
  --wheel tools/compiler/nvidia_cuda_nvcc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl
export TRITON_PTXAS_PATH="$PWD/tools/cuda-12.8/bin/ptxas"
```

Set the environment variable above in the terminal that runs MoGe3, or put the
compiler path in `models.moge3.ptxas` in your local YAML.

Do not editable-install both MoGe2 and MoGe3 as the same `moge` package, and do
not overwrite this environment with a full upstream `requirements.txt`: the
NumPy requirements declared by some training environments conflict with each
other. The built-in adapters read their respective sources directly and defer
imports of training logging components that inference does not need.

## Local Path Configuration

By default, the normal-use interface reads `models.yaml` in the repository
root, and evaluation reads `evaluation/models.yaml`. If you already have the
sources and weights, create `models.local.yaml` and fill in only the resource
paths you need to override:

```yaml
models:
  moge2:
    source: /path/to/MoGe2
    weights: /path/to/moge-2-vitl-normal/model.pt
outpainting:
  source: /path/to/OneReward
  base_model: /path/to/FLUX.1-Fill-dev
  weights: /path/to/OneReward/diffusers-bf16
```

The normal-use interface takes
`ContextDepthPipeline(config="models.local.yaml")`. The normal-use CLI and the
evaluation CLI both take `--config models.local.yaml`. Relative paths are
resolved against the directory containing the YAML file. Entries you leave out
fall back to the default configuration of the respective entry point.

## Verification Scope

What has been run, so that you know what to trust:

- `check_installation.py` and the unit tests pass on Python 3.12 with the
  package versions of `requirements.txt` (PyTorch 2.7.0, CUDA 12.8 build).
  Creating the environment from scratch with the commands above has not been
  re-run since Depth Pro 2 and its `av` dependency were added.
- In an environment with these versions, every built-in depth model was run on
  GPU through `python -m evaluation`, on images from all six benchmarks, and
  compared with the predictions of the paper's runs. See
  [reproduction.md](reproduction.md) for the protocol.
- OneReward generation was run in the same environment through both entry
  points, from a clean copy of the repository on an RTX 4090 with 48 GB:
  `run.py infer` on a single image (peak 35 GB of GPU memory, 24.4 GB with
  `--offload model`) and `python -m evaluation` on the quick subset.

Depth Pro 2 imports its video utilities at load time, so it needs the `av`
package, which `requirements-extra-depth.txt` installs.

The exact binaries of the paper's runs (Conda builds, a separate generation
environment) are described in [paper-environment.md](paper-environment.md).
They are not needed to use the code or to run the evaluation.