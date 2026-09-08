# Keep the environments separate

The original NumPy lessons still use the root `requirements.txt`. The connected
lab uses Python 3.11 and PyTorch 2.8.0. Do not replace an existing GPU environment
just to run a CPU lesson.

| Track | Environment | What it exercises |
| --- | --- | --- |
| CPU | Windows or Linux, Python 3.11, CPU PyTorch | Tiny transformer, INT8 linear layers, scoring, cache simulator, local server |
| GPU LoRA | Compatible NVIDIA driver and CUDA-enabled PyTorch | Pretrained-model adapter learning and generation |
| GPU QLoRA | GPU LoRA plus compatible bitsandbytes | Frozen NF4 base with higher-precision adapters |
| Multi-GPU | Separate machine/runtime selection | Existing capstone protocols; not an implemented cluster deployment |

## CPU: no model download

From the repository root in PowerShell:

```powershell
python -m venv pipeline\.venv
.\pipeline\.venv\Scripts\python.exe -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r pipeline\requirements-cpu.txt
.\pipeline\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The extra index supplies ordinary Python dependencies missing from the PyTorch
wheel index. `--only-binary` avoids unexpectedly building those dependencies.
This installation downloads software packages, not pretrained model weights.
On Linux, use the virtual environment's `bin/python` in place of
`Scripts\python.exe`.

To run the offline HF regression tests or the optional pretrained-model path:

```powershell
.\pipeline\.venv\Scripts\python.exe -m pip install -r pipeline\requirements-hf.txt
.\pipeline\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

CPU INT8 quantization is an x86 teaching path here. It is not a CUDA INT8
implementation or a claim that another CPU architecture selects identical kernels.

## CUDA: a different virtual environment

The following is a pinned CUDA 12.6 wheel recipe, not a driver installer:

```powershell
python -m venv pipeline\gpu\.venv
.\pipeline\gpu\.venv\Scripts\python.exe -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple -r pipeline\requirements-cpu.txt
.\pipeline\gpu\.venv\Scripts\python.exe -m pip install -r pipeline\requirements-hf.txt
.\pipeline\gpu\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Despite its filename, `requirements-cpu.txt` pins the framework version; the
index selects the CPU or CUDA build. Stop if the last command reports CUDA
unavailable. Do not silently fall back to CPU and label the result GPU training.
Check driver compatibility against the official
[PyTorch installation recipes](https://pytorch.org/get-started/previous-versions/)
and [CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/).

For QLoRA only:

```powershell
.\pipeline\gpu\.venv\Scripts\python.exe -m pip install -r pipeline\requirements-qlora.txt
```

bitsandbytes support depends on OS, GPU architecture, CUDA build, and operation.
Native Windows support for one operation does not establish support for another.
Use a supported Linux/WSL2 environment if the needed operation is unsupported;
WSL2 has its own Python environment, not the Windows virtual environment above.
Consult [bitsandbytes hardware compatibility](https://huggingface.co/docs/bitsandbytes/installation)
before changing the system driver or installing another toolkit.

On a 6 GiB device, begin with a small text checkpoint, one example per microbatch,
short sequences, and a single optimizer step. A suggested 0.5B checkpoint is
not a guarantee that every configuration fits. Record actual allocated and
reserved peaks.

## Reproduction record

The manifests pin direct dependencies, not every transitive package.
After creating a working environment, record its complete resolution next to
the experiment, not in a global configuration:

```powershell
.\pipeline\.venv\Scripts\python.exe -m pip freeze > results\pipeline\environment.txt
```

Also record checkpoint commit SHA, tokenizer/template, dataset digest, model
configuration, device, driver, effective precision, and command line. A package
freeze without model/data identity does not reproduce a training run.

No model weights, optimizer checkpoints, caches, or new result files belong in
the repository. Keep them under `results`, `models`, or `artifacts`. Models and
teacher outputs have separate license and data-use requirements.
