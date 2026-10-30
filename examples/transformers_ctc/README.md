# Stock Transformers CTC model → CUDA alignment

This example installs `fast_forced_aligner` from
[GitHub](https://github.com/brthor/fast-forced-aligner) with pip inside the official
PyTorch 2.8 / CUDA 12.8 **development** image, then uses the standard Transformers
[`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h)
model to produce CTC emissions. It streams a few audio/transcript pairs from
[LibriSpeech's small example dataset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy),
aligns them on the GPU, checks the returned paths, and prints alignment rows/s.

## Run

Clone the repository to get the example files:

```bash
git clone https://github.com/brthor/fast-forced-aligner.git
cd fast-forced-aligner
./examples/transformers_ctc/run.sh build
./examples/transformers_ctc/run.sh --rows 4
./examples/transformers_ctc/run.sh --rows 4 --mode conventional
```

Docker with NVIDIA GPU support is required. First use downloads the model and
sample data and compiles the alignment extension; subsequent runs reuse caches
under `.local/transformers-example/`. The runner uses your UID/GID. An existing
compatible PyTorch development image can be selected with `PYTORCH_IMAGE` at
build time; `RUNTIME_IMAGE` selects the output/runtime image name and `GPU_DEVICE`
selects the GPU. The default base image supplies `nvcc` and the C++ toolchain.

The aligner installs from the repository's default branch. Set `ALIGNER_REF` to
a full commit SHA or tag when building to select a specific revision:

```bash
ALIGNER_REF=FULL_COMMIT_SHA ./examples/transformers_ctc/run.sh build
```

Replace `FULL_COMMIT_SHA` with an actual commit.

The relevant Dockerfile installation steps are simply:

```dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
WORKDIR /opt/example
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY examples/transformers_ctc/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
ARG ALIGNER_REF=HEAD
RUN python -m pip install --no-cache-dir \
    "fast_forced_aligner @ git+https://github.com/brthor/fast-forced-aligner.git@${ALIGNER_REF}"
COPY examples/transformers_ctc/align.py ./align.py
ENTRYPOINT ["python", "align.py"]
```

The checked-in Dockerfile additionally pins the base image digest and verifies
the PyTorch version. Pip installs the core package directly from GitHub until a
PyPI release is available. No editable install or source-tree `PYTHONPATH` is used.
The example installs only the core aligner plus stock Transformers/data-reading
dependencies. Acoustic inference uses the stock model's forward pass.

## Read the code

[`align.py`](align.py) contains the complete flow. It loads `AutoProcessor` and
`AutoModelForCTC`, uses dataset transcripts as the alignment targets, and computes
`model(**inputs).logits[0].float().log_softmax(-1)`. Blank comes from the model's
`pad_token_id`; transcript IDs come from its tokenizer. Wav2Vec2 expects mono
16 kHz audio and this checkpoint's character targets are uppercase.

Acoustic inference runs one recording at a time so each emission tensor has its
true frame length without padding. The alignment kernel receives batches of
these variable-length tensors. `--batch-size` controls alignment batch size;
the final partial batch is also processed. The supplied demo dataset has 73
records, so larger `--rows` values stop at its end.

Timing covers the `align(...)` call and final CUDA synchronization, including
default input admission checks, preparation, and native outputs. Model inference,
initial device placement, data loading, compilation, and output inspection are
outside that timer. This small usage example is not a statistically controlled
benchmark. The reported rows/s describes only the alignment pass.

To use your own data, replace the dataset iterator and provide mono 16 kHz
waveforms and known transcripts. To inspect token spans, split each output path
into equal-label runs, remove blank runs, and multiply run start/end frames by
`model.config.inputs_to_logits_ratio / 16000` seconds. This is a frame-grid
timestamp convention, not a guarantee of human boundary accuracy.
