# Build from the publish directory: docker build -f benchmarks/Dockerfile .
ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6
FROM ${PYTORCH_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MAX_JOBS=2 \
    BENCHMARK_CACHE=/cache TORCH_EXTENSIONS_DIR=/cache/extensions \
    HOME=/cache/home MPLCONFIGDIR=/cache/matplotlib
WORKDIR /opt/benchmarks

COPY benchmarks/requirements.txt /opt/benchmarks/requirements.txt
RUN python -c 'import sys, torch; assert sys.version_info[:2] == (3, 11); assert torch.__version__.split("+")[0] == "2.8.0"; assert torch.version.cuda == "12.8"' && \
    python -m pip install --no-cache-dir -r requirements.txt

# The core package is installed through pip, rather than imported from a checkout.
COPY pyproject.toml README.md LICENSE NOTICE /opt/package/
COPY benchmarks/patches/LICENSE.ctc-forced-aligner /opt/package/benchmarks/patches/
COPY src /opt/package/src
RUN python -m pip install --no-cache-dir /opt/package && \
    python -c 'import fast_forced_aligner, torchaudio, k2, matplotlib; print(fast_forced_aligner.__file__)'

COPY benchmarks /opt/benchmarks
# Sources are pinned and verified again by the adapters before execution.
# Atomic downloads are private to the build user until made readable for runtime.
RUN BENCHMARK_CACHE=/opt/benchmark-seed python upstream.py && \
    chmod -R a+rX /opt/benchmark-seed
CMD ["python", "benchmark.py", "--help"]
