ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6
FROM ${PYTORCH_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MAX_JOBS=2
WORKDIR /opt/example

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY examples/transformers_ctc/requirements.txt ./requirements.txt
RUN python -c 'import torch; assert torch.__version__.split("+")[0] == "2.8.0"' && \
    python -m pip install --no-cache-dir -r requirements.txt

# Install the published repository; set ALIGNER_REF to a full commit SHA or tag.
ARG ALIGNER_REF=HEAD
RUN python -m pip install --no-cache-dir \
    "fast_forced_aligner @ git+https://github.com/brthor/fast-forced-aligner.git@${ALIGNER_REF}" && \
    python -c 'from fast_forced_aligner import align'

COPY examples/transformers_ctc/align.py ./align.py
ENTRYPOINT ["python", "align.py"]
