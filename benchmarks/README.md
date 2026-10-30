# Reproduce the alignment benchmarks

This subproject compares CTC alignment implementations on identical, locally cached
emissions and transcripts. It measures the alignment pass, including each
implementation's input preparation, and audits its outputs against the packaged
conventional DP kernel. Acoustic model inference is outside the comparison.

Clone the repository and run the commands below from its root. The Docker image installs
`fast_forced_aligner` through pip and contains the comparator dependencies;
a separate preparation image downloads public audio and stock model checkpoints.
No separately hosted input bundle or private files are required.

## Build and run

Requirements are Linux x86-64, Docker with NVIDIA Container Toolkit, and an NVIDIA
GPU/driver compatible with CUDA 12.8. The pinned image uses Python 3.11, PyTorch
2.8.0, and the CUDA development toolkit. Compilation happens during setup or
warmup, before measurement. The original measurements use an RTX 3090 with
24 GiB; available memory and kernel support on other GPUs can differ.

```bash
git clone https://github.com/brthor/fast-forced-aligner.git
cd fast-forced-aligner
./benchmarks/run.sh build
./benchmarks/run.sh build-prepare

# Download public sources, run stock acoustic models, and cache inputs.
./benchmarks/run.sh prepare
./benchmarks/run.sh verify

# All 11 methods, ten workloads, B64, two rounds, three repetitions per round.
./benchmarks/run.sh run
./benchmarks/run.sh report
```

The default cache is `.local/benchmarks/`, with inputs in `inputs/` and results
in `results/latest/`. Set `BENCHMARK_CACHE` to an absolute directory on another
disk before running commands if needed. Docker runs under your UID/GID and
mounts the checkout read-only. Downloaded sources, compiled extensions, and
results persist in the cache.

For a quick harness check without downloading audio or models:

```bash
./benchmarks/run.sh run --suite synthetic --methods optimized conventional \
    --rounds 1 --repetitions 1 --output ./.local/benchmark-smoke
./benchmarks/run.sh report --input ./.local/benchmark-smoke --output ./.local/benchmark-smoke/report
```

This deterministic synthetic suite checks the harness; it does **not** reproduce
the README's aggregate. An explicit subset of methods is labeled as a subset.
The conventional reference requires CUDA even when the selected comparators
run on the CPU.

## Regenerate the inputs

The default `regenerated-readme` suite follows the versioned public selection in
[manifests/regeneration.json](manifests/regeneration.json): **3,553 entries across
ten workloads**. [regenerate.py](regenerate.py) downloads the selected public
sources, decodes/resamples audio to 16 kHz, tokenizes transcripts, runs stock
acoustic inference through unmodified public model forward methods, and writes
FP32 emissions and int64 targets. Generation does not import the alignment package.

| Source | Workloads | Stock acoustic model |
| --- | --- | --- |
| [LibriTTS](https://huggingface.co/datasets/mythicinfinity/libritts_opus) | Natural 64/1,024; 64 approximately 60-second concatenations | [Jbetker Wav2Vec2](https://huggingface.co/jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli) via Transformers |
| [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) | 2,048 character and 256 subword examples | [Facebook Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) via Transformers; [NVIDIA Conformer CTC](https://huggingface.co/nvidia/stt_en_conformer_ctc_small) via NeMo |
| [Earnings22](https://github.com/revdotcom/speech-datasets) | Three continuous-long panels, 81 recordings total | Facebook Wav2Vec2 via Transformers |
| Seeded synthetic inputs | Long frame sequences and tied scores, 16 entries | No acoustic model |

The recipe pins source/model revisions, recording IDs, transcript normalization,
clip boundaries, and cohort order. Existing historical selection exclusions are
explicit; generation never selects replacements based on alignment outcomes.
Some Earnings22 transcripts are sparse because the source timestamps are
incomplete; these inputs remain in the suite. Workloads intentionally overlap.

[Dockerfile.prepare](Dockerfile.prepare) keeps stock Transformers and NeMo
inference dependencies separate from the comparator environment. Preparation
runs outside every alignment timer. It can take substantially longer than the
benchmark and downloads several GB of public audio and model weights; allow
additional disk space for extracted audio and cached arrays. Downloads and
completed records are cached, so repeating `prepare` resumes preparation. The
complete manifest is published only after all requested records succeed.

The preparation image installs the complete Python environment from
[requirements-prepare-lock.txt](requirements-prepare-lock.txt), captured from
the image used for successful full-suite regeneration. Every installed Python
distribution, including transitive dependencies and packages supplied by the
base image, has an exact version. Installation disables dependency resolution
and runs `pip check` plus stock-model import checks. The lock targets the
digest-pinned Linux x86-64 / Python 3.11 base in `Dockerfile.prepare`; it is not
a standalone environment specification for arbitrary Python installations.

To update the lock, resolve the edited
[direct requirements](requirements-prepare.txt) in a disposable image using the
same base, validate stock-model execution, and capture
`python -m pip list --format=freeze` inside that image. Replace the lock's package
lines, retain its provenance header, review the version changes, and rebuild
with `build-prepare`. Ordinary builds consume the existing lock without updating it.

Every comparator consumes the **same generated emissions and targets**. The
loader verifies the pinned recipe, exact workload membership/order, file hashes,
raw array hashes, dtypes, shapes, and CTC input validity. Different GPUs and
software can produce slightly different emission values; matching the old
bundle's checksum is neither expected nor required. Reports retain recipe and
input fingerprints, while correctness is measured against conventional DP on
those actual generated inputs. The checked-in timings remain historical
measurements; local reports do not overwrite them.

To exercise every workload with a small selection before preparing everything:

```bash
./benchmarks/run.sh prepare --limit 2 --root ./.local/inputs-smoke
./benchmarks/run.sh verify --root ./.local/inputs-smoke --suite regenerated-readme-partial
./benchmarks/run.sh run --inputs ./.local/inputs-smoke --suite regenerated-readme-partial \
    --methods optimized conventional ctc-corrected --rounds 1 --repetitions 1 \
    --output ./.local/regeneration-smoke
```

A limited preparation is explicitly marked **partial**, and the default full
suite rejects it. A small selection still needs each selected model and its
source downloads. `BENCHMARK_GPUS=none ./benchmarks/run.sh prepare` selects CPU
inference when necessary. Use `--root`/`--inputs` to choose a different cache;
explicit host paths are mounted automatically.

### Archived inputs

An existing historical bundle can still be imported with
`prepare --bundle /path/to/bundle.zip` (or `--url HTTPS_URL --sha256 HASH`).
Then explicitly select `verify --suite readme` and `run --suite readme`.
[manifests/readme.json](manifests/readme.json) pins that archive and its original
manifest. This optional compatibility path is not needed for public regeneration.
Use a separate input directory when switching between archived and regenerated
inputs; existing evidence is never silently overwritten.

## Implementations

| Method argument | Implementation |
| --- | --- |
| `optimized` | Packaged optimized kernel, admitted inputs, `check_inputs=False` |
| `conventional` | Packaged full-record FP32 DP, admitted inputs, `check_inputs=False` |
| `optimized-checked` | Optimized kernel with default per-call input validation |
| `conventional-checked` | Conventional kernel with default per-call input validation |
| `ctc-corrected` | Corrected ctc-forced-aligner C++, eight persistent workers by default |
| `ctc-cpp` | Unchanged ctc-forced-aligner C++, serial execution |
| `ta-cpu` / `ta-cuda` | TorchAudio 2.8.0 `functional.forced_align`, one recording per call |
| `k2-cuda` | k2 1.24.4.dev20260625, CTC graph intersection and shortest-path recovery |
| `nemo-cpu` / `nemo-cuda` | Isolated, unmodified NeMo Viterbi function; eight CPU Torch threads by default |

The [Dockerfile](Dockerfile) pins the base image by digest;
[requirements.txt](requirements.txt) pins dependencies and the k2 wheel hash.
[upstream.py](upstream.py) pins and verifies the NeMo function at revision
`9bd9a949554abae9262327840bf72b03f0bd5b74` and ctc-forced-aligner at
`64293cc6d711e57666c4a8b098e9fd93b381fd88`. NeMo model imports and inference are
not part of the comparator environment: its original function is extracted from the
verified source without rewriting its body.

The [corrected C++ comparator](patches/README.md) is explicitly separate from
unchanged upstream. Its source, complete patch, BSD-2-Clause license, and
Flashlight attribution are included. Corrections cover predecessor ties,
frame-zero traceback, input guards, and GIL release for independent-record CPU
parallelism. Both C++ lanes use the same conservative supported input domain;
unsupported inputs are reported as failures rather than silently removed.

## Timing contract

The default comparison gives every method the same ordered recordings and batch limit. Emissions
and targets start resident on that method's appropriate device. A measured
queue runs every recording in a workload through completion, including:

- Per-call layout preparation, padding or graph/topology construction.
- Scratch and output allocation, recurrence, traceback, and result assembly.
- Transfers performed internally by the implementation.
- Synchronization until the complete queue's outputs are ready.

Downloading inputs, initial device placement, one-time input admission,
compilation, warmup, conventional-reference execution, and host copies used only
for auditing are outside the timer. The two `*-checked` lanes additionally time
the public API's asynchronous GPU target checks. Completion synchronization is included
in every GPU lane, independently of those input checks.

k2 receives transcripts as a device-resident ragged tensor. Creating its row
splits, converting token IDs, constructing the CTC graph, and remapping nonzero
blank IDs are all timed; transcripts are not copied through Python host lists.

The default run uses B64, two rounds, and three timed repetitions per round.
Method order reverses in the second round. All six observations remain in the
results. For each method:

```text
total time = sum over workloads(mean of complete-queue times)
rows/second = total suite entries / total time
optimized speedup = comparator total time / optimized total time
```

This is a ratio of aggregate processing times, not the mean of per-workload
speedup ratios. Repeated runs and overlapping workloads are not independent
accuracy samples. An unavailable implementation, unsupported input, invalid
output, or incomplete observation is reported explicitly; an incomplete suite
must not be presented as a successful full-suite speedup.

## Results and correctness

The [basic results report](RESULTS.md) contains the measured timings and CPU/GPU
details and is included in the repository. Detailed reports and raw results stay
local: `.local/`,
`benchmarks/results/`, and `benchmarks/reports/` are excluded from Git, Docker
build contexts, and source distributions.

The checked-in summary includes the device-resident k2 adapter, matched-B64
controls, and fresh confirmations of NeMo's selected bucketed-B64 schedule.
Every row uses the same frozen package build and six measurements per workload.
The main README retains the input-order controls; its chart uses the confirmed
NeMo selections. Pilot timings and interrupted runs are excluded from these
published means.

The summary reports our kernels with target checks disabled. Full-suite timings
for the current asynchronous default validation are pending; use the
`*-checked` lanes to measure them on your hardware.

Each run writes `campaign.json` plus detailed observations under `raw/`.
The `report` command writes `report.md`, `summary.json`, and, for a complete run,
`performance.png` and `performance.svg` in its output directory. Reports include
aggregate timings and throughput, per-workload timings, relative speedups,
correctness summaries, and a rendered performance chart. Reports are separate
artifacts; running the benchmark does not overwrite the repository's README
table or image. Use a new run output directory for each campaign to preserve
previous measurements.

Every measured output is audited after timing against conventional DP. Checks
cover complete CTC collapse to the supplied transcript, frame-label validity,
selected emission scores, exact path agreement, additional objective loss, and
token start/end displacement in emission frames. Objective differences use
exact sums of the supplied binary FP32 values, separating equal-score alternative
paths from better or worse paths. Different paths are retained for inspection.
These metrics do not assess agreement with human timestamps.

The campaign records input fingerprints, source hashes, package/dependency
versions, GPU/driver and CPU information, execution settings, and failures.
Keep `campaign.json`, the raw observations, and the report together locally.
Publish selected timing summaries rather than these diagnostic artifacts, which
can contain local paths and input provenance. GPU UUIDs and process listings are
not collected. The full benchmark suite does not include the README's separate
138-case adversarial accuracy study.

## Adapt the run to your hardware

```bash
# A separate output directory preserves the previous run.
./benchmarks/run.sh run --batch-size 32 --workers 4 --output ./.local/benchmark-b32
./benchmarks/run.sh report --input ./.local/benchmark-b32 --output ./.local/benchmark-b32/report

# Select one visible GPU and increase extension compilation parallelism.
BENCHMARK_GPUS=device=0 BENCHMARK_MAX_JOBS=4 ./benchmarks/run.sh run \
    --output ./.local/benchmark-gpu0
```

The CLI also accepts `--rounds`, `--repetitions`, `--device`, and `--methods`.
`--workers` controls corrected C++ workers and NeMo CPU Torch threads; other
lanes retain their documented execution strategy. Lowering the batch limit can
help with GPU memory, but changes the benchmark configuration and must be
reported. Explicit method selection or different synthetic inputs likewise
produce a different comparison.

### NeMo configuration comparison

The matched B64 result is a controlled configuration, not a claim that every
toolkit achieves its best throughput at that batch size. NeMo pads both the
frame and transcript dimensions. Smaller batches or grouping similar recording
lengths can reduce that work without changing its upstream alignment function.

```bash
# Five full-suite candidates, followed by fresh measurements of the selected
# schedules and the B64 input-order control. This can take substantially longer
# than the default benchmark, especially the batch-one GPU candidate.
./benchmarks/run.sh sweep --output ./.local/nemo-sweep

# Reproduce one particular schedule with the ordinary audited runner.
./benchmarks/run.sh run --methods nemo-cpu nemo-cuda --batch-size 16 \
    --nemo-batch-order length --output ./.local/nemo-b16-length
```

The fixed candidates are input-order batches 1, 16, and 64, and length-bucketed
batches 16 and 64. Bucketing sorts each workload by decreasing frame count, then
transcript length. Sorting, batch assembly, padding, and restoring outputs to
their original order happen inside every timed queue. Initial tensor placement
and admission remain outside. The same recordings and correctness audit apply
to every candidate. No model inference or modified NeMo recurrence is involved.

Each candidate receives a warmup and one pilot measurement per workload by
default. CPU and GPU independently select one global configuration using the
sum of all workload times; the selection never combines per-workload winners.
The selected configurations and the B64 control then receive two rounds of
three fresh measurements per workload. Report these confirmation means, not
the fastest pilot observation. `--pilot-repetitions`, `--rounds`, `--repetitions`,
and `--methods` can narrow or strengthen the sweep. CPU threads stay at the
specified `--workers` value throughout; this is not a thread-count search.

The local `sweep.json` records every candidate, the selections, and confirmations.
Every child directory is a normal campaign with per-record audit artifacts,
usable with `run.sh report`. Any failed candidate, changed input/source, or
incomplete audit aborts the sweep rather than silently disappearing from the
comparison. Configuration selection and confirmation reuse the frozen suite;
they establish performance on this suite, not on held-out production data.

Use an otherwise idle GPU and avoid competing CPU work. Record any contention
that cannot be eliminated. Driver, clock, thermal, thread-count, and memory
differences can affect both absolute time and relative speedups; reproducing the
input suite does not guarantee the same numbers on another machine.

`BENCHMARK_IMAGE` selects the Docker tag, `BENCHMARK_CPU_WORKERS` changes the
default worker count, and `BENCHMARK_NETWORK=none` disables runtime networking
after the image and inputs are available. `./benchmarks/run.sh fetch` verifies
cached upstream sources. `./benchmarks/run.sh test` runs harness unit tests
without GPU access; it is not a substitute for executing and auditing the real
implementations.
