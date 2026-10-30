# Measured benchmark results

**GPU:** NVIDIA GeForce RTX 3090, 24 GiB

**CPU:** AMD Ryzen 7 5800XT, 8 cores / 16 threads

Mean total alignment time across ten workloads (3,553 entries), with a batch
limit of 64 and six measurements per workload. Per-call input preparation is
included; acoustic model inference and initial device placement are excluded.
All measurements use the same frozen package build. NeMo bucketed rows are
fresh confirmations of the selected schedule; the input-order controls remain
listed separately. Sorting and restoring output order are timed.

| Implementation | Mean total time |
| --- | ---: |
| Ours optimized, input checks disabled | 30.44 ms |
| Ours conventional DP, input checks disabled | 58.91 ms |
| ctc-forced-aligner, corrected CPU / 8 workers | 349.97 ms |
| ctc-forced-aligner, upstream CPU | 1,782.99 ms |
| TorchAudio CPU | 2,694.28 ms |
| TorchAudio CUDA | 6,001.72 ms |
| k2 CUDA | 6,337.72 ms |
| NeMo Viterbi CPU / 8 threads, B64 input order | 41,026.61 ms |
| NeMo Viterbi CUDA, B64 input order | 59,420.43 ms |
| NeMo Viterbi CPU / 8 threads, B64 length-bucketed | 18,892.84 ms |
| NeMo Viterbi CUDA, B64 length-bucketed | 28,182.72 ms |

## CPU worker comparison

Fresh measurements on the same CPU, ten workloads, batch limit of 64, and six
measurements per workload. Sixteen workers can use all 16 logical CPUs; the
container has no CPU quota.

| Corrected ctc-forced-aligner | Mean total time | Recordings/second |
| --- | ---: | ---: |
| 8 workers | 355.21 ms | 10,002 |
| 16 workers | 332.64 ms | 10,681 |
