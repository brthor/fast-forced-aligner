# Corrected C++ comparator

`cpu_corrected.cpp` adapts [MahmoudAshraf97/ctc-forced-aligner at revision 64293cc6d711e57666c4a8b098e9fd93b381fd88](https://github.com/MahmoudAshraf97/ctc-forced-aligner/blob/64293cc6d711e57666c4a8b098e9fd93b381fd88/ctc_forced_aligner/forced_align_impl.cpp).
The unchanged source has SHA-256 `0f6410d89e5c1d3c1ee226d6b024fabc82f16a99baa0fe8defdf97f9875d2804`.
Its BSD-2-Clause license is retained in `LICENSE.ctc-forced-aligner`, along with the source's Flashlight attribution.

`cpu_corrected.patch` describes the complete adaptation:

- The predecessor selector chooses an actual maximum when advancing one or two states ties above staying. Ties prefer staying, then advancing one, then skipping two states.
- Native recurrence and traceback release the Python GIL, allowing independent recordings to execute in a persistent thread pool.
- Traceback stops after writing frame zero, avoiding the upstream predecessor read that can underflow the traceback index.
- Native guards reject unsafe dimensions, token/blank indices, transcript lengths, and signed 32-bit loop overflows.
- Standard library includes and a separate extension name make the adaptation independently buildable.

The FP32 recurrence, rolling score rows, reachability windows, packed two-bit traceback, and score recovery otherwise remain upstream code. This lane is explicitly identified as corrected; it is not an unchanged upstream result. The original lane remains available separately.

Both lanes admit nonempty transcripts with `T >= U + adjacent_repeats + 2`. This conservative shared domain protects the original implementation. It does not imply that standard CTC requires two spare frames. Inputs containing no finite path are not established as supported by these C++ implementations.

The threaded lane submits one record per task. Worker startup is warmed up, while dispatch, result collection, layout preparation, allocation and native execution are timed. All submitted futures are drained before a worker exception is raised. Every build records its source/patch hashes, compiler, flags, Python ABI and PyTorch version beneath `BENCHMARK_CACHE/build`.
