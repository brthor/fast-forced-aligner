"""Measure complete alignment calls on a synthetic, already-resident dataset.

This is a usage example, not a reproduction of the published speech panels.
"""
import argparse
import statistics
from time import perf_counter

import torch
from fast_forced_aligner import align, load_extension, validate_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--check-inputs", action="store_true", help="also repeat admission validation inside each timed call")
    args = parser.parse_args()
    if min(args.rows, args.batch_size, args.repeats) <= 0:
        parser.error("rows, batch-size, and repeats must be positive")
    torch.manual_seed(7)
    torch.set_num_threads(1)
    emissions = [torch.randn(400 + i % 80, 32, device="cuda").log_softmax(-1) for i in range(args.rows)]
    targets = [torch.randint(1, 32, (60,), device="cuda") for _ in emissions]
    validate_inputs(emissions, targets)
    load_extension()

    def run(mode):
        # Keep complete outputs alive through the final synchronization.
        results = [align(emissions[i:i + args.batch_size], targets[i:i + args.batch_size],
                         mode=mode, check_inputs=args.check_inputs)
                   for i in range(0, args.rows, args.batch_size)]
        torch.cuda.synchronize()
        return results

    times = {"optimized": [], "conventional": []}
    for mode in times:
        run(mode)
    for repeat in range(args.repeats):
        for mode in (tuple(times) if repeat % 2 == 0 else tuple(reversed(times))):
            start = perf_counter()
            results = run(mode)
            times[mode].append(perf_counter() - start)
            for result in results:
                result.raise_for_status()
            del results
    print(f"Synthetic rows={args.rows}, batch={args.batch_size}, repeated admission={args.check_inputs}")
    for mode, samples in times.items():
        mean = statistics.mean(samples)
        print(f"{mode}: {mean * 1000:.3f} ms/dataset, {args.rows / mean:.1f} rows/s")


if __name__ == "__main__":
    main()
