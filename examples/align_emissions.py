"""Align a small, explicit CTC example in both modes."""
import torch
from fast_forced_aligner import align

# Vocabulary: blank=0, a=1, b=2. Transcript: a, a, b.
emissions = torch.tensor([
    [0., -4., -4.], [-4., 0., -4.], [0., -4., -4.],
    [-4., 0., -4.], [-4., -4., 0.], [0., -4., -4.],
], device="cuda").log_softmax(-1)
targets = torch.tensor([1, 1, 2], dtype=torch.int64, device="cuda")
for mode in ("optimized", "conventional"):
    result = align([emissions], [targets], mode=mode)
    result.raise_for_status()
    print(mode, result.paths[0].tolist())
