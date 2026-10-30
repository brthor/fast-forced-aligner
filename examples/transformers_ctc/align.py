"""Stock Transformers Wav2Vec2 emissions -> batched CUDA forced alignment."""

import argparse
from io import BytesIO
from itertools import islice
from time import perf_counter

from datasets import Audio, load_dataset
import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor

from fast_forced_aligner import align, load_extension

MODEL = "facebook/wav2vec2-base-960h"
MODEL_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
DATASET_REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mode", choices=("optimized", "conventional"), default="optimized")
    args = parser.parse_args()
    if min(args.rows, args.batch_size) < 1:
        parser.error("rows and batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Run with an NVIDIA GPU, for example docker run --gpus all")

    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = AutoModelForCTC.from_pretrained(
        MODEL, revision=MODEL_REVISION, use_safetensors=True,
    ).to("cuda").eval()
    blank = model.config.pad_token_id
    load_extension()  # Compile once before timing alignment.
    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation",
        revision=DATASET_REVISION, streaming=True,
    ).cast_column("audio", Audio(decode=False))
    rows = iter(dataset.take(args.rows))

    processed, alignment_seconds = 0, 0.0
    while batch := list(islice(rows, args.batch_size)):
        emissions, targets = [], []
        for row in batch:
            waveform, sample_rate = sf.read(BytesIO(row["audio"]["bytes"]), dtype="float32")
            if sample_rate != 16000 or waveform.ndim != 1:
                raise ValueError("This example expects mono 16 kHz audio")
            inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").to("cuda")
            # Standard Transformers inference, one recording at a time: no
            # padding to remove and no custom model implementation.
            logits = model(**inputs).logits[0]
            emissions.append(logits.float().log_softmax(-1).contiguous())

            token_ids = processor.tokenizer(row["text"].upper(), add_special_tokens=False).input_ids
            if processor.tokenizer.unk_token_id in token_ids:
                raise ValueError("Transcript contains characters outside the model's vocabulary")
            targets.append(torch.tensor(token_ids, dtype=torch.int64, device="cuda"))

        # Emissions and targets are resident. Count the complete alignment call,
        # including default input checks, preparation, and complete GPU outputs.
        torch.cuda.synchronize()
        started = perf_counter()
        result = align(emissions, targets, blank_id=blank, mode=args.mode)
        torch.cuda.synchronize()
        alignment_seconds += perf_counter() - started
        result.raise_for_status()

        # Check that each full frame path collapses to the supplied transcript.
        # Output inspection/copies are outside the alignment timer.
        for row, path, target in zip(batch, result.paths, targets, strict=True):
            collapsed = path.unique_consecutive()
            if not torch.equal(collapsed[collapsed != blank], target):
                raise RuntimeError("Alignment does not collapse to the supplied transcript")
            print(f"{row['id']}: {path.numel()} aligned frames | {row['text']}")
        processed += len(batch)
        print(f"{args.mode}: {processed} rows | {alignment_seconds * 1000:.2f} ms alignment | "
              f"{processed / alignment_seconds:.1f} alignment rows/s", flush=True)


if __name__ == "__main__":
    main()
