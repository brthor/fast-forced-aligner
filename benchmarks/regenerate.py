"""Regenerate benchmark inputs from public audio using unmodified stock models.

Preparation is untimed. Each completed record is checksummed and reusable after
an interruption. No alignment outcome is used to choose or discard recordings.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from inputs import Case, _validate_case, sha256

SAMPLE_RATE = 16000
GENERATOR_VERSION = 1


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def decode_audio(payload: bytes, *, end_seconds: float | None = None) -> np.ndarray:
    """Use FFmpeg's stock decoder and resampler; return mono 16 kHz FP32 audio."""
    command = ['ffmpeg', '-v', 'error', '-i', 'pipe:0', '-map', '0:a:0']
    if end_seconds is not None:
        command += ['-t', str(end_seconds)]
    command += ['-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 'f32le', 'pipe:1']
    result = subprocess.run(command, input=payload, capture_output=True, check=True, timeout=180)
    waveform = np.frombuffer(result.stdout, dtype='<f4').copy()
    if waveform.size < 400 or not np.isfinite(waveform).all():
        raise ValueError('decoded audio is too short or contains nonfinite values')
    return waveform


def synthetic_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    """Reproduce the seeded long-frame and flat-tie challenge construction."""
    c = record['challenge']
    frames, tokens, vocabulary = c['frames'], c['tokens'], c['vocabulary']
    if frames < 2 * tokens or tokens < 1 or vocabulary < 16:
        raise ValueError('invalid synthetic dimensions')
    if c.get('pattern', 'cycle') != 'cycle' or c.get('blank_fraction', 0) or c.get('mismatch', False):
        raise ValueError('unsupported synthetic challenge')
    targets = np.resize(np.arange(1, 13, dtype=np.int64), tokens)
    expanded = np.zeros(2 * tokens + 1, dtype=np.int64)
    expanded[1::2] = targets
    labels = expanded[np.minimum(np.arange(frames) * len(expanded) // frames, len(expanded) - 1)]
    rng = np.random.default_rng(record['seed'])
    logits = rng.normal(0.0, 0.15, size=(vocabulary, frames)).T
    logits[np.arange(frames), labels] += c['margin']
    if c['margin'] == 0:
        logits.fill(0.0)
    logits -= logits.max(axis=1, keepdims=True)
    logits -= np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return np.asarray(logits, dtype=np.float32, order='C'), targets, 0


class AudioSources:
    """Download pinned Parquet shards once and decode only selected recordings."""

    def __init__(self, recipe: dict[str, Any], root: Path):
        self.recipe = recipe
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def audio_path(self, dataset: str, segment: dict[str, Any]) -> Path:
        identity = {'dataset': self.recipe['datasets'][dataset],
                    'source_id': segment['source_id'], 'split': segment.get('split')}
        if 'audio_url' in segment:
            identity['end_seconds'] = segment['end_seconds']
        return self.root / (digest(identity) + '.npy')

    def _save(self, path: Path, waveform: np.ndarray) -> None:
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            np.save(handle, waveform, allow_pickle=False)
        temporary.replace(path)
        atomic_json(path.with_suffix('.json'), {'sha256': sha256(path)})

    def _cached(self, path: Path) -> bool:
        metadata = path.with_suffix('.json')
        if not path.exists() or not metadata.exists():
            return False
        if sha256(path) != json.loads(metadata.read_text())['sha256']:
            raise ValueError(f'cached audio digest mismatch: {path.name}')
        return True

    def prepare(self, records: dict[str, dict[str, Any]]) -> None:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        needed: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
        for record in records.values():
            if record.get('kind') == 'synthetic':
                continue
            dataset = record['dataset']
            for segment in record['segments']:
                if not self._cached(self.audio_path(dataset, segment)):
                    needed.setdefault(dataset, {})[(segment.get('split', ''), segment['source_id'])] = segment
        for dataset, segments in needed.items():
            specification = self.recipe['datasets'][dataset]
            if all('audio_url' in segment for segment in segments.values()):
                for segment in segments.values():
                    # Earnings excerpts start at zero and fit within this public
                    # MP3 prefix; never download the entire multihour recording.
                    bound = specification.get('audio_download_limit_bytes', 8 * 1024 * 1024)
                    request = Request(segment['audio_url'], headers={'Range': f'bytes=0-{bound - 1}'})
                    with urlopen(request, timeout=120) as response:
                        payload = response.read(bound)
                    end = float(segment['end_seconds'])
                    audio = decode_audio(payload, end_seconds=end)
                    if len(audio) < round(end * SAMPLE_RATE) - 1:
                        raise ValueError(f'audio prefix does not cover selected endpoint: {segment["source_id"]}')
                    self._save(self.audio_path(dataset, segment), audio)
                continue
            remaining = dict(segments)
            for shard in specification['files']:
                if not any(split == shard['split'] for split, _ in remaining):
                    continue
                path = hf_hub_download(specification['repo_id'], shard['path'],
                                       revision=specification['revision'], repo_type='dataset')
                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(batch_size=32, columns=['id', 'audio']):
                    for row in batch.to_pylist():
                        key = (shard['split'], str(row['id']))
                        if key not in remaining:
                            continue
                        segment = remaining.pop(key)
                        payload = row['audio'].get('bytes')
                        if not isinstance(payload, bytes) or not payload:
                            raise ValueError('pinned dataset must supply embedded audio bytes')
                        self._save(self.audio_path(dataset, segment), decode_audio(payload))
                if not remaining:
                    break
            if remaining:
                raise ValueError(f'{dataset}: selected recordings missing from pinned shards: {list(remaining)[:5]}')

    def waveform(self, record: dict[str, Any]) -> np.ndarray:
        pieces = []
        for segment in record['segments']:
            path = self.audio_path(record['dataset'], segment)
            if not self._cached(path):
                raise ValueError('selected audio has not been prepared')
            audio = np.load(path, allow_pickle=False)
            start = round(float(segment.get('start_seconds', 0)) * SAMPLE_RATE)
            end_seconds = segment.get('end_seconds')
            stop = len(audio) if end_seconds is None else round(float(end_seconds) * SAMPLE_RATE)
            if start < 0 or stop > len(audio) or stop <= start:
                raise ValueError('selected audio boundaries lie outside decoded recording')
            pieces.append(audio[start:stop])
        return np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32)


class StockModel:
    """Use public Transformers/NeMo forward methods, one unpadded recording at a time."""

    def __init__(self, specification: dict[str, Any], device: str):
        import torch
        self.specification = specification
        self.device = device
        self.backend = specification['backend']
        if self.backend == 'transformers':
            from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
            self.model = AutoModelForCTC.from_pretrained(
                specification['repo_id'], revision=specification['revision'],
                    trust_remote_code=False, attn_implementation=specification.get('attention', 'sdpa'), weights_only=True,
            ).to(device=device, dtype=torch.float32).eval()
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                specification.get('tokenizer_repo_id', specification['repo_id']),
                revision=specification.get('tokenizer_revision', specification['revision']),
                trust_remote_code=False,
            )
            # The Jbetker checkpoint uses sample-variance normalization, whereas
            # the Facebook checkpoint uses its stock feature extractor policy.
            self.processor = Wav2Vec2FeatureExtractor(
                feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0,
                do_normalize=specification['normalization'] == 'stock-processor',
                return_attention_mask=False,
            )
            self.blank = int(self.model.config.pad_token_id)
        elif self.backend == 'nemo':
            from huggingface_hub import hf_hub_download
            from nemo.collections.asr.models import EncDecCTCModelBPE
            checkpoint = hf_hub_download(specification['repo_id'],
                                         specification['checkpoint_filename'],
                                         revision=specification['revision'])
            if specification.get('checkpoint_sha256') and sha256(Path(checkpoint)) != specification['checkpoint_sha256']:
                raise ValueError('stock checkpoint digest differs from recipe')
            self.model = EncDecCTCModelBPE.restore_from(checkpoint, map_location=torch.device('cpu'))
            self.model = self.model.to(device=device, dtype=torch.float32).eval()
            self.blank = int(self.model.decoder.num_classes_with_blank) - 1
        else:
            raise ValueError(f'unsupported stock backend: {self.backend}')
        if self.blank != specification['blank_id']:
            raise ValueError('stock checkpoint blank ID differs from recipe')

    def __call__(self, audio: np.ndarray, transcript: str) -> tuple[np.ndarray, np.ndarray, int]:
        import torch
        with torch.inference_mode():
            if self.backend == 'transformers':
                ids = self.tokenizer(transcript, add_special_tokens=False)['input_ids']
                if self.tokenizer.unk_token_id in ids:
                    raise ValueError('selected transcript contains unknown character tokens')
                if self.specification['normalization'] == 'sample-variance-correction1-eps1e-7':
                    audio = ((audio - audio.mean()) / np.sqrt(audio.var(ddof=1) + np.float32(1e-7))).astype(np.float32)
                elif self.specification['normalization'] != 'stock-processor':
                    raise ValueError('unsupported waveform normalization policy')
                values = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors='pt')
                output = self.model(**{key: value.to(self.device) for key, value in values.items()})
                emissions = output.logits[0].float().log_softmax(-1).cpu().numpy()
            else:
                ids = self.model.tokenizer.text_to_ids(transcript)
                output = self.model(input_signal=torch.from_numpy(audio).to(self.device)[None],
                                    input_signal_length=torch.tensor([len(audio)], device=self.device))
                emissions = output[0][0, :int(output[1][0])].float().cpu().numpy()
        if emissions.shape[1] != self.specification['vocabulary']:
            raise ValueError('stock checkpoint vocabulary differs from recipe')
        return np.ascontiguousarray(emissions, dtype=np.float32), np.asarray(ids, dtype=np.int64), self.blank


def save_record(root: Path, identity: str, case: Case) -> dict[str, Any]:
    _validate_case(case)
    path = root / 'arrays' / f'{identity}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, emissions=case.emissions, targets=case.targets)
    temporary.replace(path)
    row = {'file': str(path.relative_to(root)), 'sha256': sha256(path),
           'emissions_shape': list(case.emissions.shape), 'targets_shape': list(case.targets.shape),
           'emissions_sha256': hashlib.sha256(case.emissions.tobytes()).hexdigest(),
           'targets_sha256': hashlib.sha256(case.targets.tobytes()).hexdigest(), 'blank_id': case.blank_id}
    atomic_json(root / '.prepare' / 'records' / f'{identity}.json', row)
    return row


def cached_record(root: Path, identity: str) -> dict[str, Any] | None:
    path = root / '.prepare' / 'records' / f'{identity}.json'
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    archive = root / 'arrays' / f'{identity}.npz'
    if row['file'] != str(archive.relative_to(root)) or sha256(archive) != row['sha256']:
        raise ValueError('cached generated record differs from its digest')
    with np.load(archive, allow_pickle=False) as saved:
        case = Case(identity, saved['emissions'], saved['targets'], row['blank_id'])
        _validate_case(case)
        for label, array in [('emissions', case.emissions), ('targets', case.targets)]:
            if list(array.shape) != row[label + '_shape'] or hashlib.sha256(array.tobytes()).hexdigest() != row[label + '_sha256']:
                raise ValueError('cached generated array differs from its digest')
    return row


def selected_cohorts(recipe: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive')
    cohorts = []
    names = set()
    for cohort in recipe['cohorts']:
        if cohort['name'] in names or not cohort['case_ids']:
            raise ValueError('recipe cohort names must be unique and nonempty')
        names.add(cohort['name'])
        ids = cohort['case_ids'][:limit]
        if any(key not in recipe['records'] for key in ids):
            raise ValueError('recipe cohort references a missing record')
        cohorts.append({'name': cohort['name'], 'case_ids': ids})
    return cohorts


def regenerate(root: Path, recipe_path: Path, device: str = 'cuda', limit: int | None = None) -> dict[str, Any]:
    import importlib.metadata
    import torch
    recipe = json.loads(recipe_path.read_text())
    if recipe.get('schema') != 1:
        raise ValueError('unsupported regeneration recipe')
    cohorts = selected_cohorts(recipe, limit)
    records = {key: recipe['records'][key] for cohort in cohorts for key in cohort['case_ids']}
    versions = {}
    for distribution in ('torch', 'transformers', 'numpy', 'nemo_toolkit', 'huggingface_hub', 'pyarrow'):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    settings = {'recipe_sha256': sha256(recipe_path), 'generator_version': GENERATOR_VERSION,
                'device': device, 'versions': versions, 'precision': 'float32',
                'sample_rate': SAMPLE_RATE, 'model_execution': 'stock unpadded forward, batch size 1',
                'attention': 'per model recipe', 'allow_tf32': False}
    root.mkdir(parents=True, exist_ok=True)
    settings_path = root / '.prepare' / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('preparation settings changed; use a different --root to avoid mixing emissions')
    if (root / 'manifest.json').exists() and not settings_path.exists():
        raise ValueError('output root already contains inputs not produced by this recipe; choose a new --root')
    atomic_json(settings_path, settings)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = AudioSources(recipe, root / '.prepare' / 'audio')
    identities = {key: digest({'settings': settings, 'case_id': key, 'record': row}) for key, row in records.items()}
    arrays = {}
    pending = {}
    for key, record in records.items():
        saved = cached_record(root, identities[key])
        if saved is None:
            pending[key] = record
        else:
            arrays[identities[key]] = saved
    if any(row.get('kind') != 'synthetic' for row in pending.values()):
        sources.prepare(pending)
    # Group models to keep only one checkpoint resident and avoid reloads.
    groups: dict[str | None, list[str]] = {}
    for key, record in pending.items():
        groups.setdefault(record.get('model'), []).append(key)
    for model_key, keys in groups.items():
        model = None if model_key is None else StockModel(recipe['models'][model_key], device)
        for index, key in enumerate(keys):
            record = records[key]
            if record.get('kind') == 'synthetic':
                emissions, targets, blank = synthetic_arrays(record)
            else:
                if model is None:
                    raise ValueError('speech record has no stock model')
                emissions, targets, blank = model(sources.waveform(record), record['transcript'])
            if len(targets) != record.get('expected_target_tokens', len(targets)):
                raise ValueError(f'{key}: tokenizer output length differs from the recipe')
            arrays[identities[key]] = save_record(root, identities[key], Case(key, emissions, targets, blank))
            if (index + 1) % 32 == 0 or index + 1 == len(keys):
                print(json.dumps({'model': model_key, 'completed': index + 1, 'total': len(keys)}), flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    partial = any(len(selected['case_ids']) != len(full['case_ids']) for selected, full in zip(cohorts, recipe['cohorts']))
    manifest = {'schema': 1, 'suite': 'regenerated-readme', 'partial': partial,
                'recipe_version': recipe['version'], 'recipe_sha256': settings['recipe_sha256'],
                'provenance': {'settings': settings, 'models': recipe['models'], 'datasets': recipe['datasets']},
                'arrays': arrays, 'cohorts': [
                    {'name': cohort['name'], 'cases': [
                        {'case_id': key, 'array_id': identities[key], 'metadata': {
                            'kind': records[key].get('kind', 'speech'), 'model': records[key].get('model')}}
                        for key in cohort['case_ids']]}
                    for cohort in cohorts]}
    atomic_json(root / 'manifest.json', manifest)
    print(json.dumps({'suite': manifest['suite'], 'partial': partial, 'records': len(records),
                      'entries': sum(len(cohort['case_ids']) for cohort in cohorts)}), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/cache/inputs'))
    parser.add_argument('--recipe', type=Path, default=Path(__file__).parent / 'manifests' / 'regeneration.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--limit', type=int, help='take this many entries per cohort; explicitly marks a partial suite')
    args = parser.parse_args()
    regenerate(args.root, args.recipe, args.device, args.limit)


if __name__ == '__main__':
    main()
