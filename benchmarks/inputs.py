"""Portable, checksummed CTC benchmark inputs; no model execution or pickle loading."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any
from urllib.request import urlopen
import zipfile

import numpy as np


@dataclass(frozen=True)
class Case:
    """One unpadded recording with contiguous FP32 emissions and int64 targets."""
    case_id: str
    emissions: np.ndarray
    targets: np.ndarray
    blank_id: int
    metadata: dict[str, Any] = field(default_factory=dict)


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _safe_path(root: Path, name: str) -> Path:
    parts = PurePosixPath(name).parts
    if not parts or name.startswith('/') or '\\' in name or any(p in ('.', '..') for p in parts):
        raise ValueError(f'unsafe input path: {name!r}')
    path = root.joinpath(*parts)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f'symlinks are not accepted in input paths: {path}')
        if parent == root:
            break
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'input path escapes root: {name!r}')
    return path


def _validate_case(case: Case) -> None:
    e, t = case.emissions, case.targets
    if e.dtype != np.float32 or e.ndim != 2 or min(e.shape) < 1 or not e.flags.c_contiguous:
        raise ValueError(f'{case.case_id}: expected contiguous float32[T,V] emissions')
    if t.dtype != np.int64 or t.ndim != 1 or not t.flags.c_contiguous:
        raise ValueError(f'{case.case_id}: expected contiguous int64[U] targets')
    if type(case.blank_id) is not int or not 0 <= case.blank_id < e.shape[1]:
        raise ValueError(f'{case.case_id}: invalid blank ID')
    if np.isnan(e).any() or np.isposinf(e).any():
        raise ValueError(f'{case.case_id}: NaN/+infinity emissions')
    if ((t < 0) | (t >= e.shape[1]) | (t == case.blank_id)).any():
        raise ValueError(f'{case.case_id}: invalid target IDs')
    if len(t) + np.count_nonzero(t[1:] == t[:-1]) > len(e):
        raise ValueError(f'{case.case_id}: insufficient CTC frames')


def load_manifest(root: Path) -> dict[str, Any]:
    """Read a versioned bundle manifest and validate its paths and membership."""
    path = _safe_path(Path(root), 'manifest.json')
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 1 or manifest.get('suite') not in ('readme', 'regenerated-readme', 'synthetic'):
        raise ValueError('unsupported benchmark input manifest')
    if manifest['suite'] == 'readme':
        release = json.loads((Path(__file__).parent / 'manifests' / 'readme.json').read_text())
        if sha256(path) != release['manifest_sha256']:
            raise ValueError('README input manifest differs from the pinned release')
    if manifest['suite'] == 'regenerated-readme':
        _validate_regenerated_manifest(manifest)
    names: set[str] = set()
    for cohort in manifest['cohorts']:
        if cohort['name'] in names or not cohort['cases']:
            raise ValueError('cohort names must be unique and cohorts nonempty')
        names.add(cohort['name'])
        for row in cohort['cases']:
            key = row['array_id']
            if not re.fullmatch(r'[0-9a-f]{64}', key) or key not in manifest['arrays']:
                raise ValueError('invalid or missing array ID')
    for row in manifest['arrays'].values():
        _safe_path(Path(root), row['file'])
    return manifest


def _validate_regenerated_manifest(manifest: dict[str, Any]) -> None:
    """Bind regenerated inputs to a public recipe without pinning GPU numerics."""
    recipe_path = Path(__file__).parent / 'manifests' / 'regeneration.json'
    recipe = json.loads(recipe_path.read_text())
    if (manifest.get('recipe_sha256') != sha256(recipe_path)
            or manifest.get('recipe_version') != recipe['version']):
        raise ValueError('regenerated input recipe differs from the checked-in recipe')
    if type(manifest.get('partial')) is not bool:
        raise ValueError('regenerated manifest must declare whether it is partial')
    expected = {cohort['name']: cohort['case_ids'] for cohort in recipe['cohorts']}
    actual = {cohort['name']: [case['case_id'] for case in cohort['cases']]
              for cohort in manifest['cohorts']}
    if not manifest['partial']:
        if actual != expected or list(actual) != list(expected):
            raise ValueError('full regenerated suite must contain every recipe case in order')
    else:
        if not actual or set(actual) - set(expected):
            raise ValueError('partial regenerated suite contains unknown cohorts')
        for name, case_ids in actual.items():
            if not case_ids or case_ids != expected[name][:len(case_ids)]:
                raise ValueError('partial regenerated cohort must be a nonempty recipe prefix')


def load_cohorts(root: Path, suite: str = 'readme') -> list[tuple[str, list[Case]]]:
    """Load ordered cohorts, checking file and raw-array hashes before returning.

    The synthetic suite is generated deterministically in memory and needs no
    bundle. It is a smoke/performance workload, not the README benchmark suite.
    Repeated suite entries remain repeated; identical arrays share host storage.
    """
    if suite == 'synthetic':
        return synthetic_cohorts()
    if suite not in ('readme', 'regenerated-readme', 'regenerated-readme-partial'):
        raise ValueError(f'unknown suite: {suite}')
    root = Path(root)
    manifest = load_manifest(root)
    expected_suite = 'regenerated-readme' if suite == 'regenerated-readme-partial' else suite
    if manifest['suite'] != expected_suite:
        raise ValueError('requested suite does not match input manifest')
    if manifest.get('partial') and suite != 'regenerated-readme-partial':
        raise ValueError('partial inputs require --suite regenerated-readme-partial')
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, row in manifest['arrays'].items():
        path = _safe_path(root, row['file'])
        if sha256(path) != row['sha256']:
            raise ValueError(f'input file digest mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'emissions', 'targets'}:
                raise ValueError(f'unexpected arrays in {path}')
            e, t = data['emissions'], data['targets']
        if list(e.shape) != row['emissions_shape'] or list(t.shape) != row['targets_shape']:
            raise ValueError(f'input shape mismatch: {path}')
        for label, value in (('emissions', e), ('targets', t)):
            if hashlib.sha256(value.tobytes()).hexdigest() != row[f'{label}_sha256']:
                raise ValueError(f'{label} raw-array digest mismatch: {path}')
        _validate_case(Case(key, e, t, row['blank_id']))
        arrays[key] = e, t
    result = []
    for cohort in manifest['cohorts']:
        cases = []
        for row in cohort['cases']:
            entry = manifest['arrays'][row['array_id']]
            e, t = arrays[row['array_id']]
            cases.append(Case(row['case_id'], e, t, entry['blank_id'], row.get('metadata', {})))
        result.append((cohort['name'], cases))
    return result


def synthetic_cohorts() -> list[tuple[str, list[Case]]]:
    """Return a small fixed arithmetic fixture without RNG/version dependencies."""
    cohorts = []
    for name, vocabulary, blank, frames in (
        ('synthetic-short', 32, 0, 160), ('synthetic-wide', 1025, 1024, 192),
        ('synthetic-long', 32, 0, 2048), ('synthetic-ties', 32, 0, 128),
    ):
        cases = []
        for index in range(8):
            length = frames + index * 3
            targets = (np.arange(24, dtype=np.int64) % 15) + 1
            emissions = np.full((length, vocabulary), -8.0, dtype=np.float32)
            path = np.full(length, blank, dtype=np.int64)
            positions = (np.arange(24) * (length - 2) // 24) + 1
            path[positions] = targets
            emissions[np.arange(length), path] = 0.0
            if name == 'synthetic-ties':
                emissions.fill(0.0)
            case = Case(f'{name}-{index:03d}', emissions, targets, blank,
                        {'kind': 'synthetic', 'generator': 'arithmetic-ctc-v1',
                         'normalized_log_probabilities': False})
            _validate_case(case)
            cases.append(case)
        cohorts.append((name, cases))
    return cohorts


def prepare(root: Path, bundle: Path | None = None, url: str | None = None,
            expected_sha256: str | None = None) -> None:
    """Verify and install a ZIP bundle atomically into a new directory.

    The committed release manifest pins the default bundle digest. A supplied
    digest permits another transport archive of the same pinned input manifest.
    URLs must be HTTPS. Archives may contain regular files only; absolute paths,
    traversal, duplicate members, and links are rejected before extraction.
    """
    root = Path(root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f'input destination already exists: {root}; use verify')
    if (bundle is None) == (url is None):
        raise ValueError('provide exactly one --bundle or --url')
    release = json.loads((Path(__file__).parent / 'manifests' / 'readme.json').read_text())
    expected = expected_sha256 or release['bundle_sha256']
    if not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('expected SHA-256 must contain 64 lowercase hexadecimal digits')
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.inputs-', dir=root.parent) as temporary:
        temporary = Path(temporary)
        if url is not None:
            if not url.startswith('https://'):
                raise ValueError('input download URL must use HTTPS')
            bundle = temporary / 'download.zip'
            with urlopen(url, timeout=60) as response, bundle.open('wb') as output:
                if not response.geturl().startswith('https://'):
                    raise ValueError('HTTPS download redirected to an insecure URL')
                shutil.copyfileobj(response, output)
        assert bundle is not None
        if sha256(Path(bundle)) != expected:
            raise ValueError('bundle SHA-256 mismatch')
        extracted = temporary / 'extracted'
        extracted.mkdir()
        with zipfile.ZipFile(bundle) as archive:
            seen: set[str] = set()
            total = 0
            for member in archive.infolist():
                path = _safe_path(extracted, member.filename)
                mode = member.external_attr >> 16
                if member.filename in seen or member.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ValueError(f'archive member is duplicate or not a regular file: {member.filename}')
                seen.add(member.filename)
                total += member.file_size
                if total > 16 * 1024 ** 3:
                    raise ValueError('unpacked bundle exceeds 16 GiB safety limit')
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, path.open('xb') as output:
                    shutil.copyfileobj(source, output)
        manifest = load_manifest(extracted)
        expected_files = {'manifest.json'} | {row['file'] for row in manifest['arrays'].values()}
        if seen != expected_files:
            raise ValueError('archive members differ from input manifest')
        load_cohorts(extracted)
        extracted.rename(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='install a locally supplied or HTTPS bundle')
    prep.add_argument('--root', type=Path, required=True)
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle', type=Path)
    source.add_argument('--url')
    prep.add_argument('--sha256')
    verify = sub.add_parser('verify', help='verify every input hash, dtype, and shape')
    verify.add_argument('--root', type=Path, required=True)
    verify.add_argument('--suite', choices=('readme', 'regenerated-readme', 'regenerated-readme-partial', 'synthetic'),
                        default='regenerated-readme')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root, args.bundle, args.url, args.sha256)
    cohorts = load_cohorts(args.root, getattr(args, 'suite', 'readme'))
    print(json.dumps({'cohorts': {name: len(cases) for name, cases in cohorts},
                      'entries': sum(len(cases) for _, cases in cohorts)}, indent=2))


if __name__ == '__main__':
    main()
