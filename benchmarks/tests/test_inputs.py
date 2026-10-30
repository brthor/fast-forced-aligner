"""Behavioral checks for portable benchmark input admission and archive safety."""
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inputs
from inputs import Case, _safe_path, _validate_case, load_cohorts, load_manifest, prepare


class InputTests(unittest.TestCase):
    def make_bundle_fixture(self, root):
        """Create a tiny release with overlapping cohorts and repeated entries."""
        source = root / 'source'
        source.mkdir()
        e = np.zeros((5, 3), dtype=np.float32)
        t = np.array([1, 2], dtype=np.int64)
        key = 'a' * 64
        np.savez(source / 'array.npz', emissions=e, targets=t)
        manifest = {
            'schema': 1, 'suite': 'readme',
            'arrays': {key: {
                'file': 'array.npz',
                'sha256': inputs.sha256(source / 'array.npz'),
                'emissions_sha256': hashlib.sha256(e.tobytes()).hexdigest(),
                'targets_sha256': hashlib.sha256(t.tobytes()).hexdigest(),
                'emissions_shape': list(e.shape), 'targets_shape': list(t.shape),
                'blank_id': 0,
            }},
            'cohorts': [
                {'name': 'first', 'cases': [
                    {'case_id': 'b', 'array_id': key},
                    {'case_id': 'a', 'array_id': key},
                ]},
                {'name': 'second', 'cases': [{'case_id': 'b', 'array_id': key}]},
            ],
        }
        release_dir = root / 'manifests'
        release_dir.mkdir()
        self.pin_fixture_manifest(root, source, manifest)
        return source, manifest

    def pin_fixture_manifest(self, root, source, manifest):
        (source / 'manifest.json').write_text(json.dumps(manifest))
        (root / 'manifests' / 'readme.json').write_text(json.dumps({
            'manifest_sha256': inputs.sha256(source / 'manifest.json'),
            'bundle_sha256': '0' * 64,
        }))

    def test_verified_bundle_preserves_order_and_shared_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.make_bundle_fixture(root)
            bundle = root / 'bundle.zip'
            with zipfile.ZipFile(bundle, 'w') as archive:
                for path in source.iterdir():
                    archive.write(path, path.name)
            with patch.object(inputs, '__file__', str(root / 'inputs.py')):
                prepare(root / 'installed', bundle=bundle,
                        expected_sha256=inputs.sha256(bundle))
                cohorts = load_cohorts(root / 'installed')
                self.assertEqual([name for name, _ in cohorts], ['first', 'second'])
                self.assertEqual([c.case_id for c in cohorts[0][1]], ['b', 'a'])
                self.assertIs(cohorts[0][1][0].emissions, cohorts[1][1][0].emissions)
                with self.assertRaises(FileExistsError):
                    prepare(root / 'installed', bundle=bundle,
                            expected_sha256=inputs.sha256(bundle))
            self.assertFalse(list(root.glob('.inputs-*')))

    def test_manifest_pin_membership_and_file_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, manifest = self.make_bundle_fixture(root)
            with patch.object(inputs, '__file__', str(root / 'inputs.py')):
                with (source / 'manifest.json').open('a') as handle:
                    handle.write('\n')
                with self.assertRaisesRegex(ValueError, 'differs from the pinned release'):
                    load_manifest(source)
                manifest['cohorts'][0]['cases'][0]['array_id'] = 'b' * 64
                self.pin_fixture_manifest(root, source, manifest)
                with self.assertRaisesRegex(ValueError, 'missing array ID'):
                    load_manifest(source)
                manifest['cohorts'][0]['cases'][0]['array_id'] = 'a' * 64
                self.pin_fixture_manifest(root, source, manifest)
                with (source / 'array.npz').open('ab') as handle:
                    handle.write(b'tampered')
                with self.assertRaisesRegex(ValueError, 'file digest mismatch'):
                    load_cohorts(source)

    def test_raw_digest_and_shape_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, manifest = self.make_bundle_fixture(root)
            array = manifest['arrays']['a' * 64]
            with patch.object(inputs, '__file__', str(root / 'inputs.py')):
                original = array['targets_sha256']
                array['targets_sha256'] = '0' * 64
                self.pin_fixture_manifest(root, source, manifest)
                with self.assertRaisesRegex(ValueError, 'raw-array digest mismatch'):
                    load_cohorts(source)
                array['targets_sha256'] = original
                array['emissions_shape'] = [4, 3]
                self.pin_fixture_manifest(root, source, manifest)
                with self.assertRaisesRegex(ValueError, 'shape mismatch'):
                    load_cohorts(source)

    def make_regenerated_fixture(self, root):
        source, manifest = self.make_bundle_fixture(root)
        recipe = {'schema': 1, 'version': 'fixture-v1', 'cohorts': [
            {'name': row['name'], 'case_ids': [case['case_id'] for case in row['cases']]}
            for row in manifest['cohorts']]}
        recipe_path = root / 'manifests' / 'regeneration.json'
        recipe_path.write_text(json.dumps(recipe))
        manifest.update(suite='regenerated-readme', partial=False,
                        recipe_version='fixture-v1', recipe_sha256=inputs.sha256(recipe_path))
        (source / 'manifest.json').write_text(json.dumps(manifest))
        return source, manifest

    def test_regenerated_recipe_and_coverage_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, manifest = self.make_regenerated_fixture(root)
            with patch.object(inputs, '__file__', str(root / 'inputs.py')):
                self.assertEqual(len(load_cohorts(source, 'regenerated-readme')), 2)
                manifest['recipe_sha256'] = '0' * 64
                (source / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'recipe differs'):
                    load_manifest(source)
                manifest['recipe_sha256'] = inputs.sha256(root / 'manifests/regeneration.json')
                manifest['cohorts'][0]['cases'].reverse()
                (source / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'every recipe case in order'):
                    load_manifest(source)

    def test_partial_regeneration_requires_explicit_suite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, manifest = self.make_regenerated_fixture(root)
            manifest['partial'] = True
            manifest['cohorts'][0]['cases'] = manifest['cohorts'][0]['cases'][:1]
            (source / 'manifest.json').write_text(json.dumps(manifest))
            with patch.object(inputs, '__file__', str(root / 'inputs.py')):
                with self.assertRaisesRegex(ValueError, 'partial inputs require'):
                    load_cohorts(source, 'regenerated-readme')
                self.assertEqual(len(load_cohorts(source, 'regenerated-readme-partial')[0][1]), 1)
                manifest['cohorts'][0]['cases'][0]['case_id'] = 'unknown'
                (source / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'nonempty recipe prefix'):
                    load_manifest(source)

    def test_regenerated_arrays_remain_checksummed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, manifest = self.make_regenerated_fixture(root)
            with (source / 'array.npz').open('ab') as output:
                output.write(b'changed')
            with patch.object(inputs, '__file__', str(root / 'inputs.py')):
                with self.assertRaisesRegex(ValueError, 'file digest mismatch'):
                    load_cohorts(source, 'regenerated-readme')

    def test_synthetic_is_deterministic_and_has_nonzero_blank(self):
        a = load_cohorts(Path('/unused'), 'synthetic')
        b = load_cohorts(Path('/unused'), 'synthetic')
        self.assertEqual(sum(len(cases) for _, cases in a), 32)
        self.assertTrue(any(case.blank_id != 0 for _, cases in a for case in cases))
        for (name, cases), (other_name, others) in zip(a, b):
            self.assertEqual(name, other_name)
            for case, other in zip(cases, others):
                np.testing.assert_array_equal(case.emissions, other.emissions)
                np.testing.assert_array_equal(case.targets, other.targets)
                _validate_case(case)

    def test_invalid_target_and_nonfinite_emission_rejected(self):
        e = np.zeros((5, 3), dtype=np.float32)
        for tokens in ([0], [3], [1, 1, 1, 1]):
            with self.assertRaises(ValueError):
                _validate_case(Case('bad', e, np.array(tokens, dtype=np.int64), 0))
        for value in (np.nan, np.inf):
            broken = e.copy()
            broken[0, 0] = value
            with self.assertRaises(ValueError):
                _validate_case(Case('bad', broken, np.array([1], dtype=np.int64), 0))
        with self.assertRaises(ValueError):
            _validate_case(Case('bad', e.astype(np.float64), np.array([1], dtype=np.int64), 0))

    def test_path_traversal_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('../escape', '/absolute', 'arrays/../../escape', 'arrays\\escape'):
                with self.assertRaises(ValueError):
                    _safe_path(root, name)
            (root / 'link').symlink_to('/tmp', target_is_directory=True)
            with self.assertRaises(ValueError):
                _safe_path(root, 'link/escape')

    def test_archive_rejects_digest_traversal_and_link_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (name, mode) in enumerate((('../escape', 0o100644), ('link', stat.S_IFLNK | 0o777))):
                bundle = root / f'{index}.zip'
                with zipfile.ZipFile(bundle, 'w') as archive:
                    info = zipfile.ZipInfo(name)
                    info.external_attr = mode << 16
                    archive.writestr(info, b'unsafe')
                digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    prepare(root / 'output', bundle=bundle, expected_sha256='0' * 64)
                with self.assertRaises(ValueError):
                    prepare(root / 'output', bundle=bundle, expected_sha256=digest)
                self.assertFalse((root / 'output').exists())
                self.assertFalse((root / 'escape').exists())


if __name__ == '__main__':
    unittest.main()
