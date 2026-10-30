"""Behavioral checks for resumable public preparation without model downloads."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import regenerate as preparation


def fixture():
    records = {}
    for index in range(3):
        records[str(index)] = {
            'kind': 'synthetic', 'seed': 20261026 + index,
            'challenge': {'frames': 128, 'tokens': 24, 'vocabulary': 32, 'margin': 8.0},
            'expected_target_tokens': 24,
        }
    return {'schema': 1, 'version': 'test-v1', 'models': {}, 'datasets': {}, 'records': records,
            'cohorts': [{'name': 'first', 'case_ids': ['0', '1', '2']},
                        {'name': 'overlapping', 'case_ids': ['1', '0']}]}


class RegenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.recipe = self.root / 'recipe.json'
        self.recipe.write_text(json.dumps(fixture()))
        self.output = self.root / 'inputs'

    def run_preparation(self, limit=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return preparation.regenerate(self.output, self.recipe, 'cpu', limit)

    def test_full_suite_preserves_order_and_shared_records(self):
        result = self.run_preparation()
        self.assertFalse(result['partial'])
        self.assertEqual(result['suite'], 'regenerated-readme')
        self.assertEqual(len(result['arrays']), 3)
        self.assertEqual([row['case_id'] for row in result['cohorts'][0]['cases']], ['0', '1', '2'])
        self.assertEqual(result['cohorts'][0]['cases'][1]['array_id'], result['cohorts'][1]['cases'][0]['array_id'])
        self.assertEqual(result['recipe_sha256'], preparation.sha256(self.recipe))

    def test_partial_run_can_resume_to_full_without_regenerating_completed_rows(self):
        initial = self.run_preparation(limit=1)
        self.assertTrue(initial['partial'])
        with patch.object(preparation, 'synthetic_arrays', wraps=preparation.synthetic_arrays) as generate:
            final = self.run_preparation()
        self.assertEqual(generate.call_count, 1)
        self.assertFalse(final['partial'])
        for key, value in initial['arrays'].items():
            self.assertEqual(final['arrays'][key], value)

    def test_completed_run_does_no_inference(self):
        original = self.run_preparation()
        with patch.object(preparation, 'synthetic_arrays', side_effect=AssertionError('should be cached')):
            resumed = self.run_preparation()
        self.assertEqual(original, resumed)

    def test_changed_recipe_is_rejected(self):
        self.run_preparation()
        recipe = fixture()
        recipe['records']['0']['seed'] += 1
        self.recipe.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            self.run_preparation()

    def test_corrupt_cached_array_is_rejected(self):
        result = self.run_preparation()
        row = next(iter(result['arrays'].values()))
        (self.output / row['file']).write_bytes(b'invalid archive')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.run_preparation()

    def test_failure_does_not_publish_incomplete_manifest(self):
        original = preparation.synthetic_arrays

        def fail_second(record):
            if record['seed'] == 20261027:
                raise RuntimeError('model failure')
            return original(record)

        with patch.object(preparation, 'synthetic_arrays', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'model failure'):
                self.run_preparation()
        self.assertFalse((self.output / 'manifest.json').exists())
        self.assertEqual(len(list((self.output / '.prepare' / 'records').glob('*.json'))), 1)
        final = self.run_preparation()
        self.assertFalse(final['partial'])

    def test_tokenizer_length_mismatch_is_not_silently_admitted(self):
        recipe = fixture()
        recipe['records']['0']['expected_target_tokens'] = 25
        self.recipe.write_text(json.dumps(recipe))
        with self.assertRaisesRegex(ValueError, 'tokenizer output length'):
            self.run_preparation()

    def test_synthetic_ties_are_normalized_and_seeded_samples_are_stable(self):
        record = fixture()['records']['0']
        first = preparation.synthetic_arrays(record)
        second = preparation.synthetic_arrays(record)
        np.testing.assert_array_equal(first[0], second[0])
        record['challenge']['margin'] = 0
        flat, targets, blank = preparation.synthetic_arrays(record)
        self.assertEqual(blank, 0)
        np.testing.assert_allclose(flat, -np.log(32), atol=1e-6)
        self.assertEqual(targets.dtype, np.int64)

    def test_bad_limit_and_missing_members_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            preparation.selected_cohorts(fixture(), 0)
        recipe = fixture()
        recipe['cohorts'][0]['case_ids'].append('missing')
        with self.assertRaisesRegex(ValueError, 'missing record'):
            preparation.selected_cohorts(recipe, None)

    def test_audio_concatenation_uses_segment_order_and_endpoints(self):
        source = preparation.AudioSources({'datasets': {'test': {'revision': 'x'}}}, self.root / 'audio')
        segments = [{'source_id': 'one', 'start_seconds': 0.5, 'end_seconds': 1},
                    {'source_id': 'two', 'start_seconds': 0, 'end_seconds': 0.25}]
        source._save(source.audio_path('test', segments[0]), np.ones(16000, dtype=np.float32))
        source._save(source.audio_path('test', segments[1]), np.full(16000, 2, dtype=np.float32))
        waveform = source.waveform({'dataset': 'test', 'segments': segments})
        np.testing.assert_array_equal(waveform[:8000], 1)
        np.testing.assert_array_equal(waveform[8000:], 2)
        self.assertEqual(len(waveform), 12000)
        segments[1]['end_seconds'] = 2
        with self.assertRaisesRegex(ValueError, 'boundaries'):
            source.waveform({'dataset': 'test', 'segments': segments})


if __name__ == '__main__':
    unittest.main()
