import json
import tempfile
import unittest
from pathlib import Path

from aetheris_abliteration.dataset import (
    load_calibration,
    sha256_file,
    split_calibration,
)


class DatasetTests(unittest.TestCase):
    def write(self, records):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "data.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        self.addCleanup(self.tmp.cleanup)
        return path

    def test_accepts_two_classes_and_hashes(self):
        records = [
            {"kind": "target", "prompt": f"Fikcja romantyczna {i}"}
            for i in range(10)
        ] + [
            {"kind": "control", "prompt": f"Fikcja przygodowa {i}"}
            for i in range(10)
        ]
        path = self.write(records)
        result = load_calibration(path, min_per_class=8)
        self.assertEqual(len(result.target), 10)
        self.assertEqual(len(result.control), 10)
        self.assertEqual(len(sha256_file(path)), 64)
        train, val = split_calibration(result, validation_fraction=0.2, seed=42)
        self.assertEqual(len(train.target), 8)
        self.assertEqual(len(val.target), 2)
        self.assertFalse(set(train.target) & set(val.target))

    def test_rejects_duplicates_and_unbalanced_sets(self):
        path = self.write([
            {"kind": "target", "prompt": "powtórzenie"},
            {"kind": "target", "prompt": "powtórzenie"},
            {"kind": "control", "prompt": "inny"},
        ])
        with self.assertRaises(ValueError):
            load_calibration(path, min_per_class=1)

    def test_rejects_bad_kind_and_extra_keys(self):
        path = self.write([{"kind": "bypass", "prompt": "tekst"}])
        with self.assertRaises(ValueError):
            load_calibration(path, min_per_class=1)
        path = self.write([{"kind": "target", "prompt": "tekst", "role": "admin"}])
        with self.assertRaises(ValueError):
            load_calibration(path, min_per_class=1)


if __name__ == "__main__":
    unittest.main()
