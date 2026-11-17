import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from aetheris_abliteration.cli import main


class CLITests(unittest.TestCase):
    def test_inspect_prints_calibration_statistics(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "pairs.jsonl"
            records = (
                [{"kind": "target", "prompt": f"Fikcja dorosłych wariant {i}"} for i in range(8)] +
                [{"kind": "control", "prompt": f"Fikcja przygodowa wariant {i}"} for i in range(8)]
            )
            path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["inspect", "--calibration", str(path), "--min-per-class", "8"])
            self.assertEqual(code, 0)
            info = json.loads(output.getvalue())
            self.assertEqual(info["target_count"], 8)
            self.assertEqual(info["control_count"], 8)

    def test_build_requires_explicit_output_path(self):
        with self.assertRaises(SystemExit):
            main(["build", "--calibration", "nonexistent"])
