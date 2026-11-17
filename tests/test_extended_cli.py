import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aetheris_abliteration.benchmark_runner import benchmark_to_file
from aetheris_abliteration.cli import main


class ExtendedCliTests(unittest.TestCase):
    def test_audit_cli_nonzero_if_insufficient(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "evidence.jsonl"
            path.write_text(json.dumps({"id": "target1", "cohort": "target", "before": "complied", "after": "complied"}) + "\n")
            with patch("builtins.print") as output:
                result = main(["audit-evidence", "--input", str(path)])
            self.assertEqual(result, 2)
            self.assertTrue(output.called)

    def test_benchmark_atomic_and_refuses_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "llama-cli"
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)
            model = root / "model.gguf"
            model.write_bytes(b"GGUF" + bytes(100))
            suite = root / "prompts.jsonl"
            suite.write_text(json.dumps({"id": "smoke", "prompt": "Opisz fotograficzną scenę."}) + "\n")
            output = root / "results.jsonl"
            fake = {"model": "model.gguf", "input": "Opisz fotograficzną scenę.", "raw_output": "test", "stderr": "", "judgement": "UNREVIEWED"}
            with patch("aetheris_abliteration.benchmark_runner.run_one_turn", return_value=fake):
                self.assertEqual(benchmark_to_file(binary, model, suite, output, timeout=10), 1)
            result = json.loads(output.read_text().splitlines()[0])
            self.assertEqual(result["id"], "smoke")
            self.assertEqual(result["judgement"], "UNREVIEWED")
            with self.assertRaises(FileExistsError):
                benchmark_to_file(binary, model, suite, output, timeout=10)
