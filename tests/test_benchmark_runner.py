import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aetheris_abliteration.benchmark_runner import load_prompts, run_one_turn


class BenchmarkRunnerTests(unittest.TestCase):
    def test_binary_and_model_paths_required(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            exe = root / "llama-cli"
            exe.write_text("#! /bin/sh")
            model = root / "model.gguf"
            model.write_bytes(b"GGUF" + bytes(30))
            with self.assertRaises(ValueError):
                run_one_turn(exe, model, "hi", timeout=12)

    def test_subprocess_success_and_no_shell(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            exe = root / "llama-cli"
            exe.write_text("#!/bin/sh\n")
            exe.chmod(0o755)
            model = root / "model.gguf"
            model.write_bytes(b"GGUF" + bytes(30))
            with patch("aetheris_abliteration.benchmark_runner.subprocess.run") as runner:
                runner.return_value.stdout = "Szanowny tekst"
                runner.return_value.stderr = ""
                runner.return_value.returncode = 0
                result = run_one_turn(exe, model, "testowy prompt", timeout=12)
                self.assertIn("Szanowny", result["raw_output"])
                argv = runner.call_args.args[0]
                self.assertEqual(argv[0], str(exe.resolve()))
                self.assertIn("--single-turn", argv)
                self.assertNotIn("--interactive", argv)
                self.assertFalse(runner.call_args.kwargs.get("shell", False))

    def test_rejects_duplicate_prompt_ids(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "prompts.jsonl"
            rows = [{"id": "one", "prompt": "opisz krajobraz"}, {"id": "one", "prompt": "opisz niebo"}]
            p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prompts(p)
