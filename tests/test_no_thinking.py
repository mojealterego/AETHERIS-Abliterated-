import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aetheris_abliteration.benchmark_runner import run_one_turn


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["tokenize"] is False
        return "<|im_start|>user\\n" + messages[0]["content"] + "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"


class NoThinkingTests(unittest.TestCase):
    def test_qwen_no_think_uses_raw_chat_template(self):
        with tempfile.TemporaryDirectory() as root:
            binary = Path(root) / "llama-cli"
            binary.write_text("#!/bin/sh")
            binary.chmod(0o755)
            model = Path(root) / "sample.gguf"
            model.write_bytes(b"GGUF" + b"\\x00" * 100)
            with patch("aetheris_abliteration.benchmark_runner.subprocess.run") as proc:
                proc.return_value.stdout = "43"
                proc.return_value.stderr = ""
                result = run_one_turn(
                    binary, model, "Ile wynosi 17 plus 26?",
                    timeout=30, tokenizer=TinyTokenizer(),
                )
                argv = proc.call_args.args[0]
                self.assertIn("--no-conversation", argv)
                self.assertIn("<think>", argv[argv.index("-p") + 1])
                self.assertEqual(result["model_output_mode"], "QWEN_NO_THINK")
