import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aetheris_abliteration.preflight import (
    available_memory_bytes,
    check_cpu_memory,
    require_approved_model,
)


class PreflightTests(unittest.TestCase):
    def test_license_and_architecture_must_be_verified(self):
        require_approved_model("qwen3", "apache-2.0")
        for model_type, license_id in [
            ("qwen2", "apache-2.0"),
            ("qwen3", None),
            ("qwen3", "unknown"),
        ]:
            with self.assertRaises(ValueError):
                require_approved_model(model_type, license_id)

    def test_cpu_memory_preflight_blocks_8b_on_8gib(self):
        with self.assertRaises(MemoryError):
            check_cpu_memory(8_000_000_000, 8 * 1024**3)
        self.assertGreater(check_cpu_memory(750_000_000, 8 * 1024**3), 0)

    def test_memory_probe_uses_reclaimable_available_not_raw_free_pages(self):
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=7 * 1024**3)):
            self.assertEqual(available_memory_bytes(), 7 * 1024**3)

    def test_unknown_parameter_count_fails_closed(self):
        with self.assertRaises(ValueError):
            check_cpu_memory(None, 8 * 1024**3)
