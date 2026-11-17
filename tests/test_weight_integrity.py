import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from aetheris_abliteration.weight_integrity import validate_modified_weights


def fixture(root: Path, *, tied: bool = True, wrong_other: bool = False):
    embedding = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.bfloat16)
    head = embedding.clone() if tied else embedding + 1
    original = {
        "model.embed_tokens.weight": embedding,
        "lm_head.weight": head,
        "model.layers.0.self_attn.o_proj.weight": torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.tensor([1., 1.], dtype=torch.bfloat16),
    }
    modified = {
        k: v.float().clone()
        for k, v in original.items() if k != "lm_head.weight"
    }
    modified["model.layers.0.self_attn.o_proj.weight"][0, 0] = 0.5
    if wrong_other:
        modified["model.layers.0.input_layernorm.weight"][0] = 0.5
    before, after = root / "original.safetensors", root / "edited.safetensors"
    save_file(original, before)
    save_file(modified, after)
    return before, after


class WeightIntegrityTests(unittest.TestCase):
    def test_identical_tied_head_may_be_omitted_but_edit_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after = fixture(Path(tmp))
            report = validate_modified_weights(before, after, tied_embeddings=True)
            self.assertTrue(report["tied_alias_verified"])
            self.assertEqual(report["edited_tensors"], 1)
            self.assertEqual(report["unexpected_changes"], 0)

    def test_untied_head_must_never_be_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after = fixture(Path(tmp), tied=False)
            with self.assertRaises(ValueError):
                validate_modified_weights(before, after, tied_embeddings=True)

    def test_config_disallows_tied_head_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after = fixture(Path(tmp))
            with self.assertRaises(ValueError):
                validate_modified_weights(before, after, tied_embeddings=False)

    def test_non_target_weight_mutation_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after = fixture(Path(tmp), wrong_other=True)
            with self.assertRaises(ValueError):
                validate_modified_weights(before, after, tied_embeddings=True)

    def test_no_actual_weight_edit_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after = fixture(Path(tmp))
            from safetensors import safe_open
            state = {}
            with safe_open(before, framework="pt", device="cpu") as f:
                keys = tuple(f.keys())
                for key in keys:
                    if key != "lm_head.weight":
                        state[key] = f.get_tensor(key).float()
            save_file(state, after)
            with self.assertRaises(ValueError):
                validate_modified_weights(before, after, tied_embeddings=True)
