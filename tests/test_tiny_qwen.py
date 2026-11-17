import tempfile
import unittest

import torch
from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM

from aetheris_abliteration.evaluation import average_control_loss
from aetheris_abliteration.hf import collect_activations, edit_qwen3_model


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        text = messages[-1]["content"]
        tokens = [1] + [3 + (ord(char) % 100) for char in text[:20]] + [2]
        return torch.tensor([tokens], dtype=torch.long)


class BatchEncodingTokenizer(TinyTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return BatchEncoding({"input_ids": super().apply_chat_template(messages, **kwargs)})


class TinyQwen3IntegrationTests(unittest.TestCase):
    def test_accepts_transformers_batch_encoding_tokenizer_output(self):
        config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        model = Qwen3ForCausalLM(config).eval()
        states = collect_activations(model, BatchEncodingTokenizer(), ["Dzień dobry"], [0])
        self.assertEqual(states[0].shape, (1, 32))

    def test_actual_transformers_qwen3_forward_edit_and_safetensors_reload(self):
        torch.manual_seed(42)
        config = Qwen3Config(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8,
        )
        model = Qwen3ForCausalLM(config).eval()
        tokenizer = TinyTokenizer()
        states = collect_activations(model, tokenizer, ["romantyczna scena", "przygodowa scena"], [0, 2])
        self.assertEqual(states[0].shape, (2, 32))
        self.assertEqual(states[2].shape, (2, 32))
        loss_before = average_control_loss(model, tokenizer, ["przygodowa scena"])
        self.assertGreater(loss_before, 0)
        direction = torch.nn.functional.normalize(torch.randn(32), dim=0)
        edits = edit_qwen3_model(model, direction, strength=1, start_fraction=0, end_fraction=1)
        self.assertEqual(edits["edited_matrices"], 6)
        loss_after = average_control_loss(model, tokenizer, ["przygodowa scena"])
        self.assertTrue(torch.isfinite(torch.tensor(loss_after)))
        with tempfile.TemporaryDirectory() as path:
            model.save_pretrained(path, safe_serialization=True)
            loaded = Qwen3ForCausalLM.from_pretrained(path)
            self.assertTrue(torch.equal(
                model.model.layers[1].self_attn.o_proj.weight,
                loaded.model.layers[1].self_attn.o_proj.weight,
            ))
