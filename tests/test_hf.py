import unittest
from types import SimpleNamespace

import torch
from torch import nn

from aetheris_abliteration.hf import edit_qwen3_model, select_direction


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = SimpleNamespace(o_proj=nn.Linear(4, 4, bias=False))
        self.mlp = SimpleNamespace(down_proj=nn.Linear(8, 4, bias=False))


class FakeModel:
    def __init__(self):
        self.model = SimpleNamespace(layers=[FakeBlock(), FakeBlock(), FakeBlock()])


class HFTests(unittest.TestCase):
    def test_edits_qwen3_residual_outputs(self):
        model = FakeModel()
        direction = torch.tensor([1., 0., 0., 0.])
        result = edit_qwen3_model(model, direction, strength=1.0, start_fraction=0, end_fraction=1)
        self.assertEqual(result["edited_matrices"], 6)
        for block in model.model.layers:
            self.assertLess(block.self_attn.o_proj.weight[0].abs().max().item(), 1e-5)
            self.assertLess(block.mlp.down_proj.weight[0].abs().max().item(), 1e-5)

    def test_selects_direction_with_best_holdout(self):
        train = {
            1: (torch.tensor([[1., 0.], [2., 0.]]), torch.tensor([[0., 0.], [-1., 0.]])),
            2: (torch.tensor([[0., 1.], [0., 2.]]), torch.tensor([[0., 0.], [0., -1.]])),
        }
        val = {
            1: (torch.tensor([[2., 0.], [3., 0.]]), torch.tensor([[-2., 0.], [-3., 0.]])),
            2: (torch.tensor([[0., 1.], [0., -1.]]), torch.tensor([[0., 1.], [0., -1.]])),
        }
        layer, direction, scores = select_direction(train, val)
        self.assertEqual(layer, 1)
        self.assertEqual(direction.shape, (2,))
        self.assertGreater(scores[1], scores[2])


if __name__ == "__main__":
    unittest.main()
