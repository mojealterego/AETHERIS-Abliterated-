import unittest

import torch

from aetheris_abliteration.core import (
    estimate_separation,
    project_weight_,
    refusal_direction,
)


class DirectionTests(unittest.TestCase):
    def test_recovers_mean_difference(self):
        target = torch.tensor([[3., 1., 0.], [5., 1., 0.]])
        control = torch.tensor([[0., 1., 0.], [2., 1., 0.]])
        direction = refusal_direction(target, control)
        self.assertTrue(torch.allclose(direction, torch.tensor([1., 0., 0.])))

    def test_rejects_non_finite_and_identical_means(self):
        with self.assertRaises(ValueError):
            refusal_direction(torch.ones(2, 3), torch.ones(2, 3))
        with self.assertRaises(ValueError):
            refusal_direction(torch.tensor([[float("nan"), 0.], [1., 0.]]), torch.zeros(2, 2))

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            refusal_direction(torch.zeros(2, 3), torch.zeros(2, 4))

    def test_holdout_separation_is_positive(self):
        target = torch.tensor([[2., 0.], [3., 1.], [4., -1.]])
        control = torch.tensor([[-2., 0.], [-3., 1.], [-4., -1.]])
        self.assertGreater(estimate_separation(target, control, torch.tensor([1., 0.])), 3)


class OrthogonalizationTests(unittest.TestCase):
    def test_output_projection_is_orthogonal(self):
        w = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float32)
        direction = torch.tensor([1., 1.])
        project_weight_(w, direction, axis="output", strength=1.0, chunk_size=2)
        self.assertLess(torch.max(torch.abs(direction @ w)).item(), 1e-5)

    def test_input_projection_is_orthogonal(self):
        w = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float32)
        direction = torch.tensor([1., 1., 1.])
        project_weight_(w, direction, axis="input", strength=1.0, chunk_size=1)
        self.assertLess(torch.max(torch.abs(w @ direction)).item(), 1e-5)

    def test_strength_zero_preserves_weights(self):
        w = torch.randn(4, 5)
        before = w.clone()
        project_weight_(w, torch.ones(4), axis="output", strength=0.0)
        self.assertTrue(torch.equal(w, before))

    def test_strength_half_reduces_component_by_half(self):
        w = torch.tensor([[2., 1.], [0., 1.]])
        d = torch.tensor([1., 0.])
        project_weight_(w, d, axis="output", strength=0.5)
        self.assertTrue(torch.allclose(w, torch.tensor([[1., 0.5], [0., 1.]])))

    def test_rejects_invalid_strength_and_axis(self):
        with self.assertRaises(ValueError):
            project_weight_(torch.zeros(2, 2), torch.ones(2), axis="output", strength=1.2)
        with self.assertRaises(ValueError):
            project_weight_(torch.zeros(2, 2), torch.ones(2), axis="unknown")
        with self.assertRaises(ValueError):
            project_weight_(torch.zeros(2, 2), torch.ones(3), axis="output")

    def test_bfloat16_weights_remain_bfloat16(self):
        w = torch.randn(4, 8).to(torch.bfloat16)
        project_weight_(w, torch.ones(4), axis="output", strength=0.8)
        self.assertEqual(w.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(w).all())


if __name__ == "__main__":
    unittest.main()
