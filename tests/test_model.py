"""Numerical software checks using generated tensors only."""

import unittest
import torch
from mwaf.loss import FocalLoss
from mwaf.model import MWAFNet


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_output_tasks_and_single_segment_inference(self):
        for n in [2, 3, 5]:
            model = MWAFNet(n).eval()
            with torch.no_grad():
                out = model(torch.randn(1, 1, 1500))
            self.assertEqual(tuple(out.shape), (1, n))
            self.assertTrue(torch.isfinite(out).all())

    def test_gradient_and_state_reload(self):
        torch.manual_seed(7)
        model = MWAFNet(5)
        x = torch.randn(2, 1, 1500)
        target = torch.tensor([0, 4])
        loss = FocalLoss(5)(model(x), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name in [
            "branches.0.input_conv.0.conv1.weight",
            "fusion.branch_weights",
            "classifier.5.weight",
        ]:
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)
        model.eval()
        clone = MWAFNet(5).eval()
        clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
