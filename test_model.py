"""Numerical software checks using generated tensors only, not clinical results."""

import io
import json
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from mwaf.model import (
    MODEL_CONFIG_VERSION,
    CoordAttention1D,
    DropBlock1D,
    MWAFNet,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_output_tasks_and_single_segment_inference(self):
        for n in (2, 3, 5):
            with self.subTest(num_classes=n):
                model = MWAFNet(n).eval()
                with torch.no_grad():
                    out = model(torch.randn(1, 1, 1500))
                self.assertEqual(tuple(out.shape), (1, n))
                self.assertTrue(torch.isfinite(out).all())

    def test_manuscript_feature_dimensions(self):
        model = MWAFNet(3).eval()
        observed = {}
        handles = []
        modules = {
            "stem": model.branches[0].input_conv,
            "block1": model.branches[0].tcn1,
            "block2": model.branches[0].tcn2,
            "post_fusion": model.enhance,
        }
        for name, module in modules.items():
            def record(module, args, output, name=name):
                observed[name] = tuple(output.shape)
            handles.append(module.register_forward_hook(record))
        with torch.no_grad():
            model(torch.randn(1, 1, 1500))
        for handle in handles:
            handle.remove()
        self.assertEqual(observed, {
            "stem": (1, 64, 375),
            "block1": (1, 64, 375),
            "block2": (1, 128, 375),
            "post_fusion": (1, 256, 375),
        })
        self.assertEqual(model.branches[0].tcn1.intermediate.kernel_size, (1,))
        self.assertEqual(model.classifier[4].p, 0.2)
        self.assertTrue(all(isinstance(norm, nn.InstanceNorm1d) for norm in model.fusion.branch_norms))

    def test_gradient_and_checkpoint_round_trip(self):
        torch.manual_seed(7)
        model = MWAFNet(5).set_epoch(8).train()
        x = torch.randn(2, 1, 1500)
        target = torch.tensor([0, 4])
        loss = nn.CrossEntropyLoss()(model(x), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        parameters = dict(model.named_parameters())
        for name in (
            "branches.0.input_conv.0.conv1.weight",
            "branches.0.input_conv.0.se.gate.1.weight",
            "branches.0.input_conv.0.mcfe.gate",
            "branches.0.tcn1.intermediate.weight",
            "fusion.ca.g_att",
            "fusion.branch_weights",
            "classifier.5.weight",
        ):
            grad = parameters[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        before = parameters["branches.0.input_conv.0.conv1.weight"].detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, parameters["branches.0.input_conv.0.conv1.weight"]))
        buffer = io.BytesIO()
        torch.save({
            "version": MODEL_CONFIG_VERSION,
            "num_classes": 5,
            "model_config": json.loads(json.dumps(model.configuration())),
            "state_dict": model.state_dict(),
        }, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        clone = MWAFNet(checkpoint["num_classes"], **checkpoint["model_config"]).eval()
        clone.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        self.assertEqual(int(clone.training_epoch), 8)
        self.assertAlmostEqual(clone.enhance.dropblock.drop_prob, 0.08, places=6)
        with torch.no_grad():
            torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)

    def test_coordinate_attention_uses_learnable_mixture(self):
        attention = CoordAttention1D(4).eval()
        with torch.no_grad():
            attention.fc_x.weight.zero_()
            attention.fc_c.weight.zero_()
            attention.fc_x.bias.fill_(math.log(0.8 / 0.2))
            attention.fc_c.bias.fill_(math.log(0.2 / 0.8))
            attention.g_att.fill_(math.log(0.25 / 0.75))
        x = torch.ones(2, 4, 9)
        actual = attention(x)
        # 0.25 * 0.8 + 0.75 * 0.2 = 0.35; the old product was 0.16.
        torch.testing.assert_close(actual, x * 0.35)
        actual.sum().backward()
        self.assertGreater(float(attention.g_att.grad.abs()), 0.0)

    def test_dropblock_epoch_ramp_and_one_dimensional_probability(self):
        model = MWAFNet(2)
        for epoch, expected in ((0, 0.0), (1, 0.01), (8, 0.08), (15, 0.15), (100, 0.15)):
            model.set_epoch(epoch)
            probabilities = [m.drop_prob for m in model.modules() if isinstance(m, DropBlock1D)]
            self.assertEqual(len(probabilities), 13)
            for probability in probabilities:
                self.assertAlmostEqual(probability, expected, places=6)
        drop = DropBlock1D(0.15, block_size=3).train()
        draws = torch.ones(1, 1, 9)
        draws[..., 4] = 0.03  # Below p / K = .05, above the obsolete p / K^2.
        with patch("mwaf.model.torch.rand", return_value=draws):
            output = drop(torch.ones(1, 4, 9))
        expected = torch.full((1, 4, 9), 1.5)
        expected[..., 3:6] = 0.0
        torch.testing.assert_close(output, expected)

    def test_fully_masked_dropblock_is_finite(self):
        drop = DropBlock1D(1.0, block_size=3).train()
        x = torch.randn(2, 3, 8, requires_grad=True)
        with patch("mwaf.model.torch.rand", return_value=torch.zeros(2, 1, 8)):
            output = drop(x)
        torch.testing.assert_close(output, torch.zeros_like(x))
        output.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        drop.eval()
        self.assertIs(drop(x), x)


if __name__ == "__main__":
    unittest.main()
