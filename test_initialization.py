"""Regression checks for the old constructor's wavelet-overwrite defect."""

import unittest

import numpy as np
import pywt
import torch

from mwaf.model import MWAFNet, MultiScaleWaveletConv1d
from mwaf.wavelets import generate_wavelet_kernel


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_templates_survive_complete_network_initialization(self):
        model = MWAFNet()
        layer_count = 0
        families = set()
        for layer in model.modules():
            if not isinstance(layer, MultiScaleWaveletConv1d):
                continue
            layer_count += 1
            families.add(layer.wavelet)
            for index in (1, 2, 3):
                expected = generate_wavelet_kernel(
                    layer.wavelet, layer.kernel_size,
                    getattr(layer, f"scale{index}"), getattr(layer, f"shift{index}"),
                )
                torch.testing.assert_close(
                    getattr(layer, f"conv{index}").weight, expected, rtol=0, atol=0,
                )
        self.assertEqual(layer_count, 32)
        self.assertEqual(families, {"morl", "db4", "sym4", "coif4", "haar", "mexh"})

    def test_scales_and_shifts_are_initialization_only(self):
        layer = MultiScaleWaveletConv1d(1, 64, 7, padding=3).eval()
        parameters = dict(layer.named_parameters())
        buffers = dict(layer.named_buffers())
        for index, expected_scale in ((1, 1.0), (2, 2.0), (3, 4.0)):
            self.assertNotIn(f"scale{index}", parameters)
            self.assertNotIn(f"shift{index}", parameters)
            self.assertFalse(buffers[f"scale{index}"].requires_grad)
            self.assertTrue(torch.all(buffers[f"scale{index}"] == expected_scale))
        x = torch.randn(2, 1, 25)
        with torch.no_grad():
            before = layer(x)
            layer.scale1.fill_(20.0)
            layer.shift1.fill_(3.0)
            after = layer(x)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        layer(x).square().mean().backward()
        self.assertIsNotNone(layer.conv1.weight.grad)

    def test_compact_templates_use_phi_prefix_without_resampling(self):
        for family in ("db4", "sym4", "coif4", "haar"):
            for kernel_size in (3, 7):
                with self.subTest(family=family, kernel_size=kernel_size):
                    phi = pywt.Wavelet(family).wavefun(level=kernel_size // 2)[0]
                    reference = np.asarray(phi[:kernel_size], dtype=np.float64)
                    reference /= np.linalg.norm(reference) + 1e-8
                    actual = generate_wavelet_kernel(
                        family, kernel_size, torch.full((2, 1, 1), 4.0),
                        torch.full((2, 1, 1), 2.0),
                    )
                    torch.testing.assert_close(
                        actual[0, 0], torch.tensor(reference, dtype=torch.float32),
                        rtol=2e-6, atol=1e-7,
                    )
                    torch.testing.assert_close(actual[0], actual[1], rtol=0, atol=0)

    def test_analytic_template_grid_and_normalization(self):
        t = np.linspace(-4, 3, 7)
        for family in ("morl", "mexh"):
            normalized_t = t / 2.0
            envelope = np.exp(-normalized_t ** 2 / 2)
            reference = (np.cos(1.75 * normalized_t) if family == "morl"
                         else 1 - normalized_t ** 2) * envelope
            reference /= np.linalg.norm(reference) + 1e-8
            actual = generate_wavelet_kernel(
                family, 7, torch.full((1, 1, 1), 2.0), torch.zeros(1, 1, 1),
            )
            torch.testing.assert_close(
                actual[0, 0], torch.tensor(reference, dtype=torch.float32),
                rtol=2e-6, atol=1e-7,
            )


if __name__ == "__main__":
    unittest.main()
