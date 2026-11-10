"""AlexNet shape and gradient regression tests."""

import unittest

import torch
from torch import nn

from model import AlexNet


class AlexNetTests(unittest.TestCase):
    def test_resolutions_and_training_step(self):
        network = AlexNet(number_of_classes=3).eval()
        with torch.no_grad():
            for size in (128, 224, 256):
                with self.subTest(image_size=size):
                    logits = network(torch.zeros(1, 3, size, size))
                    self.assertEqual(tuple(logits.shape), (1, 3))
                    self.assertTrue(torch.isfinite(logits).all())

        network.train()
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        before = network.classifier[-1].weight.detach().clone()
        loss = nn.CrossEntropyLoss()(network(torch.rand(1, 3, 256, 256)), torch.tensor([1]))
        loss.backward()
        self.assertIsNotNone(network.features[0].weight.grad)
        self.assertGreater(network.features[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in network.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, network.classifier[-1].weight))


if __name__ == "__main__":
    unittest.main()
