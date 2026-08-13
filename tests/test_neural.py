import unittest

try:
    import torch
except ImportError:  # Allows the dependency-free engine suite to keep running.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PolicyValueNetworkTests(unittest.TestCase):
    def test_forward_shapes_and_value_range(self):
        from neural.model import PolicyValueNetwork

        model = PolicyValueNetwork(board_size=5, channels=16, residual_blocks=2)
        policy, value = model(torch.randn(3, 8, 5, 5))
        self.assertEqual(tuple(policy.shape), (3, 40))
        self.assertEqual(tuple(value.shape), (3, 1))
        self.assertTrue(torch.all(value >= -1))
        self.assertTrue(torch.all(value <= 1))

    def test_network_supports_official_action_size(self):
        from neural.model import PolicyValueNetwork

        model = PolicyValueNetwork(board_size=9, channels=8, residual_blocks=1)
        policy, value = model(torch.zeros(2, 8, 9, 9))
        self.assertEqual(tuple(policy.shape), (2, 136))
        self.assertEqual(tuple(value.shape), (2, 1))

    def test_masked_policy_has_no_illegal_probability(self):
        from neural.model import masked_softmax

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, False, True]])
        policy = masked_softmax(logits, mask)
        self.assertEqual(policy[0, 1].item(), 0.0)
        self.assertAlmostEqual(policy.sum().item(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()