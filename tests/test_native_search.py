import random
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from barricade.state import GameState


@unittest.skipIf(torch is None, "PyTorch is not installed")
class NativeSearchTests(unittest.TestCase):
    def test_native_wave_matches_python_search_and_successor_states(self):
        from barricade.backend import NativeRulesBackend
        from barricade.batched_mcts import BatchedMCTS
        from barricade.native_search import NativeSearchSession
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        for size, walls, count in ((5, 2, 4), (9, 10, 2)):
            with self.subTest(size=size):
                torch.manual_seed(712)
                model = PolicyValueNetwork(size, channels=8, residual_blocks=1)
                states = [GameState.initial(size=size, walls_per_player=walls) for _ in range(count)]

                native_backend = NativeRulesBackend()
                native_evaluator = NeuralEvaluator(model, encoding_backend=native_backend)
                native = NativeSearchSession(states, native_backend)
                native_results = native.search_roots(
                    native.root_ids,
                    native_evaluator,
                    simulations=8,
                    temperature=1.0,
                    add_noise=False,
                    rng=random.Random(44),
                )

                python_backend = NativeRulesBackend()
                python_evaluator = NeuralEvaluator(model, encoding_backend=python_backend)
                python_search = BatchedMCTS(
                    python_evaluator,
                    simulations=8,
                    rng=random.Random(44),
                    rules_backend=python_backend,
                )
                python_results = python_search.search_batch(states, temperature=1.0)

                for native_result, python_result in zip(native_results, python_results):
                    self.assertEqual(native_result.visits, python_result.visits)
                    np.testing.assert_allclose(
                        native_result.root_priors,
                        python_result.root_priors,
                        rtol=1e-6,
                        atol=1e-8,
                    )
                    np.testing.assert_allclose(
                        native_result.policy, python_result.policy, rtol=0.0, atol=0.0
                    )

                actions = [result.best_action for result in native_results]
                advanced, reused = native.advance(native.root_ids, actions)
                self.assertEqual(
                    native.states(advanced),
                    [
                        state.apply_known_legal_action(action)
                        for state, action in zip(states, actions)
                    ],
                )
                self.assertTrue(all(value >= 0 for value in reused))
                self.assertGreater(native.metrics.nodes, len(states))
                self.assertGreater(native.metrics.expansions, 0)
                native.close()


if __name__ == "__main__":
    unittest.main()
