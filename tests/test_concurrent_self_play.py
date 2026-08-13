import random
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ConcurrentSelfPlayTests(unittest.TestCase):
    def test_concurrent_games_emit_complete_training_examples(self):
        from barricade.concurrent_self_play import play_concurrent_games
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        result = play_concurrent_games(
            evaluator,
            games=4,
            simulations=4,
            board_size=5,
            walls_per_player=0,
            rng=random.Random(9),
            max_plies=100,
        )
        self.assertEqual(result.games, 4)
        self.assertEqual(len(result.game_lengths), 4)
        self.assertEqual(len(result.examples), sum(result.game_lengths))
        self.assertTrue(all(example.state.turn == 0 for example in result.examples))
        self.assertGreater(result.average_inference_batch_size, 1.0)
        self.assertGreater(result.tree_nodes_created, 0)
        self.assertGreater(result.tree_expansions, 0)
        self.assertGreaterEqual(result.tree_traversal_seconds, 0.0)
        self.assertGreaterEqual(result.expansion_seconds, 0.0)

    def test_draw_outcomes_are_zero_for_repetitions(self):
        from barricade.concurrent_self_play import play_concurrent_games
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        result = play_concurrent_games(
            evaluator,
            games=2,
            simulations=1,
            board_size=5,
            walls_per_player=0,
            rng=random.Random(1),
            max_plies=8,
            max_plies_as_draw=True,
        )
        self.assertEqual(result.games, 2)
        self.assertTrue(all(example.outcome in (-1.0, 0.0, 1.0) for example in result.examples))

    def test_tree_reuse_retains_search_work(self):
        from barricade.concurrent_self_play import play_concurrent_games
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(17)
        model = PolicyValueNetwork(5, channels=8, residual_blocks=1)
        without = play_concurrent_games(
            NeuralEvaluator(model), games=4, simulations=8, board_size=5,
            walls_per_player=0, rng=random.Random(18), use_tree_reuse=False,
        )
        with_reuse = play_concurrent_games(
            NeuralEvaluator(model), games=4, simulations=8, board_size=5,
            walls_per_player=0, rng=random.Random(18), use_tree_reuse=True,
        )
        self.assertGreater(with_reuse.reused_root_visits, 0)
        self.assertGreater(with_reuse.positions_evaluated, 0)
        self.assertEqual(with_reuse.games, without.games)

    def test_native_search_matches_seeded_python_self_play(self):
        from barricade.backend import NativeRulesBackend
        from barricade.concurrent_self_play import play_concurrent_games
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(619)
        model = PolicyValueNetwork(5, channels=8, residual_blocks=1)

        def run(use_native_search):
            backend = NativeRulesBackend()
            return play_concurrent_games(
                NeuralEvaluator(model, encoding_backend=backend),
                games=8,
                simulations=8,
                board_size=5,
                walls_per_player=2,
                rng=random.Random(620),
                max_plies=80,
                rules_backend=backend,
                use_native_search=use_native_search,
            )

        python = run(False)
        native = run(True)
        self.assertEqual(native.game_lengths, python.game_lengths)
        self.assertEqual(native.wins, python.wins)
        self.assertEqual(native.draws, python.draws)
        self.assertEqual(native.positions_evaluated, python.positions_evaluated)
        self.assertEqual(native.forward_calls, python.forward_calls)
        self.assertEqual(native.reused_root_visits, python.reused_root_visits)
        self.assertEqual(native.search_backend, "native")
        self.assertEqual(python.search_backend, "python")
        self.assertEqual(len(native.examples), len(python.examples))
        for native_example, python_example in zip(native.examples, python.examples):
            self.assertEqual(native_example.state, python_example.state)
            self.assertEqual(native_example.outcome, python_example.outcome)
            self.assertEqual(native_example.policy, python_example.policy)


if __name__ == "__main__":
    unittest.main()
