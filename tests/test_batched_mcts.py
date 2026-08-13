import random
import unittest

try:
    import torch
except ImportError:
    torch = None

from barricade.state import GameState


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BatchedMCTSTests(unittest.TestCase):
    def test_multiple_roots_share_model_forward_calls(self):
        from barricade.batched_mcts import BatchedMCTS
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(2)
        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        states = [GameState.initial(size=5, walls_per_player=0) for _ in range(4)]
        results = BatchedMCTS(evaluator, simulations=8, rng=random.Random(3)).search_batch(states)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(sum(result.visits) == 8 for result in results))
        self.assertLess(evaluator.forward_calls, evaluator.positions_evaluated)
        self.assertGreater(evaluator.average_batch_size, 1.0)

    def test_batched_search_finds_immediate_wins(self):
        from barricade.actions import UP
        from barricade.batched_mcts import BatchedMCTS
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        states = [GameState(size=5, pawns=((1, col), (4, 4)), walls_remaining=(0, 0)) for col in range(4)]
        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        results = BatchedMCTS(evaluator, simulations=24).search_batch(states, temperature=0)
        self.assertTrue(all(result.best_action == UP for result in results))

    def test_persistent_roots_reuse_selected_child_subtrees(self):
        from barricade.batched_mcts import BatchedMCTS
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        search = BatchedMCTS(evaluator, simulations=8, rng=random.Random(12))
        roots = search.create_roots([GameState.initial(size=5, walls_per_player=0)])
        first = search.search_roots(roots, temperature=0)[0]
        action = first.best_action
        selected_child = roots[0].children[action]
        prior_visits = sum(edge.visits for edge in selected_child.edges.values())
        advanced = search.advance_roots(roots, [action])
        self.assertIs(advanced[0], selected_child)
        search.search_roots(advanced, temperature=0)
        self.assertGreater(sum(edge.visits for edge in advanced[0].edges.values()), prior_visits)

    def test_advance_creates_child_when_action_was_not_visited(self):
        from barricade.batched_mcts import BatchedMCTS
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        state = GameState.initial(size=5, walls_per_player=0)
        search = BatchedMCTS(
            NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1)), simulations=1
        )
        roots = search.create_roots([state])
        action = state.legal_actions()[-1]
        advanced = search.advance_roots(roots, [action])
        self.assertEqual(advanced[0].state, state.apply_action(action))

    def test_search_accepts_native_rules_backend(self):
        from barricade.backend import NativeRulesBackend
        from barricade.batched_mcts import BatchedMCTS
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        state = GameState.initial(size=5, walls_per_player=2)
        search = BatchedMCTS(
            NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1)),
            simulations=2,
            rules_backend=NativeRulesBackend(),
        )
        result = search.search_batch([state])[0]
        self.assertIn(result.best_action, state.legal_actions())

    def test_native_batch_expansion_uses_builtin_integer_actions(self):
        from barricade.backend import NativeRulesBackend
        from barricade.batched_mcts import BatchedMCTS
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        search = BatchedMCTS(
            NeuralEvaluator(PolicyValueNetwork(9, channels=8, residual_blocks=1)),
            simulations=1,
            rules_backend=NativeRulesBackend(),
        )
        roots = search.create_roots([GameState.initial(size=9, walls_per_player=10)])
        search.search_roots(roots)
        self.assertTrue(all(type(action) is int for action in roots[0].edges))
        wall_action = next(action for action in roots[0].edges if action >= 8)
        child = roots[0].state.apply_known_legal_action(wall_action)
        self.assertTrue(
            all(type(value) is int for wall in child.horizontal_walls | child.vertical_walls for value in wall)
        )

    def test_batched_nodes_use_compact_edge_storage_with_mapping_contract(self):
        from barricade.batched_mcts import BatchedMCTS, CompactEdges
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        state = GameState.initial(size=5, walls_per_player=2)
        search = BatchedMCTS(
            NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1)),
            simulations=4,
        )
        roots = search.create_roots([state])
        search.search_roots(roots)
        edges = roots[0].edges
        self.assertIsInstance(edges, CompactEdges)
        self.assertEqual(list(edges), state.legal_actions())
        self.assertEqual(sum(edge.visits for edge in edges.values()), 4)
        self.assertAlmostEqual(sum(edge.prior for edge in edges.values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()