import random
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class LearnerTests(unittest.TestCase):
    def test_native_batch_assembly_matches_python_reference_exactly(self):
        from barricade.actions import action_size
        from barricade.encoding import encode_state
        from barricade.self_play import TrainingExample
        from barricade.state import GameState
        from neural.model import PolicyValueNetwork
        from training.learner import Learner

        first = GameState.initial(size=9, walls_per_player=10)
        second = first.apply_action(first.legal_actions()[0]).canonical()
        policy = [0.0] * action_size(9)
        policy[0] = 1.0
        examples = [
            TrainingExample(first, policy, 1.0),
            TrainingExample(second, policy, -1.0),
        ]
        learner = Learner(
            PolicyValueNetwork(board_size=9, channels=8, residual_blocks=1)
        )

        states, policies, outcomes = learner._batch(examples)
        expected_states = torch.tensor(
            [encode_state(example.state) for example in examples], dtype=torch.float32
        )
        expected_policies = torch.tensor(
            [example.policy for example in examples], dtype=torch.float32
        )
        expected_outcomes = torch.tensor(
            [[example.outcome] for example in examples], dtype=torch.float32
        )

        self.assertTrue(torch.equal(states.cpu(), expected_states))
        self.assertTrue(torch.equal(policies.cpu(), expected_policies))
        self.assertTrue(torch.equal(outcomes.cpu(), expected_outcomes))

    def test_training_step_updates_parameters_and_returns_finite_metrics(self):
        from barricade.mcts import MCTS, UniformEvaluator
        from barricade.self_play import play_self_play_game
        from barricade.state import GameState
        from neural.model import PolicyValueNetwork
        from training.learner import Learner, ReplayBuffer

        torch.manual_seed(1)
        examples = play_self_play_game(
            GameState.initial(size=5, walls_per_player=0),
            MCTS(UniformEvaluator(), simulations=8, rng=random.Random(2)),
            rng=random.Random(3),
        )
        replay = ReplayBuffer(capacity=100, rng=random.Random(4))
        replay.extend(examples)
        model = PolicyValueNetwork(board_size=5, channels=8, residual_blocks=1)
        learner = Learner(model, learning_rate=1e-3, weight_decay=1e-4)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        metrics = learner.train_batch(replay.sample(min(8, len(replay))))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_replay_buffer_capacity_and_seeded_sampling(self):
        from training.learner import ReplayBuffer

        first = ReplayBuffer(capacity=3, rng=random.Random(5))
        second = ReplayBuffer(capacity=3, rng=random.Random(5))
        for buffer in (first, second):
            buffer.extend(range(5))
        self.assertEqual(list(first), [2, 3, 4])
        self.assertEqual(first.sample(2), second.sample(2))


if __name__ == "__main__":
    unittest.main()
