import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from barricade.actions import rotate_action
from barricade.state import GameState


@unittest.skipIf(torch is None, "PyTorch is not installed")
class NeuralEvaluatorTests(unittest.TestCase):
    def test_evaluation_mode_is_not_reapplied_for_every_batch(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        class CountingNetwork(PolicyValueNetwork):
            def __init__(self):
                self.train_calls = 0
                super().__init__(5, channels=8, residual_blocks=1)

            def train(self, mode=True):
                self.train_calls += 1
                return super().train(mode)

        model = CountingNetwork()
        evaluator = NeuralEvaluator(model)
        calls_after_initialization = model.train_calls
        states = [GameState.initial(size=5, walls_per_player=0)]
        evaluator.evaluate_batch_arrays(states)
        evaluator.evaluate_batch_arrays(states)
        self.assertEqual(model.train_calls, calls_after_initialization)
        model.train()
        evaluator.evaluate_batch_arrays(states)
        self.assertEqual(model.train_calls, calls_after_initialization + 2)
        self.assertFalse(model.training)

    def test_prepared_batch_evaluation_returns_matching_legality(self):
        from barricade.backend import NativeRulesBackend
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        states = [GameState.initial(size=5, walls_per_player=2)]
        states.append(states[0].apply_action(states[0].legal_actions()[0]))
        backend = NativeRulesBackend()
        evaluator = NeuralEvaluator(
            PolicyValueNetwork(5, channels=8, residual_blocks=1),
            encoding_backend=backend,
        )
        policies, values, offsets, actions = (
            evaluator.evaluate_prepared_batch_arrays(states)
        )
        expected_offsets, expected_actions = backend.legal_actions_batch(states)
        self.assertEqual(policies.shape, (2, states[0].action_size))
        self.assertEqual(values.shape, (2,))
        np.testing.assert_array_equal(offsets, expected_offsets)
        np.testing.assert_array_equal(actions, expected_actions)

    def test_policy_is_legal_and_normalized_for_both_players(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(4)
        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        first = GameState.initial(size=5, walls_per_player=2)
        second = first.apply_action(first.legal_pawn_actions()[0])
        for state in (first, second):
            policy, value = evaluator.evaluate(state)
            self.assertAlmostEqual(sum(policy), 1.0, places=6)
            self.assertTrue(-1 <= value <= 1)
            legal = set(state.legal_actions())
            self.assertTrue(all(policy[action] == 0 for action in range(state.action_size) if action not in legal))

    def test_player_two_policy_is_rotated_back_to_original_coordinates(self):
        from neural.evaluator import NeuralEvaluator

        class FixedModel(torch.nn.Module):
            def forward(self, inputs):
                logits = torch.arange(40, dtype=torch.float32).repeat(inputs.shape[0], 1)
                return logits, torch.zeros(inputs.shape[0], 1)

        state = GameState.initial(size=5, walls_per_player=0).apply_action(0)
        evaluator = NeuralEvaluator(FixedModel())
        policy, _ = evaluator.evaluate(state)
        canonical = state.canonical()
        canonical_policy, _ = NeuralEvaluator(FixedModel()).evaluate(canonical)
        for action in state.legal_actions():
            self.assertAlmostEqual(policy[action], canonical_policy[rotate_action(action, 5)])

    def test_batch_evaluation_matches_individual_evaluation(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(8)
        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        first = GameState.initial(size=5, walls_per_player=0)
        second = first.apply_action(first.legal_pawn_actions()[0])
        batched = evaluator.evaluate_batch([first, second])
        individual = [evaluator.evaluate(first), evaluator.evaluate(second)]
        for (batch_policy, batch_value), (single_policy, single_value) in zip(batched, individual):
            self.assertEqual(len(batch_policy), len(single_policy))
            for actual, expected in zip(batch_policy, single_policy):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertAlmostEqual(batch_value, single_value, places=6)

    def test_batch_metrics_count_forward_calls_and_positions(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        states = [GameState.initial(size=5, walls_per_player=0)] * 3
        evaluator.evaluate_batch(states)
        evaluator.evaluate(states[0])
        self.assertEqual(evaluator.forward_calls, 2)
        self.assertEqual(evaluator.positions_evaluated, 4)
        self.assertEqual(evaluator.average_batch_size, 2.0)

    def test_unmasked_batch_skips_legal_action_generation(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        state = GameState.initial(size=5, walls_per_player=2)
        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        policy, value = evaluator.evaluate_batch([state], mask_legal=False)[0]
        self.assertAlmostEqual(sum(policy), 1.0, places=6)
        self.assertTrue(-1 <= value <= 1)

    def test_native_and_python_encoding_produce_same_evaluation(self):
        from barricade.backend import NativeRulesBackend, PythonRulesBackend
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(301)
        model = PolicyValueNetwork(9, channels=8, residual_blocks=1)
        states = [GameState.initial(size=9, walls_per_player=10)]
        python_result = NeuralEvaluator(
            model, encoding_backend=PythonRulesBackend()
        ).evaluate_batch(states, mask_legal=False)
        native_result = NeuralEvaluator(
            model, encoding_backend=NativeRulesBackend()
        ).evaluate_batch(states, mask_legal=False)
        self.assertEqual(python_result, native_result)

    def test_cpu_input_tensor_shares_native_batch_buffer(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        evaluator = NeuralEvaluator(PolicyValueNetwork(5, channels=8, residual_blocks=1))
        states = [GameState.initial(size=5, walls_per_player=2)] * 2
        array, tensor = evaluator.encode_inputs(states)
        self.assertEqual(tensor.data_ptr(), array.ctypes.data)
        self.assertEqual(tuple(tensor.shape), (2, 8, 5, 5))

    def test_array_evaluation_matches_public_list_results(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(71)
        model = PolicyValueNetwork(5, channels=8, residual_blocks=1)
        states = [
            GameState.initial(size=5, walls_per_player=2),
            GameState.initial(size=5, walls_per_player=2).apply_action(0),
        ]
        array_evaluator = NeuralEvaluator(model)
        policies, values = array_evaluator.evaluate_batch_arrays(states, mask_legal=False)
        list_evaluator = NeuralEvaluator(model)
        expected = list_evaluator.evaluate_batch(states, mask_legal=False)
        self.assertEqual(policies.dtype, np.float32)
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(policies.flags.c_contiguous)
        self.assertEqual(policies.shape, (2, 40))
        for index, (policy, value) in enumerate(expected):
            np.testing.assert_allclose(policies[index], policy, rtol=1e-6, atol=1e-7)
            self.assertAlmostEqual(float(values[index]), value, places=6)

    def test_cpu_evaluator_records_forward_time_and_fp32_outputs(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        evaluator = NeuralEvaluator(
            PolicyValueNetwork(5, channels=8, residual_blocks=1),
            mixed_precision=True,
        )
        policies, values = evaluator.evaluate_batch_arrays(
            [GameState.initial(size=5, walls_per_player=0)]
        )
        self.assertFalse(evaluator.mixed_precision)
        self.assertEqual(policies.dtype, np.float32)
        self.assertEqual(values.dtype, np.float32)
        self.assertGreaterEqual(evaluator.model_forward_seconds, 0.0)


@unittest.skipUnless(
    torch is not None and torch.cuda.is_available(), "CUDA is not available"
)
class NeuralEvaluatorCudaTests(unittest.TestCase):
    def test_cpu_and_cuda_agree_on_masks_rotation_policy_and_value(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        torch.manual_seed(401)
        cpu_model = PolicyValueNetwork(5, channels=8, residual_blocks=1)
        cuda_model = PolicyValueNetwork(5, channels=8, residual_blocks=1)
        cuda_model.load_state_dict(cpu_model.state_dict())
        first = GameState.initial(size=5, walls_per_player=2)
        states = [first, first.apply_action(first.legal_pawn_actions()[0])]
        cpu_policies, cpu_values = NeuralEvaluator(
            cpu_model, device="cpu"
        ).evaluate_batch_arrays(states)
        cuda_policies, cuda_values = NeuralEvaluator(
            cuda_model, device="cuda", mixed_precision=False
        ).evaluate_batch_arrays(states)
        np.testing.assert_allclose(cuda_policies, cpu_policies, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cuda_values, cpu_values, rtol=1e-4, atol=1e-5)
        for index, state in enumerate(states):
            legal = set(state.legal_actions())
            self.assertTrue(
                all(
                    cuda_policies[index, action] == 0
                    for action in range(state.action_size)
                    if action not in legal
                )
            )

    def test_cuda_mixed_precision_uses_one_batch_transfer_and_fp32_outputs(self):
        from neural.evaluator import NeuralEvaluator
        from neural.model import PolicyValueNetwork

        class RecordingModel(PolicyValueNetwork):
            def forward(self, inputs):
                self.input_device = inputs.device
                self.input_dtype = inputs.dtype
                self.cuda_autocast_enabled = torch.is_autocast_enabled("cuda")
                return super().forward(inputs)

        model = RecordingModel(5, channels=8, residual_blocks=1)
        evaluator = NeuralEvaluator(model, device="cuda", mixed_precision=True)
        states = [GameState.initial(size=5, walls_per_player=2)] * 3
        policies, values = evaluator.evaluate_batch_arrays(states)
        self.assertEqual(model.input_device.type, "cuda")
        self.assertEqual(model.input_dtype, torch.float32)
        self.assertTrue(model.cuda_autocast_enabled)
        self.assertEqual(evaluator.host_to_device_transfers, 2)
        self.assertEqual(policies.dtype, np.float32)
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_allclose(policies.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
