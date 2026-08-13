# Barricade Zero

A tested, dependency-free reference engine forming the first milestone of an AlphaZero-style Barricade/Quoridor project.

Implemented:

- Configurable odd-sized boards, including official 9x9 with 10 walls
- Fixed action space `8 + 2(N-1)^2` (136 actions on 9x9)
- Ordinary pawn movement, straight jumps, and diagonal go-arounds
- Horizontal and vertical walls with overlap/crossing rejection
- BFS validation that both players retain a route to goal
- Immutable state transitions and terminal outcomes
- Canonical side-to-move rotation and policy rotation
- Eight-plane neural encoding and legal-action mask
- Stable serialization/key generation
- Random and shortest-path baseline agents
- Reference PUCT MCTS with root noise and temperature control
- Canonical self-play trajectory generation `(state, MCTS policy, outcome)`
- Configurable residual policy-value network and AdamW learner
- Replay buffer and executable 5x5 end-to-end training smoke cycle
- Neural-network-guided self-play generations
- Balanced deterministic checkpoint arena and promotion threshold
- Atomic generation checkpoints with resume support
- Wave-batched MCTS across concurrent self-play games
- Inference batching metrics and sequential-vs-batched benchmark

Create the training environment:

```bash
python3 -m venv .venv
uv pip install --python .venv/bin/python -e .
```

Run the tests:

```bash
python3 -m unittest discover -s tests -v
```

Run the random-game stress test:

```bash
python3 scripts/stress_random_games.py --games 1000
```

Run a tiny CPU training cycle:

```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/train_smoke.py
```

Run one small neural self-play generation:

```bash
PYTHONPATH=. python scripts/train_generations.py \
  --generations 1 --games 2 --simulations 8 --steps 4 --arena-games 2
```

Resume from a checkpoint:

```bash
PYTHONPATH=. python scripts/train_generations.py \
  --resume checkpoints/generations/generation_001.pt
```

Benchmark sequential versus batched inference:

```bash
PYTHONPATH=. python scripts/benchmark_batching.py --games 8 --simulations 16
```

The next milestone is persistent MCTS tree reuse, larger concurrent game pools, and stronger checkpoint tournaments. The reference search is intentionally readable Python; production 9x9 self-play will later need a compiled search path.
