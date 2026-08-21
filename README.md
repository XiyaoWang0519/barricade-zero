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
- Wave-batched two-player training arena and checkpoint evaluation matches
- Atomic generation checkpoints with resume support
- Wave-batched MCTS across concurrent self-play games
- Inference batching metrics and sequential-vs-batched benchmark
- Persistent MCTS subtree reuse between moves
- Tree-reuse benchmark and retained-root visit metrics
- Reproducible 5x5/9x9 cProfile harness
- Bitset candidate-wall path checking and optional native-backend boundary
- Dependency-free C++ rules backend loaded through ctypes with Python fallback
- Differential-tested native 8-plane encoder including BFS distance features
- Contiguous native float32 batch encoding with zero-copy CPU tensors
- Native CSR legal-action batching with strict NumPy/Python scalar boundaries
- Cached involutive policy/action rotation permutations

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

Run the synchronized full-workload CUDA benchmark (on an NVIDIA CUDA host):

```bash
PYTHONPATH=. python scripts/profile_self_play.py \
  --board-size 9 --walls 10 --games 128 --simulations 8 \
  --channels 64 --blocks 6 --device cuda --mixed-precision \
  --profile profiles/9x9-64x6-cuda.prof \
  --json-output profiles/9x9-64x6-cuda.json --top 18
```

The JSON report includes synchronized model-forward time, complete workload
throughput, peak allocated/reserved VRAM, and sampled GPU utilization.

Resume from a checkpoint:

```bash
PYTHONPATH=. python scripts/train_generations.py \
  --resume checkpoints/generations/generation_001.pt
```

Launch the gameplay UI to watch games, play against baselines, and inspect checkpoints:

```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/serve_ui.py --open --checkpoint-dir checkpoints
```

The board runs at `http://127.0.0.1:8765`. Human moves are clicks on highlighted squares or wall slots; checkpoint networks can be probed for policy, value, and MCTS visits.

Benchmark sequential versus batched inference:

```bash
PYTHONPATH=. python scripts/benchmark_batching.py --games 8 --simulations 16
```

Benchmark persistent-tree reuse:

```bash
PYTHONPATH=. python scripts/benchmark_tree_reuse.py --games 32 --simulations 64
```

The next milestone is stronger checkpoint tournaments and profiling the Python rules/search hot paths before a compiled implementation. With 64 concurrent 5x5 games, the current CPU pipeline reaches an average inference batch near 32; production 9x9 training should target 64–256.

See [`docs/performance.md`](docs/performance.md) for current 9x9 profiles,
optimization results, the native-extension boundary, and the GPU readiness gate.
