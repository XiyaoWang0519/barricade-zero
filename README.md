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

Run the tests:

```bash
python3 -m unittest discover -s tests -v
```

Run the random-game stress test:

```bash
python3 scripts/stress_random_games.py --games 1000
```

The next milestone is a small PyTorch policy-value network and replay learner on the 5x5 configuration before scaling to 9x9. The reference MCTS is intentionally readable Python; production self-play will later need batching and a compiled search path.
