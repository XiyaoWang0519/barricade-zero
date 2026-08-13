# Performance notes

Reproduce profiles with:

```bash
PYTHONPATH=. python scripts/profile_self_play.py \
  --board-size 9 --walls 10 --games 4 --simulations 4
```

## 9x9 profile, 2026-08-13

Fixed workload: 4 games, 4 simulations, 10 walls per player, seed 51.

| implementation | seconds | evaluated positions/s |
|---|---:|---:|
| original Python rules + duplicate legal mask | 58.65 | 8.73 |
| single legality scan | 40.24 | 12.72 |
| bitset edges + candidate-wall BFS | 17.52 | 29.23 |

The current implementation is 3.35x faster than the original measured path.
In the final profile, candidate-wall path checks remain the largest cumulative
cost at about 10.9 seconds. Neural forward is about 5.0 seconds under cProfile.

## Native extension boundary

The first compiled extension should implement the narrow interface in
`barricade/backend.py`:

1. `legal_actions(state)`
2. `has_path_with_extra_wall(state, player, orientation, row, col)`

Represent cells and blocked edges as integers/bitsets. Do not move PyTorch,
self-play orchestration, replay storage, or checkpoint logic into C++ yet.
The native backend must be differential-tested against `PythonRulesBackend`
over random reachable states before it becomes the default.

## GPU gate

Do not rent a GPU while rules/search dominate. Re-profile after native rules.
GPU training becomes worthwhile when neural inference is the sustained dominant
cost and concurrent 9x9 self-play keeps average inference batches at least
64 (preferably 128+) without starving the device.