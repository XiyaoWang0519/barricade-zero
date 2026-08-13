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
| C++ rules + known-legal state transition | 1.87 | 274.07 |
| C++ rules + C++ 8-plane encoding | 0.876 | 584.51 |
| contiguous float32 batch encoding, 128 games | 30.78 | 1121.20 |
| native CSR legal-action batches, 128 games | 28.77 | 1199.67 |
| cached policy rotation permutation, 128 games | 25.14 | 1373.08 |
| fixed-capacity native path-search storage, 128 games | 22.80 | 1514.08 |
| precomputed native open-edge graph, 128 games | 19.70 | 1751.79 |
| compact parallel-list MCTS edges, 128 games | 14.30 | 2412.83 |

The current small-batch implementation is 67.0x faster than the original
measured path. Native encoding is differential-tested plane-by-plane against
the Python reference, including both BFS distance planes.

At 128 concurrent 9x9 games, average inference batch size reaches 86.1.
Contiguous native `float32[N,8,S,S]` encoding plus `torch.from_numpy` raises
throughput from about 967 to 1,121 evaluated positions/s and reduces wall time
from 35.68 to 30.78 seconds. The CPU tensor shares the NumPy buffer without a
copy. The remaining profile is dominated by legal-action object creation,
policy rotation, PUCT selection, and Python tree traversal.

Native CSR legal-action batches reduce wall time from 30.78 to 28.77 seconds.
The CSR boundary uses contiguous `int32` offsets/actions and converts NumPy
scalars to built-in Python integers before they enter tree/domain state. This
conversion is required: allowing `np.int32` wall coordinates into `GameState`
causes fixed-width bit-shift overflow in the Python fallback.

Caching the involutive action permutation removes repeated per-policy action
decoding and lowers the same workload from 28.77 to 25.14 seconds. Policy
rotation no longer appears among the leading cumulative-time functions.

Replacing per-candidate heap-backed BFS containers in the native path check
with fixed-capacity stack storage (the backend's supported maximum is 9x9)
lowers the fixed 128-game workload from 25.14 to 22.80 seconds, a 9.3% wall-time
reduction, and raises throughput from 1,373 to 1,514 positions/s. An expanded
128-state differential test now samples up to 59 reachable plies and compares
the batch result against the Python backend. Native batch legality still leads
CPU cost at 4.73 seconds; Python expansion/tree selection account for roughly
4.76/4.54 seconds, while model forward is 2.14 seconds.

Precomputing each state's open-edge adjacency once, then copying the 81-byte
table and removing the candidate wall's two edges, lowers the same workload
from 22.80 to 19.70 seconds. Native batch legality falls from 4.73 to 2.14
seconds. Python tree descent/selection (about 5.93/4.52 seconds) is now the
largest sustained bottleneck; model forward is about 2.19 seconds.

Replacing batched-MCTS `dict[int, EdgeStats]` storage with parallel action,
prior, visit, and value arrays lowers the workload from 19.70 to 14.30 seconds.
PUCT selection falls from about 4.52 to 1.43 seconds and tree descent from 5.93
to 2.77 seconds. A mapping-compatible facade preserves tree-reuse and arena
callers while the hot path uses indices directly. Evaluator/model forward now
take about 4.53/2.09 seconds; native legality takes about 2.12 seconds.

## Native extension boundary

The first compiled extension implements the narrow interface in
`barricade/backend.py`:

1. `legal_actions(state)`
2. `has_path_with_extra_wall(state, player, orientation, row, col)`

Represent cells and blocked edges as integers/bitsets. Do not move PyTorch,
self-play orchestration, replay storage, or checkpoint logic into C++ yet.
The native backend is differential-tested against `PythonRulesBackend` over
random reachable 5x5 and 9x9 states. It is loaded through `ctypes`, requires no
third-party binding package, and falls back to Python when the shared library
is absent. Build it with `python scripts/build_native.py`.

## GPU gate

Do not rent a GPU while rules/search dominate. Re-profile after native rules.
The batch-size half of the GPU gate is satisfied: 128 concurrent games produce
an average batch of 86.1. However, model forward is only about 4.14 of 30.78
seconds in the pre-CSR workload, 2.47 of 28.77 seconds after CSR batching, and
2.09 of 14.30 seconds after compact MCTS edges (roughly 15%). GPU rental
should wait until
batched/native legal-action generation and tree-search work make model forward
the sustained dominant cost.