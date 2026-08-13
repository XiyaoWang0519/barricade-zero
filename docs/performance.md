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
seconds in the pre-CSR workload and about 2.47 of 28.77 seconds after CSR
batching. GPU rental should wait until
batched/native legal-action generation and tree-search work make model forward
the sustained dominant cost.