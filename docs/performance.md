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
| contiguous evaluator outputs, 128 games | 14.26 | 2420.19 |
| single-FFI native batch encoding, 128 games | 14.40 | 2397.46 |
| vectorized legal-prior normalization, 128 games | 13.40 | 2575.57 |
| candidate 64x6 network, original CPU baseline | 30.24 | 1320.76 |
| candidate 64x6 network, post-CUDA-path CPU control | 16.99 | 2350.46 |
| candidate 64x6 network, RTX 3090 CUDA FP16 | 13.07 | 3030.12 |

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

Keeping evaluator policies/values as contiguous NumPy arrays through MCTS
expansion preserves numerical equivalence and the public list API, but changes
the fixed workload only from 14.30 to 14.26 seconds (about 0.3%). This confirms
that output list materialization is not a material bottleneck. The array path
is retained as the cleaner future CPU-to-GPU batch boundary.

A true native batch-encoding ABI reduces ctypes crossings and keeps a single
contiguous `float32[N,8,S,S]` output buffer. On the fixed workload it is
performance-neutral within run noise (14.26 versus 14.40 seconds): packing the
structure-of-arrays arguments offsets the saved calls. It is retained because
it preserves differential equivalence and is the required whole-batch input
boundary for a future GPU path, not because it improves the current CPU run.

Vectorized legal-action indexing and prior normalization lowers the 16x2
search workload from 14.40 to 13.40 seconds. Expansion falls to about 1.34
seconds and the measured call count drops by roughly 1.8 million.

The historical fixed profile deliberately used a 16-channel, 2-block smoke
network to expose rules and search overhead. The harness now records and accepts
network dimensions explicitly. With a candidate training network of 64 channels
and 6 residual blocks, the 128-game workload averages a 90.55-position inference
batch and takes 30.24 seconds. Model forward alone takes 17.36 seconds (about
57% of wall time), while expansion is about 4.04 seconds and all other individual
hotspots are smaller. This is the first configuration that satisfies the full
GPU-readiness gate.

After adding the explicit CUDA boundary, the exact 64x6 control workload was
rerun on the development Mac in a recreated Python 3.14.4 / PyTorch 2.13.0
environment. It completed in 16.99 seconds at 2,350.46 evaluated positions/s,
with 39,934 positions, 441 forward calls, an average batch of 90.55, and 13.55
seconds of measured model-forward time. This confirms no CPU regression, but
it is not an apples-to-apples optimization claim against the original 30.24
second run because the Python/PyTorch environment was recreated.

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
The GPU gate is now satisfied for the candidate 64x6 training network. At 128
concurrent games the average inference batch is 90.55 and model forward is
17.36 of 30.24 seconds (about 57%), making it the sustained dominant cost.
Rules, encoding, checkpoint/resume, arena, and real optimizer/checkpoint learning
loops have all passed their verification gates. Further throughput work should
therefore move model inference/training to a CUDA GPU rather than continue CPU
micro-optimization of the 16x2 smoke profile.

## CUDA evaluator and benchmark contract, 2026-08-13

The evaluator now has an explicit CUDA mixed-precision path. It transfers one
contiguous encoded batch and one contiguous legal-mask batch to the device,
runs model inference under CUDA float16 autocast when requested, and converts
logits and values back to float32 before policy normalization and host output.
CUDA events are synchronized around model-forward timing so reported forward
seconds do not measure only asynchronous kernel dispatch.

The profile harness accepts `--device cuda` and `--mixed-precision`. It records
the GPU name and total VRAM, full wall time, positions and forward calls,
average inference batch, synchronized model-forward time, peak allocated and
reserved VRAM, and sampled average/maximum utilization. `--json-output`
persists the summary for comparison with the CPU baseline.

```bash
PYTHONPATH=. python scripts/profile_self_play.py \
  --board-size 9 --walls 10 --games 128 --simulations 8 \
  --channels 64 --blocks 6 --device cuda --mixed-precision \
  --profile profiles/9x9-64x6-cuda.prof \
  --json-output profiles/9x9-64x6-cuda.json --top 18
```

CUDA-only tests cover CPU/CUDA numerical agreement, legal masks on both turns,
player-two policy rotation, input device/dtype, mixed-precision autocast with
float32 outputs, fixed whole-batch transfer counts, and CPU/CUDA checkpoint
loading. They skip cleanly on non-CUDA hosts.

### Measured RTX 3090 result

The complete workload ran on a RunPod Secure Cloud RTX 3090 with 24 GB VRAM,
driver 580.159.03, PyTorch 2.4.1+cu124, and CUDA 12.4. All three CUDA-only
evaluator/checkpoint tests passed before benchmarking.

The mixed-precision 64x6 benchmark produced:

- 13.07 seconds full wall time
- 39,593 evaluated positions and 489 forward calls
- 3,030.12 positions/s with average batch 80.97
- 2.57 seconds synchronized model-forward time
- 24.4 MB peak allocated and 50 MB peak reserved VRAM
- 3.61% average and 8% peak sampled GPU utilization

Model forward is about 5.3x faster than the 13.55-second current CPU control,
but complete workload throughput improves only about 1.3x over that control
(and about 2.3x over the original CPU baseline). Mixed precision changes small
policy values enough to produce a slightly different deterministic trajectory,
so compare the configured workload and throughput rather than expecting an
identical position count. Low utilization and the profile show that native
legality, expansion, and Python tree descent now dominate; a larger GPU alone
would be poor value for this workload.

### CUDA learning-cycle evidence

A short real 9x9 cycle used 10 walls, a 64x6 model, four concurrent self-play
games, eight simulations, four optimizer steps, and a four-game balanced arena.
It completed self-play, CUDA training, atomic checkpointing, arena rejection,
and resume into generation 2. Generation 1 changed the candidate model hash
from `c855a557...` to `ad71e35e...`, with loss `4.9167 -> 4.8914`; the arena
score was 0.25, so the candidate correctly did not promote. Generation 2
restored generation number and champion hash, reproduced the deterministic
training result, and wrote a distinct checkpoint.

Artifacts are stored locally under `runs/runpod-20260813-rtx3090/`, including
the JSON benchmark, cProfile data, generation logs, and both checkpoints. The
RunPod account was verified at zero Pods afterward; the temporary lifecycle API
key and watchdog credential were removed. Observed account balance changed by
about $0.18 for the complete session.
