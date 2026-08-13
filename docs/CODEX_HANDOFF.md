# Codex handoff: Barricade Zero

This document is the continuation brief for a local Codex agent. Start from the
repository root and read this file, `README.md`, and `docs/performance.md` before
changing code.

## Current status

- Branch: `master`
- Handoff baseline commit before this document: `f2bf187`
- Python: 3.11+
- Tests: standard-library `unittest`; do not assume `pytest` is installed
- Latest committed training suite: **76 tests: 73 passed, 3 CUDA-only skipped
  locally**; all 5 focused evaluator, checkpoint, and generation tests passed
  on an RTX 3090
- Latest random stress: **1,000 games passed**, wins `[499, 501]`, 51,575 plies
- Python reference rules remain the correctness oracle and fallback
- Native C++ backend supports board sizes through 9x9 because its wall/edge
  representation uses 64-bit bitsets
- CUDA evaluation, mixed-precision benchmarking, optimizer steps, checkpoint,
  resume, balanced arena, and promotion rejection are validated on an RTX 3090
- No RunPod Pod remains. The successful GPU test Pod was terminated, so
  there is no ongoing GPU or storage charge.

## Objective

Continue toward a reliable AlphaZero-style 9x9 Barricade/Quoridor trainer:

1. Preserve immutable, configurable, strictly tested game rules.
2. Preserve fixed action encoding, canonical side-to-move orientation, action
   masks, and policy rotation.
3. Keep the residual policy-value network, batched PUCT, concurrent self-play,
   subtree reuse, replay, checkpoint/resume, arena, and promotion path working.
4. Use Python as the correctness oracle and automatic fallback.
5. Differential-test every native rules/encoding change on random reachable
   5x5 and 9x9 states.
6. Move the production-sized model to CUDA and measure complete generation
   throughput, not model forward in isolation.
7. Do not buy or rent compute without explicit user approval.

## Non-negotiable correctness contracts

### State and rules

- `GameState` is immutable.
- Supported boards are configurable odd sizes. Official target is 9x9 with 10
  walls per player, but the exact target game's rule text still needs final
  confirmation before long production training.
- Walls must reject boundary errors, overlap, crossing, and any placement that
  removes either player's last path to goal.
- Pawn movement includes ordinary steps, straight jumps, and diagonal
  go-arounds when the straight jump is blocked or unavailable.
- Repeated self-play positions are draws to prevent weak-network cycles.

### Action space

```text
8 pawn actions + 2 * (N - 1)^2 wall actions
```

- 5x5: 40 actions
- 9x9: 136 actions
- `rotate_action_180` must be an involution.
- A policy rotated twice must equal the original policy.

### Canonical perspective

The state is encoded from the current player's perspective. The current player
moves from the bottom toward the top. On turn change, rotate the board 180
degrees and exchange player information. State planes, actions, policies, and
value signs must remain consistent with this convention.

### Native boundary

Native and Python backends share these contracts:

- `legal_actions(state)`
- `has_path_with_extra_wall(...)`
- `encode_state(state)`
- `encode_batch(states) -> C-contiguous float32[N, 8, S, S]`
- `legal_actions_batch(states) -> int32 CSR offsets/actions`

Important: convert NumPy fixed-width action scalars to built-in Python `int`
before storing them in a tree or `GameState`. Allowing `np.int32` wall
coordinates into Python bitset operations can overflow.

## Architecture map

- `barricade/actions.py`: fixed action encoding and 180-degree rotation
- `barricade/state.py`: immutable Python reference rules and serialization
- `barricade/backend.py`: Python/native backend boundary and `ctypes` loader
- `native/rules.cpp`: native rules, wall path checking, CSR legal actions, and
  batch 8-plane encoding
- `barricade/encoding.py`: Python encoding oracle, legal masks, policy rotation
- `barricade/mcts.py`: simple reference PUCT
- `barricade/batched_mcts.py`: optimized wave-batched PUCT and compact edges
- `barricade/concurrent_self_play.py`: concurrent games and subtree reuse
- `barricade/arena.py`: deterministic balanced candidate/champion matches
- `neural/model.py`: residual policy-value network
- `neural/evaluator.py`: native batch encoding, CPU zero-copy tensor path, and
  contiguous array output
- `training/learner.py`: replay, soft-target policy CE, value MSE, AdamW,
  gradient clipping
- `training/checkpoint.py`: atomic checkpoint save/restore
- `training/generations.py`: self-play, train, arena, promotion, resume
- `scripts/profile_self_play.py`: reproducible profile harness; accepts explicit
  `--channels` and `--blocks`

## Verified learning behavior

The optimizer, checkpoint, resume, and arena paths have all executed against
real self-play samples.

Examples from smoke validation:

- Two-generation forced-continuity test: loss `3.625 -> 3.101`, then
  `3.162 -> 2.845`; replay grew `159 -> 291`.
- Latest vector-prior regression: loss `3.359 -> 3.264`; checkpoint and arena
  completed; candidate did not promote.

Interpretation: parameters learn the generated policy/value targets and updates
can continue across generations. This does **not** yet prove sustained playing
strength. Strength requires larger deterministic arena samples and fixed
baselines.

## Performance baseline

### Search/rules smoke network

Fixed workload:

```bash
PYTHONPATH=. python scripts/profile_self_play.py \
  --board-size 9 --walls 10 --games 128 --simulations 8 \
  --channels 16 --blocks 2 \
  --profile profiles/9x9-vector-priors-128.prof --top 18
```

Latest result:

- 13.40 seconds
- 34,515 evaluated positions
- 2,575.57 positions/s
- average inference batch approximately 86.1

### Candidate production network

```bash
PYTHONPATH=. python scripts/profile_self_play.py \
  --board-size 9 --walls 10 --games 128 --simulations 8 \
  --channels 64 --blocks 6 \
  --profile profiles/9x9-64x6-128.prof --top 18
```

Original CPU baseline:

- 30.24 seconds
- 39,934 evaluated positions
- 1,320.76 positions/s
- average inference batch 90.55
- model forward 17.36 seconds, about 57% of wall time
- expansion about 4.04 seconds

Post-CUDA-path CPU control in a recreated Python 3.14.4 / PyTorch 2.13.0
environment:

- 16.99 seconds
- 39,934 evaluated positions
- 2,350.46 positions/s
- average inference batch 90.55
- synchronized model forward 13.55 seconds

This verifies no local CPU regression but is not an apples-to-apples speedup
claim against the original environment.

This is why CUDA is now the correct next step. The earlier 16x2 profile is for
finding rules/search overhead and must not be used to decide production GPU
needs.

## Optimizations already completed

Do not redo these without a profile proving regression:

- Python bitset blocked-edge representation
- candidate-wall BFS without temporary states
- native rules and wall path checks
- fixed-capacity native BFS storage
- precomputed open-edge adjacency for candidate walls
- native 8-plane encoding and continuous float32 output
- one-call native batch encoding ABI
- native CSR legal-action generation
- cached action/policy rotation permutations
- persistent subtree reuse
- compact parallel-list MCTS edges
- contiguous evaluator policy/value arrays
- vectorized legal-prior indexing and normalization

See `docs/performance.md` for the complete progression.

## RunPod attempt and lessons

The GPU code path is validated on CUDA. A successful 2026-08-13 run used a
Secure Cloud RTX 3090 at $0.50/hour with:

```text
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

Full public-IP SSH worked with the registered local `runpod_autorae` key. The
64x6 benchmark completed in 13.07 seconds at 3,030.12 positions/s; model
forward took 2.57 seconds, GPU utilization averaged only 3.61%, and peak
allocated VRAM was about 24.4 MB. A two-generation 9x9 CUDA cycle also passed
self-play, optimizer, checkpoint, resume, arena, and no-promotion paths. Local
artifacts are under `runs/runpod-20260813-rtx3090/`.

The Pod was terminated, the account was verified at zero Pods, and the
temporary restricted lifecycle key was revoked. The balance changed from
$19.40 to $19.22 during the complete session.

Historical failed attempt:

The attempted RunPod setup used an RTX 3090 at $0.22/hour. The first image was:

```text
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

It failed because the Community host driver supported CUDA 12.5 while the image
required CUDA 12.8. A CUDA 12.4 image was then selected:

```text
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

That resolved the CUDA compatibility issue, but RunPod basic SSH continued to
reject the temporary key despite `runpodctl ssh list-keys` showing the matching
fingerprint. The successful run avoided basic SSH by using a Secure Cloud Pod
with public-IP SSH.

If revisiting RunPod:

1. Use the registered local `runpod_autorae` key explicitly for SSH.
2. Select an image whose CUDA requirement is no newer than the host driver.
3. Prefer a public-IP Pod with full SSH if SCP/rsync is required.
4. Install a hard budget watchdog before transferring code.
5. Start with RTX 3090/4090. Do not jump to A100/H100 until complete-generation
   cost efficiency is measured.

## Recommended next work, in order

### 1. Confirm exact Barricade rules

Before expensive 9x9 training, obtain and encode the exact target rules. Check:

- board size and wall inventory
- jump and diagonal rules
- wall anchor convention
- repetition/draw handling
- whether the target differs from standard Quoridor

Update tests before changing engine behavior. The short CUDA cycle produced all
self-play draws, reinforcing that long paid training should wait for rule and
evaluation confirmation.

### 2. Improve evaluation quality

The current tiny arena is only a plumbing smoke test. Add:

- fixed deterministic shortest-path and reference-MCTS baselines
- larger confidence-aware candidate/champion arena
- explicit draw handling and score intervals
- fixed evaluation seeds and reproducible reports

Loss decrease alone is not evidence of stronger play.

### 3. Scale training where profiles justify it

The RTX 3090 averaged only 3.61% utilization. Before renting a larger GPU:

- increase concurrent self-play games and measure batch/utilization response
- profile native legality, expansion, and Python tree descent on the GPU host
- persist generation duration, training duration, and arena duration separately
- compare cost per generated example and cost per promoted checkpoint

Do not interpret low VRAM use as a reason to enlarge the network until stronger
evaluation and exact-rule gates are satisfied.

## Development discipline

Use strict RED-GREEN-REFACTOR:

1. Add one failing test.
2. Run the specific test and confirm expected failure.
3. Add minimal implementation.
4. Run focused tests.
5. Run the full suite and stress test.
6. Profile before and after performance changes.
7. Update docs and commit a logically isolated change.

Do not remove Python fallback or weaken differential tests for speed.

## Required verification commands

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
PYTHONPATH=. python3 scripts/stress_random_games.py --games 1000
git diff --check
git status --short
```

Build and directly test native backend changes with:

```bash
source .venv/bin/activate
python scripts/build_native.py
python -m unittest tests.test_backend -v
```

## Suggested first Codex prompt

```text
Read docs/CODEX_HANDOFF.md, README.md, and docs/performance.md. Continue
Barricade Zero from the current branch. Preserve the verified CUDA path and
local RunPod artifacts. Confirm the exact target-game rules and add fixed
evaluation baselines before long paid training. Then scale concurrent self-play
while measuring complete-generation throughput and cost, not isolated model
forward speed. Do not rent or purchase compute without explicit approval.
```
