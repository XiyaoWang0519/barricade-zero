# Model evaluation

The evaluation system is deliberately independent of `training/`. It reads
checkpoints, fixed opening suites, and optional held-out labeled positions; it
never changes a checkpoint or decides what the training loop should save.

It answers two separate questions:

1. **Is the network learning its targets?** Use the `predictions` command on a
   frozen dataset that is never sampled for optimizer updates.
2. **Is the game-playing agent stronger?** Use paired matches against a frozen
   champion, permanent baselines, and historical checkpoints.

## Fixed opening suite

Generate the suite once after the target game rules are confirmed. Keep the
result and its SHA-256 stable across generations:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_checkpoints.py \
  generate-openings \
  --output benchmarks/openings-9x9-v1.json \
  --count 200 --board-size 9 --walls 10 \
  --min-plies 2 --max-plies 16 --seed 20260813
```

Openings are random legal trajectories generated from a recorded seed. The
file includes the complete states, not just the seed, so a report remains
reproducible if random-number implementation details later change. Duplicate
canonical states and terminal states are rejected.

## Checkpoint matches

Evaluate a candidate against a champion and the permanent baselines:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_checkpoints.py run \
  --candidate checkpoints/generations/generation_027.pt \
  --champion checkpoints/champions/champion_026.pt \
  --openings benchmarks/openings-9x9-v1.json \
  --baseline random \
  --baseline shortest-path \
  --baseline uniform-mcts \
  --baseline untrained \
  --simulations 64 --minimum-pairs 100 \
  --output reports/evaluation/generation_027.json \
  --device cuda --mixed-precision
```

Additional frozen checkpoint opponents can be repeated:

```bash
--opponent-checkpoint checkpoints/anchors/generation_000.pt \
--opponent-checkpoint checkpoints/anchors/generation_010.pt
```

Every opening is played twice, with the candidate controlling player 0 once
and player 1 once. A win is worth 1, a draw 0.5, and a loss 0. Confidence
intervals are paired bootstraps over opening pairs, so side advantage does not
artificially inflate the sample size.

The champion decision is one of:

- `promote`: the lower confidence bound exceeds the configured threshold.
- `reject`: the upper confidence bound is below the threshold.
- `inconclusive`: the interval crosses the threshold.
- `insufficient_data`: fewer than the required opening pairs were played.
- `identical_model`: candidate and champion checkpoint files contain the same
  network weights, even if their checkpoint metadata differs.

The CLI only reports this decision. It does not promote, overwrite, or delete
checkpoints. The default promotion threshold is 0.5 with 95% confidence and a
minimum of 100 opening pairs.

Games also record whether they ended by a goal, threefold repetition, or the
maximum-ply limit. A score dominated by maximum-ply draws should be treated as
an evaluation failure, not evidence that the models are equally strong.

## Held-out policy and value metrics

The training pipeline should export a frozen JSON or JSONL dataset once and
exclude those positions from replay sampling. Each row has this shape:

```json
{
  "id": "validation-000001",
  "state": {
    "size": 9,
    "pawns": [[8, 4], [0, 4]],
    "horizontal_walls": [],
    "vertical_walls": [],
    "walls_remaining": [10, 10],
    "turn": 0,
    "winner": null
  },
  "policy": [0.0, 0.0, 1.0],
  "value": 1.0
}
```

The policy example above is abbreviated; the real vector must have the full
action-space length and sum to one. Policy coordinates must match the supplied
state. Values use `-1`, `0`, and `1` from the state's current-player
perspective.

Run the metrics with:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_checkpoints.py predictions \
  --candidate checkpoints/generations/generation_027.pt \
  --dataset benchmarks/heldout-9x9-v1.jsonl \
  --output reports/evaluation/generation_027-predictions.json \
  --device cuda --mixed-precision
```

The report includes policy cross-entropy, policy top-1 and top-3 agreement,
value MSE and MAE, value calibration error, and raw probability mass assigned
to illegal actions.

## Longitudinal scoreboard

Combine generation reports into a compact machine-readable history:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_checkpoints.py scoreboard \
  --report reports/evaluation/generation_025.json \
  --report reports/evaluation/generation_026.json \
  --report reports/evaluation/generation_027.json \
  --output reports/evaluation/scoreboard.json
```

The scoreboard refuses to describe reports as directly comparable when the
opening-suite hash, search simulations, maximum plies, confidence level, or
promotion threshold differs. Strength trends should only be interpreted when
`comparable` is true.

## Integration contract for training

The training pipeline only needs to provide:

- candidate and champion checkpoint paths;
- a stable checkpoint network architecture and metadata;
- one frozen held-out dataset that is never trained on;
- immutable historical anchor checkpoints.

It may invoke the CLI after saving a candidate, but promotion should consume
the report's top-level `promotion.status` rather than recomputing a raw win-rate
threshold. Do not freeze the production 9x9 opening suite until the exact
target rules are confirmed.
