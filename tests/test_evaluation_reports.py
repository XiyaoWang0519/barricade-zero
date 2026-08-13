import unittest

from barricade.evaluation.reports import build_scoreboard


def _report(generation, score, suite="suite-a", simulations=64):
    return {
        "format_version": 1,
        "candidate": {
            "generation": generation,
            "path": f"generation_{generation:03d}.pt",
            "model_sha256": f"model-{generation}",
        },
        "opening_suite": {"sha256": suite},
        "settings": {
            "simulations": simulations,
            "max_plies": 500,
            "confidence": 0.95,
            "promotion_threshold": 0.5,
        },
        "matches": [
            {
                "opponent": "champion",
                "score": score,
                "wins": 120,
                "draws": 80,
                "losses": 200,
                "interval": {"lower": score - 0.04, "upper": score + 0.04},
                "decision": {"status": "inconclusive"},
            }
        ],
        "promotion": {"status": "inconclusive"},
    }


class EvaluationReportTests(unittest.TestCase):
    def test_scoreboard_sorts_generations_and_keeps_confidence_bounds(self):
        scoreboard = build_scoreboard(
            [("g2.json", _report(2, 0.55)), ("g1.json", _report(1, 0.51))]
        )
        self.assertTrue(scoreboard["comparable"])
        self.assertEqual(
            [row["generation"] for row in scoreboard["generations"]],
            [1, 2],
        )
        self.assertAlmostEqual(
            scoreboard["generations"][1]["matches"]["champion"]["lower"],
            0.51,
        )

    def test_scoreboard_warns_when_protocols_differ(self):
        scoreboard = build_scoreboard(
            [("g1.json", _report(1, 0.5)), ("g2.json", _report(2, 0.5, simulations=32))]
        )
        self.assertFalse(scoreboard["comparable"])
        self.assertEqual(len(scoreboard["warnings"]), 1)


if __name__ == "__main__":
    unittest.main()
