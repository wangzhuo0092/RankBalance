import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from elo_processor import EloProcessor  # noqa: E402


class RankBalanceTest(unittest.TestCase):
    def test_gamma_must_be_supplied(self):
        frame = pd.DataFrame([
            {"methodA": "a", "methodB": "b", "answerValue": "A", "answerer": "r", "isGolden": "false"}
        ])
        processor = EloProcessor(frame, ["r"])
        with self.assertRaisesRegex(ValueError, "gamma must be supplied"):
            processor.process(df=frame, valid_users=["r"], model="rank_balance")

    def test_joint_fit_learns_ability_and_reliability_orientation(self):
        rows = []
        for _ in range(30):
            rows.append(
                {
                    "methodA": "strong",
                    "methodB": "weak",
                    "answerValue": "A",
                    "answerer": "reliable",
                    "isGolden": "false",
                }
            )
        for _ in range(10):
            rows.append(
                {
                    "methodA": "strong",
                    "methodB": "weak",
                    "answerValue": "B",
                    "answerer": "reversed",
                    "isGolden": "false",
                }
            )

        frame = pd.DataFrame(rows)
        processor = EloProcessor(frame, ["reliable", "reversed"])
        ranking, _ = processor.process(
            df=frame,
            valid_users=["reliable", "reversed"],
            model="rank_balance",
            rank_balance_config={"gamma": 0.1},
        )

        self.assertEqual(ranking.iloc[0]["Method"], "strong")
        qualities = processor.metric.state["qualities"]
        self.assertGreater(
            qualities["reliable"]["value"], qualities["reversed"]["value"]
        )
        centered_elo = sum(
            score["value"] - 2000.0
            for score in processor.metric.state["scores"].values()
        )
        self.assertAlmostEqual(centered_elo, 0.0, places=8)
        self.assertTrue(
            processor.metric.state["rank_balance_optimization"]["success"]
        )


if __name__ == "__main__":
    unittest.main()
