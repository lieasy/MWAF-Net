"""Prevent the mean-of-class-SDs aggregation mistake using opposing recalls."""

import unittest
import numpy as np
from scripts.summarize_runs import summarize_runs


class SummaryTests(unittest.TestCase):
    def records(self):
        return [dict(task="clean_binary", class_names=["Normal", "Abnormal"],
                     prediction_unit="segment", positive_class_index=1,
                     confusion_counts=[[k, 10-k], [k, 10-k]]) for k in [1, 3, 5, 7, 9]]

    def test_uar_sd_is_computed_across_runs(self):
        result = summarize_runs(self.records())
        self.assertEqual(result["metrics"]["uar_percent"]["mean"], 50.0)
        self.assertEqual(result["metrics"]["uar_percent"]["sample_sd"], 0.0)
        self.assertGreater(result["metrics"]["class_recall_percent"]["sample_sd"][0], 0.0)
        expected = np.std([10, 30, 50, 70, 90], ddof=1)
        self.assertAlmostEqual(result["metrics"]["class_recall_percent"]["sample_sd"][0], expected)

    def test_inconsistent_run_is_rejected(self):
        rows = self.records()
        rows[0]["f1_percent"] = 99.0
        with self.assertRaisesRegex(ValueError, "f1_percent disagrees"):
            summarize_runs(rows)


if __name__ == "__main__":
    unittest.main()
