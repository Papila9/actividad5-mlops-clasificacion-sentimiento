import unittest

import numpy as np
from sklearn.model_selection import StratifiedKFold

from fuentes.auditoria_fase3 import inspect_duplicate_splits
from fuentes.train import build_search, evaluate_predictions


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.x = [
            "bad flight delayed rude staff", "awful delay never helpful", "cancelled flight very bad",
            "negative service lost luggage", "terrible seat and delay", "bad airline refund",
            "fine flight no strong opinion", "average service neutral comment", "flight was okay today",
            "nothing special but acceptable", "ordinary travel experience", "neutral response expected",
            "great flight helpful staff", "excellent service and on time", "happy with the airline",
            "wonderful crew very helpful", "good seat and great trip", "love this flight experience",
        ]
        self.y = ["negative"] * 6 + ["neutral"] * 6 + ["positive"] * 6

    def test_pipeline_and_small_grid_search_for_both_models(self):
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        for name in ("logistic_regression", "random_forest"):
            with self.subTest(model=name):
                search = build_search(name, cv=cv, quick=True)
                self.assertEqual(list(search.estimator.named_steps), ["tfidf", "classifier"])
                search.fit(self.x, self.y)
                self.assertGreaterEqual(search.best_score_, 0.0)
                self.assertEqual(len(search.cv_results_["split0_test_score"]), 1)

    def test_metrics_and_multiclass_roc_auc(self):
        y_true = ["negative", "neutral", "positive"] * 2
        y_pred = ["negative", "neutral", "positive"] * 2
        probabilities = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]] * 2)
        result = evaluate_predictions(y_true, y_pred, ["negative", "neutral", "positive"], probabilities)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1_macro"], 1.0)
        self.assertAlmostEqual(result["roc_auc_ovr_macro"], 1.0)

    def test_audit_counts_repeated_and_conflicting_text_groups(self):
        rows = [
            {"text": f"{label} unique {index}", "airline_sentiment": label}
            for label in ("negative", "neutral", "positive")
            for index in range(15)
        ]
        rows.extend([
            {"text": "repeated same label", "airline_sentiment": "negative"},
            {"text": "repeated same label", "airline_sentiment": "negative"},
            {"text": "conflicting label text", "airline_sentiment": "negative"},
            {"text": "conflicting label text", "airline_sentiment": "positive"},
        ])
        result = inspect_duplicate_splits(rows)
        self.assertEqual(result["text_groups_with_conflicting_labels"], 1)
        self.assertEqual(result["duplicate_text_label_repetitions"], 1)
        self.assertEqual(result["repeated_text_groups"], 2)


if __name__ == "__main__":
    unittest.main()
