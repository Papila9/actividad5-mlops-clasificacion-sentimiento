import json
import os
import tempfile
import unittest
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from fuentes.mlflow_tracking import _log_run


class MlflowTrackingTests(unittest.TestCase):
    def test_run_persists_params_metrics_tags_and_artifacts(self):
        previous_uri = mlflow.get_tracking_uri()
        previous_file_store = os.environ.get("MLFLOW_ALLOW_FILE_STORE")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            mlflow.set_tracking_uri((root / "tracking").as_uri())
            mlflow.set_experiment("integration-test")
            artifact_dir = root / "artifacts"
            artifact_dir.mkdir()
            (artifact_dir / "classification_report.json").write_text(
                json.dumps({"negative": {"f1-score": 0.8}}), encoding="utf-8"
            )
            result = {
                "best_params": {"classifier__C": 3.0},
                "fit_duration_seconds": 1.25,
                "test_metrics": {
                    "accuracy": 0.8, "precision_macro": 0.79, "recall_macro": 0.78,
                    "f1_macro": 0.785, "f1_weighted": 0.8, "roc_auc_ovr_macro": 0.9,
                    "roc_auc_note": None,
                },
                "cv_selected_candidate": {
                    "fold_f1_macro": [0.7, 0.8], "mean_f1_macro": 0.75, "std_f1_macro": 0.0707,
                },
            }
            run_id = _log_run(
                "evaluacion_original", "logistic_regression", result,
                {"train_size": 12, "test_rows": 3}, "a" * 64, [artifact_dir],
            )
            client = MlflowClient()
            run = client.get_run(run_id)
            self.assertEqual(run.data.tags["evaluation_type"], "evaluacion_original")
            self.assertEqual(run.data.tags["test_set_used_for_tuning"], "false")
            self.assertEqual(run.data.params["best_classifier__C"], "3.0")
            self.assertEqual(run.data.metrics["f1_macro"], 0.785)
            artifacts = client.list_artifacts(run_id)
            self.assertIn("classification_report.json", [item.path for item in artifacts])
            mlflow.delete_experiment(run.info.experiment_id)
            mlflow.set_tracking_uri(previous_uri)
            if previous_file_store is None:
                os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)
            else:
                os.environ["MLFLOW_ALLOW_FILE_STORE"] = previous_file_store


if __name__ == "__main__":
    unittest.main()
