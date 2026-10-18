"""Reproduce y registra en MLflow los dos protocolos de evaluacion de Fase 3."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient

try:
    from fuentes.auditoria_fase3 import grouped_evaluation
    from fuentes.datos_prep import load_dataset, sha256_file
    from fuentes.train import CV_SPLITS, SEED, run_experiments
except ImportError:  # Permite ejecutar tambien `python fuentes/mlflow_tracking.py`.
    from auditoria_fase3 import grouped_evaluation
    from datos_prep import load_dataset, sha256_file
    from train import CV_SPLITS, SEED, run_experiments

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datos" / "datos_limp" / "Tweets_limpios.csv"
EXPERIMENT_NAME = "Actividad5_Clasificacion_Sentimiento"
EVALUATIONS = {
    "evaluacion_original": {
        "split": "train_test_split estratificado 80/20, random_state=42",
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "report_name": "metricas_experimentos.json",
    },
    "evaluacion_agrupada": {
        "split": "Primer fold de StratifiedGroupKFold 5-fold agrupado por texto exacto",
        "cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
        "report_name": "evaluacion_agrupada.json",
    },
}
ARCHIVED_RESULTS = {
    "evaluacion_original": ROOT / "reportes" / "entrenamiento" / "metricas_experimentos.json",
    "evaluacion_agrupada": ROOT / "reportes" / "auditoria_fase3" / "evaluacion_agrupada" / "evaluacion_agrupada.json",
}


def _safe_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _verify_reproduction(scenario: str, reproduced: dict[str, Any]) -> None:
    """Exige concordancia con los resultados previamente validados antes de registrar."""
    archived_path = ARCHIVED_RESULTS[scenario]
    if not archived_path.is_file():
        raise FileNotFoundError(f"Falta el resultado validado requerido: {archived_path}")
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    for name, result in reproduced["models"].items():
        previous = archived["models"][name]
        current_params = json.loads(json.dumps(result["best_params"]))
        if current_params != previous["best_params"]:
            raise AssertionError(f"Los hiperparámetros reproducidos de {scenario}/{name} no coinciden con el archivo validado.")
        old_metrics = previous["test_metrics"]
        for metric, value in result["test_metrics"].items():
            if isinstance(value, (int, float)) and not math.isclose(
                float(value), float(old_metrics[metric]), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise AssertionError(f"La métrica {scenario}/{name}/{metric} difiere del resultado validado.")


def _log_run(
    scenario: str,
    model_name: str,
    model_result: dict[str, Any],
    protocol: dict[str, Any],
    data_hash: str,
    artifact_dirs: list[Path],
) -> str:
    """Registra un modelo y un protocolo en un run aislado."""
    with mlflow.start_run(run_name=f"{scenario}__{model_name}") as run:
        mlflow.set_tags({
            "evaluation_type": scenario,
            "model": model_name,
            "test_set_used_for_tuning": "false",
            "tfidf_fitted_inside_training_pipeline": "true",
            "test_metrics_source": "held_out_test_partition",
        })
        params = {
            "model": model_name,
            "random_seed": SEED,
            "split_strategy": EVALUATIONS[scenario]["split"],
            "cross_validation": EVALUATIONS[scenario]["cv"],
            "selection_metric": "f1_macro",
            "grid_candidate_count": model_result.get("grid_candidate_count", model_result.get("candidate_count")),
            "python_version": platform.python_version(),
            "mlflow_version": mlflow.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "matplotlib_version": importlib.metadata.version("matplotlib"),
            "dataset_sha256": data_hash,
            "train_rows": protocol["train_rows"] if "train_rows" in protocol else protocol["train_size"],
            "test_rows": protocol["test_rows"],
            "tfidf_in_pipeline": True,
        }
        params.update({f"best_{key}": _safe_value(value) for key, value in model_result["best_params"].items()})
        mlflow.log_params({key: _safe_value(value) for key, value in params.items()})

        metrics = dict(model_result["test_metrics"])
        metrics.pop("roc_auc_note", None)
        cv_folds = model_result.get("cv_selected_candidate", {}).get(
            "fold_f1_macro", model_result.get("cv_fold_f1_macro", [])
        )
        cv_mean = model_result.get("cv_selected_candidate", {}).get(
            "mean_f1_macro", model_result.get("cv_mean_f1_macro")
        )
        cv_std = model_result.get("cv_selected_candidate", {}).get(
            "std_f1_macro", model_result.get("cv_std_f1_macro_sample")
        )
        metrics.update({
            "cv_f1_macro_mean": float(cv_mean),
            "cv_f1_macro_std": float(cv_std),
            "grid_search_fit_duration_seconds": float(model_result["fit_duration_seconds"]),
        })
        metrics.update({f"cv_f1_macro_fold_{index}": float(score) for index, score in enumerate(cv_folds, 1)})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items() if value is not None})
        for artifact_dir in artifact_dirs:
            mlflow.log_artifacts(str(artifact_dir))
        return run.info.run_id


def register_experiments(data_path: Path = DEFAULT_DATA, tracking_db: Path = ROOT / "mlflow.db") -> list[dict[str, Any]]:
    """Reejecuta ambos protocolos y registra cuatro runs sin tocar salidas previas."""
    data_path = data_path.resolve()
    tracking_db = tracking_db.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_db.as_uri().replace("file:///", "sqlite:///"))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        artifact_uri = (ROOT / "mlartifacts").resolve().as_uri()
        experiment_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_uri)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)

    rows, _ = load_dataset(data_path)
    data_hash = sha256_file(data_path)
    registered: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="actividad5_mlflow_") as temp:
        temp_root = Path(temp)
        # Reproduce the protocolo original a traves de la funcion ya auditada.
        original_dir = temp_root / "evaluacion_original"
        original_evidence = temp_root / "evidencia_original"
        original = run_experiments(data_path, original_dir, original_evidence)
        _verify_reproduction("evaluacion_original", original)
        original_protocol = original["protocol"]
        original_artifacts = [original_dir, original_evidence]
        for model_name, model_result in original["models"].items():
            run_id = _log_run("evaluacion_original", model_name, model_result,
                              original_protocol, data_hash, original_artifacts)
            registered.append({"scenario": "evaluacion_original", "model": model_name, "run_id": run_id,
                               "metrics": model_result["test_metrics"]})

        # Reproduce el escenario agrupado con StratifiedGroupKFold de la auditoria.
        grouped_dir = temp_root / "evaluacion_agrupada"
        grouped_evidence = temp_root / "evidencia_agrupada"
        grouped = grouped_evaluation(rows, grouped_dir, grouped_evidence)
        _verify_reproduction("evaluacion_agrupada", grouped)
        if grouped["protocol"]["train_test_text_overlap"] != 0:
            raise AssertionError("La evaluacion agrupada contiene solapamiento de texto exacto.")
        for model_name, model_result in grouped["models"].items():
            run_id = _log_run("evaluacion_agrupada", model_name, model_result,
                              grouped["protocol"], data_hash, [grouped_dir, grouped_evidence])
            registered.append({"scenario": "evaluacion_agrupada", "model": model_name, "run_id": run_id,
                               "metrics": model_result["test_metrics"]})
    return registered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tracking-db", type=Path, default=ROOT / "mlflow.db")
    args = parser.parse_args()
    print(json.dumps(register_experiments(args.data, args.tracking_db), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
