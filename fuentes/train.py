"""Entrena y compara Logistic Regression y Random Forest con TF-IDF y CV estratificada."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

try:
    from fuentes.datos_prep import EXPECTED_LABELS, load_dataset, sha256_file
except ImportError:  # Permite ejecutar el archivo directamente desde fuentes/.
    from datos_prep import EXPECTED_LABELS, load_dataset, sha256_file

SEED = 42
TEST_SIZE = 0.20
CV_SPLITS = 5


def build_search(
    model_name: str,
    cv: Any | None = None,
    *,
    quick: bool = False,
) -> GridSearchCV:
    """Construye un pipeline TF-IDF y una cuadrícula acotada por modelo."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        max_features=30000 if model_name == "logistic_regression" else 20000,
        min_df=2,
    )
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED, solver="lbfgs"
        )
        grid = {
            "tfidf__ngram_range": [(1, 1)] if quick else [(1, 1), (1, 2)],
            "classifier__C": [1.0] if quick else [1.0, 3.0],
        }
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=1,
        )
        grid = {
            "tfidf__max_features": [200] if quick else [10000, 20000],
            "classifier__n_estimators": [5] if quick else [60],
            "classifier__max_depth": [8] if quick else [30],
            "classifier__max_features": ["sqrt"] if quick else ["sqrt", "log2"],
            "classifier__min_samples_leaf": [2] if quick else [3],
        }
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
    splitter = cv or StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="f1_macro",
        cv=splitter,
        refit=True,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )


def evaluate_predictions(
    y_true: list[str], y_pred: list[str], labels: list[str], probabilities: np.ndarray | None = None
) -> dict[str, Any]:
    """Calcula métricas de clasificación y ROC-AUC OVR cuando es válido."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": None,
        "roc_auc_note": "No se proporcionaron probabilidades para las tres clases.",
    }
    if probabilities is not None and probabilities.shape == (len(y_true), len(labels)):
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="macro")
            )
            metrics["roc_auc_note"] = None
        except ValueError as exc:
            metrics["roc_auc_note"] = f"No calculable: {exc}"
    return metrics


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _plot_results(comparison: list[dict[str, Any]], confusions: dict[str, np.ndarray], labels: list[str], evidence_dir: Path) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    names = [row["model"] for row in comparison]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, row in enumerate(comparison):
        values = [row[name] for name in metrics]
        ax.bar(x + (i - (len(comparison) - 1) / 2) * width, values, width, label=names[i])
    ax.set_xticks(x, metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Puntaje")
    ax.set_title("Métricas sobre el conjunto de prueba")
    ax.legend()
    fig.tight_layout()
    fig.savefig(evidence_dir / "comparacion_modelos.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(confusions), figsize=(6 * len(confusions), 5), squeeze=False)
    for ax, (name, matrix) in zip(axes[0], confusions.items()):
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta real")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Matrices de confusión (prueba)")
    fig.tight_layout()
    fig.savefig(evidence_dir / "matrices_confusion.png", dpi=160)
    plt.close(fig)


def run_experiments(data_path: Path, output_dir: Path, evidence_dir: Path) -> dict[str, Any]:
    """Ejecuta split, GridSearchCV, evaluación de prueba y guarda resultados agregados."""
    rows, columns = load_dataset(data_path)
    if columns[:2] != ["text", "airline_sentiment"] and not all(
        column in columns for column in ("text", "airline_sentiment")
    ):
        raise ValueError("El CSV debe incluir text y airline_sentiment.")
    texts = [row["text"] for row in rows]
    labels = [row["airline_sentiment"] for row in rows]
    if any(not value.strip() for value in texts + labels):
        raise ValueError("El dataset contiene textos o etiquetas vacíos.")
    observed = set(labels)
    if observed != set(EXPECTED_LABELS):
        raise ValueError(f"Se esperaban las clases {EXPECTED_LABELS}; se encontraron {sorted(observed)}")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
    )
    splitter = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    model_names = ("logistic_regression", "random_forest")
    comparison: list[dict[str, Any]] = []
    experiments: dict[str, Any] = {}
    confusions: dict[str, np.ndarray] = {}
    cv_rows: list[dict[str, Any]] = []

    for model_name in model_names:
        search = build_search(model_name, splitter)
        start = time.perf_counter()
        search.fit(x_train, y_train)
        elapsed = time.perf_counter() - start
        best_index = int(search.best_index_)
        fold_scores = [float(search.cv_results_[f"split{i}_test_score"][best_index]) for i in range(CV_SPLITS)]
        best_model = search.best_estimator_
        predictions = best_model.predict(x_test)
        probabilities = best_model.predict_proba(x_test) if hasattr(best_model, "predict_proba") else None
        metrics = evaluate_predictions(y_test, predictions, list(EXPECTED_LABELS), probabilities)
        matrix = confusion_matrix(y_test, predictions, labels=list(EXPECTED_LABELS))
        confusions[model_name] = matrix
        class_report = classification_report(
            y_test, predictions, labels=list(EXPECTED_LABELS), target_names=list(EXPECTED_LABELS),
            output_dict=True, zero_division=0,
        )
        cv_summary = {
            "fold_f1_macro": fold_scores,
            "mean_f1_macro": float(np.mean(fold_scores)),
            "std_f1_macro": float(np.std(fold_scores, ddof=1)),
        }
        experiments[model_name] = {
            "best_params": search.best_params_,
            "best_cv_f1_macro": float(search.best_score_),
            "cv_selected_candidate": cv_summary,
            "test_metrics": metrics,
            "classification_report": class_report,
            "confusion_matrix": matrix.tolist(),
            "fit_duration_seconds": round(elapsed, 3),
            "grid_candidate_count": len(search.cv_results_["params"]),
        }
        comparison.append({"model": model_name, **metrics, "cv_f1_macro_mean": cv_summary["mean_f1_macro"],
                           "cv_f1_macro_std": cv_summary["std_f1_macro"], "fit_duration_seconds": round(elapsed, 3)})
        for candidate_index, params in enumerate(search.cv_results_["params"]):
            row = {"model": model_name, "candidate_index": candidate_index,
                   "mean_test_f1_macro": float(search.cv_results_["mean_test_score"][candidate_index]),
                   "std_test_f1_macro": float(search.cv_results_["std_test_score"][candidate_index]),
                   "rank_test_f1_macro": int(search.cv_results_["rank_test_score"][candidate_index]),
                   "selected": candidate_index == best_index}
            row.update({key: str(value) for key, value in params.items()})
            for fold in range(CV_SPLITS):
                row[f"split{fold}_test_f1_macro"] = float(search.cv_results_[f"split{fold}_test_score"][candidate_index])
            cv_rows.append(row)

        _write_json(output_dir / f"classification_report_{model_name}.json", class_report)
        with (output_dir / f"confusion_matrix_{model_name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["actual\\predicted", *EXPECTED_LABELS])
            writer.writerows([[label, *matrix[index].tolist()] for index, label in enumerate(EXPECTED_LABELS)])

    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    report = {
        "generated_at": timestamp,
        "data": {"path": str(data_path), "sha256": sha256_file(data_path), "rows": len(rows), "columns": columns},
        "protocol": {
            "random_seed": SEED, "test_size": TEST_SIZE, "train_size": len(x_train), "test_rows": len(x_test),
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "selection_metric": "f1_macro", "tfidf_fitted_within_pipeline": True,
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "duplicate_text_train_test_overlap": len(set(x_train).intersection(x_test)),
            "class_counts_train": {label: y_train.count(label) for label in EXPECTED_LABELS},
            "class_counts_test": {label: y_test.count(label) for label in EXPECTED_LABELS},
            "cv_note": "Los puntajes por fold corresponden al candidato ganador elegido por la misma búsqueda; son descriptivos, no una prueba de significancia.",
        },
        "dependency_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": importlib.metadata.version("seaborn"),
            "mlflow": None,
        },
        "models": experiments,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metricas_experimentos.json", report)
    with (output_dir / "comparacion_modelos.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    metric_columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_ovr_macro"]
    table = ["| Modelo | " + " | ".join(metric_columns) + " |", "|" + "---|" * (len(metric_columns) + 1)]
    for row in comparison:
        table.append("| " + row["model"] + " | " + " | ".join(f"{row[name]:.4f}" for name in metric_columns) + " |")
    (output_dir / "comparacion_modelos.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    with (output_dir / "resultados_cv.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(dict.fromkeys(key for row in cv_rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cv_rows)
    _plot_results(comparison, confusions, list(EXPECTED_LABELS), evidence_dir)
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "datos" / "datos_limp" / "Tweets_limpios.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "reportes" / "entrenamiento")
    parser.add_argument("--evidence-dir", type=Path, default=root / "evidencias")
    args = parser.parse_args()
    print(json.dumps(run_experiments(args.data, args.output_dir, args.evidence_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
