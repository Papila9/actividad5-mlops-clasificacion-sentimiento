"""Audita trazabilidad de Fase 3 y ejecuta una evaluación agrupada independiente.

Todos los artefactos se escriben bajo reportes/auditoria_fase3 y
evidencias/auditoria_fase3; no reemplaza los resultados originales.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

try:
    from fuentes.train import (
        CV_SPLITS, EXPECTED_LABELS, SEED, TEST_SIZE, _plot_results, _write_json,
        build_search, evaluate_predictions, run_experiments,
    )
    from fuentes.datos_prep import load_dataset, sha256_file
except ImportError:  # Permite también `python fuentes/auditoria_fase3.py`.
    from train import (
        CV_SPLITS, EXPECTED_LABELS, SEED, TEST_SIZE, _plot_results, _write_json,
        build_search, evaluate_predictions, run_experiments,
    )
    from datos_prep import load_dataset, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datos" / "datos_limp" / "Tweets_limpios.csv"
ORIGINAL_DIR = ROOT / "reportes" / "entrenamiento"
AUDIT_DIR = ROOT / "reportes" / "auditoria_fase3"
EVIDENCE_DIR = ROOT / "evidencias" / "auditoria_fase3"


def _class_counts(values: list[str]) -> dict[str, int]:
    counts = Counter(values)
    return {label: counts.get(label, 0) for label in EXPECTED_LABELS}


def inspect_duplicate_splits(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Reconstruye el split original y compara textos/etiquetas sin guardar textos."""
    texts = [row["text"] for row in rows]
    labels = [row["airline_sentiment"] for row in rows]
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
    )
    train_label_map: dict[str, set[str]] = defaultdict(set)
    test_label_map: dict[str, set[str]] = defaultdict(set)
    full_label_map: dict[str, set[str]] = defaultdict(set)
    for text, label in zip(texts, labels):
        full_label_map[text].add(label)
    for text, label in zip(x_train, y_train):
        train_label_map[text].add(label)
    for text, label in zip(x_test, y_test):
        test_label_map[text].add(label)
    shared = set(train_label_map) & set(test_label_map)
    pairs = {(row["text"], row["airline_sentiment"]) for row in rows}
    text_frequency = Counter(texts)
    duplicate_groups = [text for text, count in text_frequency.items() if count > 1]
    return {
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "train_class_counts": _class_counts(y_train),
        "test_class_counts": _class_counts(y_test),
        "shared_unique_texts": len(shared),
        "shared_unique_texts_same_label_only": sum(len(full_label_map[text]) == 1 for text in shared),
        "shared_texts_with_same_label_in_both_splits": sum(
            bool(train_label_map[text] & test_label_map[text]) for text in shared
        ),
        "shared_texts_with_conflicting_labels_across_splits": sum(
            len(train_label_map[text] | test_label_map[text]) > 1 for text in shared
        ),
        "shared_texts_whose_full_dataset_labels_conflict": sum(
            len(full_label_map[text]) > 1 for text in shared
        ),
        "unique_text_groups": len(full_label_map),
        "repeated_text_groups": len(duplicate_groups),
        "text_groups_with_conflicting_labels": sum(len(values) > 1 for values in full_label_map.values()),
        "duplicate_text_label_repetitions": len(rows) - len(pairs),
        "train_test_unique_text_overlap": len(shared),
        "train_test_text_overlap_row_pairs": sum(
            sum(value == text for value in x_train) * sum(value == text for value in x_test)
            for text in shared
        ),
    }


def verify_existing_outputs(existing: dict[str, Any], output_dir: Path) -> dict[str, bool]:
    """Comprueba la consistencia entre JSON, tablas, folds, matrices y reportes guardados."""
    comparison = {
        row["model"]: row
        for row in csv.DictReader((ORIGINAL_DIR / "comparacion_modelos.csv").open(encoding="utf-8"))
    }
    cv_rows = list(csv.DictReader((ORIGINAL_DIR / "resultados_cv.csv").open(encoding="utf-8")))
    checks: dict[str, bool] = {
        "dataset_hash_matches_manifest": existing["data"]["sha256"] == sha256_file(DATA_PATH),
        "dataset_row_count_matches": existing["data"]["rows"] == len(
            list(csv.DictReader(DATA_PATH.open(encoding="utf-8-sig", newline="")))
        ),
    }
    for model, result in existing["models"].items():
        csv_row = comparison[model]
        metrics_match = all(
            math.isclose(float(csv_row[key]), result["test_metrics"][key], rel_tol=1e-12, abs_tol=1e-12)
            for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_ovr_macro")
        )
        folds = [
            row for row in cv_rows
            if row["model"] == model and row["selected"].lower() == "true"
        ]
        selected = folds[0] if len(folds) == 1 else {}
        folds_match = len(folds) == 1 and all(
            math.isclose(
                float(selected[f"split{i}_test_f1_macro"]),
                result["cv_selected_candidate"]["fold_f1_macro"][i],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for i in range(CV_SPLITS)
        )
        with (ORIGINAL_DIR / f"confusion_matrix_{model}.csv").open(encoding="utf-8", newline="") as stream:
            matrix_csv = list(csv.reader(stream))[1:]
        matrix_from_csv = [[int(value) for value in row[1:]] for row in matrix_csv]
        matrix_matches = matrix_from_csv == result["confusion_matrix"]
        matrix_total_matches = sum(map(sum, matrix_from_csv)) == existing["protocol"]["test_rows"]
        supports_match = [
            int(result["classification_report"][label]["support"])
            == sum(result["confusion_matrix"][index])
            for index, label in enumerate(EXPECTED_LABELS)
        ]
        class_metrics_match = True
        for index, label in enumerate(EXPECTED_LABELS):
            tp = matrix_from_csv[index][index]
            predicted_total = sum(matrix_from_csv[row][index] for row in range(len(EXPECTED_LABELS)))
            actual_total = sum(matrix_from_csv[index])
            precision = tp / predicted_total if predicted_total else 0.0
            recall = tp / actual_total if actual_total else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            class_metrics_match &= all(
                math.isclose(result["classification_report"][label][metric], value, rel_tol=1e-12, abs_tol=1e-12)
                for metric, value in (("precision", precision), ("recall", recall), ("f1-score", f1))
            )
        checks[f"{model}_test_metrics_match_comparison_csv"] = metrics_match
        checks[f"{model}_selected_cv_folds_match_csv"] = folds_match
        checks[f"{model}_confusion_matrix_matches_csv"] = matrix_matches
        checks[f"{model}_confusion_matrix_total_matches_test_rows"] = matrix_total_matches
        checks[f"{model}_classification_support_matches_test_matrix"] = all(supports_match)
        checks[f"{model}_per_class_report_metrics_match_matrix"] = class_metrics_match

    # Una segunda ejecución aislada confirma hiperparámetros, CV y prueba del código actual.
    reproduction_path = output_dir / "reproduccion_split_original" / "metricas_experimentos.json"
    if reproduction_path.exists():
        reproduced = json.loads(reproduction_path.read_text(encoding="utf-8"))
    else:
        reproduced = run_experiments(
            DATA_PATH,
            output_dir / "reproduccion_split_original",
            EVIDENCE_DIR / "reproduccion_split_original",
        )
    for model, result in existing["models"].items():
        rerun = reproduced["models"][model]
        reproduced_params = json.loads(json.dumps(rerun["best_params"]))
        checks[f"{model}_params_reproduced"] = result["best_params"] == reproduced_params
        checks[f"{model}_cv_folds_reproduced"] = all(
            math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
            for a, b in zip(
                result["cv_selected_candidate"]["fold_f1_macro"],
                rerun["cv_selected_candidate"]["fold_f1_macro"],
            )
        )
        checks[f"{model}_test_metrics_reproduced"] = all(
            math.isclose(result["test_metrics"][key], rerun["test_metrics"][key], rel_tol=1e-12, abs_tol=1e-12)
            for key in result["test_metrics"]
            if isinstance(result["test_metrics"][key], (int, float))
        )
    return checks


def grouped_evaluation(rows: list[dict[str, str]], output_dir: Path, evidence_dir: Path) -> dict[str, Any]:
    """Ejecuta holdout y CV estratificados por grupo de texto idéntico."""
    texts = [row["text"] for row in rows]
    labels = [row["airline_sentiment"] for row in rows]
    # El primer fold de SGKF funciona como holdout de aproximadamente 20%.
    holdout = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_indices = np.arange(len(rows))
    train_index, test_index = next(holdout.split(all_indices, labels, groups=texts))
    x_train = [texts[index] for index in train_index]
    y_train = [labels[index] for index in train_index]
    train_groups = x_train
    x_test = [texts[index] for index in test_index]
    y_test = [labels[index] for index in test_index]
    overlap = set(train_groups) & set(x_test)
    if overlap:
        raise AssertionError(f"La partición agrupada dejó {len(overlap)} textos compartidos.")

    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    models: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []
    cv_table: list[dict[str, Any]] = []
    confusion_matrices: dict[str, np.ndarray] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("logistic_regression", "random_forest"):
        search = build_search(name, cv)
        start = time.perf_counter()
        search.fit(x_train, y_train, groups=train_groups)
        duration = time.perf_counter() - start
        index = int(search.best_index_)
        scores = [float(search.cv_results_[f"split{i}_test_score"][index]) for i in range(CV_SPLITS)]
        model = search.best_estimator_
        predicted = model.predict(x_test)
        probabilities = model.predict_proba(x_test)
        metrics = evaluate_predictions(y_test, predicted, list(EXPECTED_LABELS), probabilities)
        matrix = confusion_matrix(y_test, predicted, labels=list(EXPECTED_LABELS))
        confusion_matrices[name] = matrix
        report = classification_report(
            y_test, predicted, labels=list(EXPECTED_LABELS), target_names=list(EXPECTED_LABELS),
            output_dict=True, zero_division=0,
        )
        models[name] = {
            "best_params": search.best_params_,
            "best_cv_f1_macro": float(search.best_score_),
            "cv_fold_f1_macro": scores,
            "cv_mean_f1_macro": float(np.mean(scores)),
            "cv_std_f1_macro_sample": float(np.std(scores, ddof=1)),
            "test_metrics": metrics,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "fit_duration_seconds": round(duration, 3),
            "candidate_count": len(search.cv_results_["params"]),
        }
        comparison.append({"model": name, **metrics, "cv_f1_macro_mean": float(np.mean(scores)),
                           "cv_f1_macro_std": float(np.std(scores, ddof=1)), "fit_duration_seconds": round(duration, 3)})
        _write_json(output_dir / f"classification_report_{name}.json", report)
        with (output_dir / f"confusion_matrix_{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["actual\\predicted", *EXPECTED_LABELS])
            writer.writerows([[label, *matrix[row_index].tolist()] for row_index, label in enumerate(EXPECTED_LABELS)])
        for candidate_index, params in enumerate(search.cv_results_["params"]):
            row = {
                "model": name,
                "candidate_index": candidate_index,
                "selected": candidate_index == index,
                "mean_test_f1_macro": float(search.cv_results_["mean_test_score"][candidate_index]),
                "std_test_f1_macro": float(search.cv_results_["std_test_score"][candidate_index]),
            }
            row.update({key: str(value) for key, value in params.items()})
            for fold in range(CV_SPLITS):
                row[f"split{fold}_test_f1_macro"] = float(search.cv_results_[f"split{fold}_test_score"][candidate_index])
            cv_table.append(row)

    report = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "scenario": "Evaluación adicional; holdout y CV agrupados por texto idéntico; no sustituye las métricas originales.",
        "protocol": {
            "splitter": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42); primer fold como holdout",
            "inner_cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)",
            "train_rows": len(train_index),
            "test_rows": len(test_index),
            "test_fraction": len(test_index) / len(rows),
            "train_class_counts": _class_counts(y_train),
            "test_class_counts": _class_counts(y_test),
            "train_unique_text_groups": len(set(x_train)),
            "test_unique_text_groups": len(set(x_test)),
            "train_test_text_overlap": len(overlap),
            "selection_metric": "f1_macro",
            "random_seed": SEED,
            "data_sha256": sha256_file(DATA_PATH),
        },
        "models": models,
    }
    _write_json(output_dir / "evaluacion_agrupada.json", report)
    with (output_dir / "comparacion_agrupada.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    with (output_dir / "resultados_cv_agrupada.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(dict.fromkeys(key for row in cv_table for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cv_table)
    table = [
        "# Evaluación agrupada por texto idéntico",
        "",
        "Escenario complementario. No sustituye la evaluación aleatoria original.",
        "",
        "| Modelo | Accuracy | Precision macro | Recall macro | F1 macro | F1 ponderado | ROC-AUC OVR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        table.append("| " + row["model"] + " | " + " | ".join(
            f"{row[key]:.4f}" for key in
            ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_ovr_macro")
        ) + " |")
    (output_dir / "comparacion_agrupada.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    _plot_results(comparison, confusion_matrices, list(EXPECTED_LABELS), evidence_dir)
    return report


def run_audit() -> dict[str, Any]:
    """Valida fuentes y artefactos, reproduce split original y evalúa grupos."""
    rows, columns = load_dataset(DATA_PATH)
    existing = json.loads((ORIGINAL_DIR / "metricas_experimentos.json").read_text(encoding="utf-8"))
    output_dir = AUDIT_DIR
    if (output_dir.exists() or EVIDENCE_DIR.exists()) and not (
        (output_dir / "reproduccion_split_original" / "metricas_experimentos.json").exists()
        and (output_dir / "evaluacion_agrupada" / "evaluacion_agrupada.json").exists()
    ):
        raise FileExistsError(
            f"La auditoría no está completa y existe {output_dir} o {EVIDENCE_DIR}; no se sobrescribió nada."
        )
    duplicate_analysis = inspect_duplicate_splits(rows)
    checks = verify_existing_outputs(existing, output_dir)
    grouped_path = output_dir / "evaluacion_agrupada" / "evaluacion_agrupada.json"
    grouped = (
        json.loads(grouped_path.read_text(encoding="utf-8"))
        if grouped_path.exists()
        else grouped_evaluation(
            rows,
            output_dir / "evaluacion_agrupada",
            EVIDENCE_DIR / "evaluacion_agrupada",
        )
    )
    source = {
        "dataset_name": "Twitter US Airline Sentiment",
        "publisher_platform": "Kaggle",
        "page_owner": "crowdflower (URL owner); page credits Figure Eight and one collaborator",
        "original_source_entity": "Crowdflower's Data for Everyone library, as stated on the Kaggle dataset page",
        "url": "https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data",
        "consulted_on": "2026-10-17",
        "local_file_match_evidence": "Kaggle lists Tweets.csv as 3.42 MB; the local file is 3,421,431 bytes and has the expected airline sentiment fields. The Kaggle page says its copy is slightly reformatted from the original Crowdflower data.",
        "provenance_limit": "No publisher checksum or local download record exists, so byte-level provenance cannot be authenticated from the project alone.",
        "license_shown_on_page": "CC BY-NC-SA 4.0",
    }
    result = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset_source": source,
        "dataset": {"columns": columns, "rows": len(rows), "sha256": sha256_file(DATA_PATH)},
        "original_split_duplicate_analysis": duplicate_analysis,
        "existing_artifact_checks": checks,
        "all_existing_artifact_checks_pass": all(checks.values()),
        "original_reproduction": "Saved separately under reproduccion_split_original; compare its metrics and params to reportes/entrenamiento.",
        "grouped_evaluation_path": "evaluacion_agrupada/; separate scenario, does not replace original metrics.",
        "grouped_evaluation_protocol": grouped["protocol"],
    }
    _write_json(output_dir / "auditoria.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run_audit()
