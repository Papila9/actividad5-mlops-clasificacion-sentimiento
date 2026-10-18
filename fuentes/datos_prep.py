"""Preparacion auditable del conjunto Twitter US Airline Sentiment.

El archivo fuente no se modifica. La salida conserva solo texto y etiqueta,
normaliza espacios y etiquetas, y excluye filas sin texto o con etiqueta ajena
a las tres clases esperadas. Los duplicados se miden, pero se conservan.
"""

from __future__ import annotations

import argparse
import csv
import html
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_LABELS = ("negative", "neutral", "positive")
REQUIRED_COLUMNS = ("text", "airline_sentiment")


def sha256_file(path: Path) -> str:
    """Devuelve el SHA-256 de un archivo, procesándolo por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Carga el CSV sin inferencia de tipos y comprueba las columnas mínimas."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        return list(reader), columns


def _duplicate_count(rows: list[dict[str, str]], keys: tuple[str, ...]) -> int:
    seen: set[tuple[str, ...]] = set()
    repeated = 0
    for row in rows:
        key = tuple(row.get(name, "") for name in keys)
        if key in seen:
            repeated += 1
        else:
            seen.add(key)
    return repeated


def quality_report(rows: list[dict[str, str]], columns: list[str]) -> dict[str, Any]:
    """Calcula calidad y distribución sobre el CSV fuente, antes de limpiar."""
    null_counts = {
        name: sum(row.get(name) is None or row.get(name) == "" for row in rows)
        for name in columns
    }
    labels = Counter((row.get("airline_sentiment") or "").strip().lower() for row in rows)
    lengths = [len((row.get("text") or "").strip()) for row in rows]
    return {
        "rows": len(rows),
        "columns": columns,
        "null_or_empty_by_column": null_counts,
        "class_counts": {label: labels.get(label, 0) for label in EXPECTED_LABELS},
        "invalid_label_rows": sum(n for label, n in labels.items() if label not in EXPECTED_LABELS),
        "empty_text_rows": sum(length == 0 for length in lengths),
        "duplicate_full_rows": _duplicate_count(rows, tuple(columns)),
        "duplicate_text_label_rows": _duplicate_count(rows, REQUIRED_COLUMNS),
        "text_length_chars": {
            "min": min(lengths, default=0),
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
            "max": max(lengths, default=0),
        },
    }


def clean_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Normaliza y filtra filas; mantiene los duplicados para no borrar evidencia."""
    cleaned: list[dict[str, str]] = []
    excluded_empty_text = 0
    excluded_invalid_label = 0
    for row in rows:
        text = (row.get("text") or "").strip()
        label = (row.get("airline_sentiment") or "").strip().lower()
        if not text:
            excluded_empty_text += 1
            continue
        if label not in EXPECTED_LABELS:
            excluded_invalid_label += 1
            continue
        cleaned.append({"text": text, "airline_sentiment": label})
    return cleaned, {
        "excluded_empty_text": excluded_empty_text,
        "excluded_invalid_label": excluded_invalid_label,
        "duplicate_text_label_rows_retained": _duplicate_count(cleaned, REQUIRED_COLUMNS),
    }


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _create_visualizations(
    original_report: dict[str, Any], cleaned: list[dict[str, str]], output_dir: Path
) -> None:
    """Guarda gráficas SVG reales usando solo la biblioteca estándar."""

    evidence_dir = output_dir.parent.parent / "evidencias"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def bar_chart(filename: str, title: str, labels: list[str], values: list[int]) -> None:
        width = max(760, len(labels) * 74)
        height, left, right, top, bottom = 480, 72, 24, 64, 150
        plot_h = height - top - bottom
        plot_w = width - left - right
        maximum = max(values, default=0) or 1
        step = plot_w / max(1, len(values))
        bars = []
        for i, (label, value) in enumerate(zip(labels, values)):
            bar_h = value / maximum * plot_h
            x = left + i * step + step * 0.18
            y = top + plot_h - bar_h
            bar_w = step * 0.64
            safe_label = html.escape(str(label))
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#4c78a8"/>'
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-size="12">{value}</text>'
                f'<text transform="translate({x + bar_w / 2:.1f},{top + plot_h + 14}) rotate(55)" text-anchor="start" font-size="11">{safe_label}</text>'
            )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<rect width="100%" height="100%" fill="white"/>'
            f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="20" font-family="sans-serif">{html.escape(title)}</text>'
            f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#444"/>'
            + "".join(bars)
            + '</svg>\n'
        )
        (evidence_dir / filename).write_text(svg, encoding="utf-8")

    classes = original_report["class_counts"]
    bar_chart("distribucion_clases.svg", "Distribución de clases (fuente)", list(classes), list(classes.values()))
    missing = original_report["null_or_empty_by_column"]
    bar_chart("valores_faltantes.svg", "Valores nulos o celdas vacías por columna", list(missing), list(missing.values()))

    lengths = [len(row["text"]) for row in cleaned]
    max_length = max(lengths, default=0)
    bin_count = 12
    counts = [0] * bin_count
    for length in lengths:
        index = min(bin_count - 1, length * bin_count // (max_length + 1)) if max_length else 0
        counts[index] += 1
    labels = [f"{i * (max_length + 1) // bin_count}-{((i + 1) * (max_length + 1) // bin_count) - 1}" for i in range(bin_count)]
    bar_chart("longitud_textos.svg", "Longitud de textos limpios (caracteres)", labels, counts)

    duplicate_values = [original_report["duplicate_text_label_rows"], original_report["duplicate_full_rows"]]
    bar_chart("duplicados.svg", "Registros duplicados en la fuente", ["Texto + etiqueta", "Fila completa"], duplicate_values)


def prepare_dataset(source: Path, output_dir: Path, original_dir: Path | None = None) -> dict[str, Any]:
    """Genera CSV limpio, informe de calidad y manifiesto con hashes."""
    source = source.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if original_dir is not None:
        original_dir.mkdir(parents=True, exist_ok=True)
        preserved = original_dir / source.name
        if source != preserved.resolve():
            shutil.copy2(source, preserved)
            if sha256_file(source) != sha256_file(preserved):
                raise OSError("La copia del CSV original no coincide con el SHA-256 fuente.")

    rows, columns = load_dataset(source)
    before = quality_report(rows, columns)
    cleaned, cleaning = clean_rows(rows)
    clean_path = output_dir / "Tweets_limpios.csv"
    _write_csv(cleaned, clean_path)

    _create_visualizations(before, cleaned, output_dir)
    visual_warning = None
    processed_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    after = quality_report(cleaned, list(REQUIRED_COLUMNS))
    quality = {
        "processed_at": processed_at,
        "source_file": source.name,
        "source_sha256": sha256_file(source),
        "before_cleaning": before,
        "cleaning_actions": cleaning,
        "after_cleaning": after,
        "visualization_note": visual_warning,
    }
    quality_path = output_dir / "calidad_datos.json"
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "processed_at": processed_at,
        "source": {
            "path": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "rows": len(rows),
        },
        "processed": {
            "path": clean_path.name,
            "size_bytes": clean_path.stat().st_size,
            "sha256": sha256_file(clean_path),
            "rows": len(cleaned),
            "columns": list(REQUIRED_COLUMNS),
        },
        "cleaning_rules": [
            "Se conservan únicamente text y airline_sentiment.",
            "Se eliminan espacios al inicio y final del texto.",
            "Se recortan espacios y se convierten etiquetas a minúsculas.",
            "Se excluyen textos vacíos y etiquetas fuera de negative, neutral, positive.",
            "Se conservan duplicados texto-etiqueta; se contabilizan para evaluación posterior.",
        ],
        "cleaning_actions": cleaning,
        "quality_report": quality_path.name,
    }
    (output_dir / "manifiesto.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"quality": quality, "manifest": manifest}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "datos" / "datos_ini" / "Tweets.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "datos" / "datos_limp")
    parser.add_argument("--original-dir", type=Path, default=None)
    args = parser.parse_args()
    result = prepare_dataset(args.source, args.output_dir, args.original_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
