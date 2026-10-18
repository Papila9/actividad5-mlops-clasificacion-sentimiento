import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from fuentes.datos_prep import clean_rows, load_dataset, prepare_dataset, quality_report


class DataPreparationTests(unittest.TestCase):
    def test_quality_and_duplicate_retention(self):
        rows = [
            {"text": " first ", "airline_sentiment": " negative ", "extra": "x"},
            {"text": " first ", "airline_sentiment": " negative ", "extra": "x"},
            {"text": "", "airline_sentiment": "neutral", "extra": ""},
            {"text": "other", "airline_sentiment": "unknown", "extra": "z"},
        ]
        report = quality_report(rows, ["text", "airline_sentiment", "extra"])
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["empty_text_rows"], 1)
        self.assertEqual(report["invalid_label_rows"], 1)
        self.assertEqual(report["duplicate_text_label_rows"], 1)
        cleaned, actions = clean_rows(rows)
        self.assertEqual(cleaned, [
            {"text": "first", "airline_sentiment": "negative"},
            {"text": "first", "airline_sentiment": "negative"},
        ])
        self.assertEqual(actions["excluded_empty_text"], 1)
        self.assertEqual(actions["excluded_invalid_label"], 1)
        self.assertEqual(actions["duplicate_text_label_rows_retained"], 1)

    def test_missing_required_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.csv"
            path.write_text("text,other\nhello,x\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "airline_sentiment"):
                load_dataset(path)

    def test_output_hashes_and_dimensions_are_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.csv"
            source.write_text(
                "text,airline_sentiment,unused\nhello,positive,x\nhello,positive,x\n",
                encoding="utf-8",
            )
            result = prepare_dataset(source, root / "datos_limp", root / "datos_ini")
            clean = root / "datos_limp" / "Tweets_limpios.csv"
            self.assertEqual(result["manifest"]["source"]["rows"], 2)
            self.assertEqual(result["manifest"]["processed"]["rows"], 2)
            self.assertEqual(result["quality"]["after_cleaning"]["duplicate_text_label_rows"], 1)
            self.assertEqual(
                result["manifest"]["source"]["sha256"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["manifest"]["processed"]["sha256"],
                hashlib.sha256(clean.read_bytes()).hexdigest(),
            )
            with clean.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)


if __name__ == "__main__":
    unittest.main()
