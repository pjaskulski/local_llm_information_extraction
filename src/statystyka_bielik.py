"""Statystyka ocen Gemini dla wyników ekstrakcji Bielika."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "test" / "dane_testowe_bielik_llm.json"

FIELDS = [
    "typ",
    "warianty_nazw",
    "powiat",
    "gmina",
    "gubernia",
    "parafia_katolicka",
    "parafia_inna",
    "autor",
    "właściciel",
    "l_mk_statystyka",
    "l_dm_statystyka",
    "własność_ziemska",
    "ludność_wyznanie",
]

FIELD_LABELS = {
    "typ": "Typ",
    "warianty_nazw": "Warianty nazw",
    "powiat": "Powiat",
    "gmina": "Gmina",
    "gubernia": "Gubernia",
    "parafia_katolicka": "Parafia katolicka",
    "parafia_inna": "Parafia inna",
    "autor": "Autor",
    "właściciel": "Właściciel",
    "l_mk_statystyka": "Liczba mieszkańców",
    "l_dm_statystyka": "Liczba domów",
    "własność_ziemska": "Własność ziemska",
    "ludność_wyznanie": "Ludność wg wyznań",
}


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list):
        raise ValueError(f"Plik {path} nie zawiera listy rekordów JSON.")
    return data


def calculate_stats(data: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    record_count = len(data)

    for field in fields:
        judge_field = f"bl_{field}"
        true_count = 0
        false_count = 0
        other_count = 0
        missing_count = 0

        for record in data:
            if judge_field not in record:
                missing_count += 1
                continue

            value = record[judge_field]
            if value is True:
                true_count += 1
            elif value is False:
                false_count += 1
            else:
                other_count += 1

        evaluated = true_count + false_count
        rows.append(
            {
                "field": field,
                "label": FIELD_LABELS.get(field, field),
                "records": record_count,
                "evaluated": evaluated,
                "true": true_count,
                "false": false_count,
                "missing": missing_count,
                "other": other_count,
                "true_pct": (true_count / evaluated * 100) if evaluated else 0.0,
                "false_pct": (false_count / evaluated * 100) if evaluated else 0.0,
            }
        )

    total_evaluated = sum(row["evaluated"] for row in rows)
    total_true = sum(row["true"] for row in rows)
    total_false = sum(row["false"] for row in rows)
    total_missing = sum(row["missing"] for row in rows)
    total_other = sum(row["other"] for row in rows)
    rows.append(
        {
            "field": "RAZEM",
            "label": "RAZEM",
            "records": record_count * len(fields),
            "evaluated": total_evaluated,
            "true": total_true,
            "false": total_false,
            "missing": total_missing,
            "other": total_other,
            "true_pct": (total_true / total_evaluated * 100) if total_evaluated else 0.0,
            "false_pct": (total_false / total_evaluated * 100) if total_evaluated else 0.0,
        }
    )
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "Pole",
        "Ocenione",
        "Poprawne",
        "Błędne",
        "% poprawne",
        "% błędne",
    ]
    table_rows = [
        [
            row["label"],
            str(row["evaluated"]),
            str(row["true"]),
            str(row["false"]),
            f"{row['true_pct']:.2f}",
            f"{row['false_pct']:.2f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in table_rows))
        for col_idx in range(len(headers))
    ]
    numeric_cols = set(range(1, len(headers)))

    def align(value: str, col_idx: int) -> str:
        if col_idx in numeric_cols:
            return value.rjust(widths[col_idx])
        return value.ljust(widths[col_idx])

    print("Statystyka ocen Gemini dla ekstrakcji Bielika")
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(align(row[idx], idx) for idx in range(len(row))))


def parse_fields(fields_arg: str) -> list[str]:
    if fields_arg.strip().lower() == "all":
        return FIELDS
    fields = [field.strip() for field in fields_arg.split(",") if field.strip()]
    unknown = [field for field in fields if field not in FIELDS]
    if unknown:
        raise ValueError(f"Nieznane pola: {', '.join(unknown)}")
    if not fields:
        raise ValueError("Lista pól jest pusta.")
    return fields


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Statystyka pól bl_* z pliku dane_testowe_bielik_llm.json."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Plik wejściowy JSON.")
    parser.add_argument(
        "--fields",
        default="all",
        help="Pola do uwzględnienia: all albo lista po przecinku, np. typ,powiat.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data = load_json(Path(args.input))
    fields = parse_fields(args.fields)
    rows = calculate_stats(data, fields)
    print_table(rows)


if __name__ == "__main__":
    main()
