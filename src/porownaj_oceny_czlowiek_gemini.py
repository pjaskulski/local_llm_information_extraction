"""Porównanie ocen człowieka i Gemini dla danych testowych SGKP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "test" / "dane_testowe_llm.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "test" / "niezgodnosci_czlowiek_gemini.txt"
DEFAULT_HUMAN_TRUE_GEMINI_FALSE_REPORT_PATH = (
    PROJECT_ROOT / "test" / "czlowiek_true_gemini_false.txt"
)

FIELDS = [
    "typ",
    "warianty_nazw",
    "powiat_ocr",
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
    "powiat_ocr": "Powiat",
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


def format_json(value: Any) -> str:
    if value is None:
        return "BRAK"
    return json.dumps(value, ensure_ascii=False, indent=2)


def format_bool(value: bool) -> str:
    return "True" if value else "False"


def compare(
    data: list[dict[str, Any]],
    fields: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    stats = {
        field: {
            "total": 0,
            "agree": 0,
            "disagree": 0,
            "human_false": 0,
            "human_false_gemini_false": 0,
            "human_true_gemini_false": 0,
        }
        for field in fields
    }
    disagreements: list[dict[str, Any]] = []
    human_true_gemini_false: list[dict[str, Any]] = []
    skipped_counts = {field: 0 for field in fields}

    for record in data:
        for field in fields:
            human_field = f"r_{field}"
            gemini_field = f"gf_{field}"

            if human_field not in record or gemini_field not in record:
                if human_field in record and gemini_field not in record:
                    skipped_counts[field] += 1
                continue

            human_value = record[human_field]
            gemini_value = record[gemini_field]
            if not isinstance(human_value, bool) or not isinstance(gemini_value, bool):
                skipped_counts[field] += 1
                continue

            stats[field]["total"] += 1
            if human_value == gemini_value:
                stats[field]["agree"] += 1
            else:
                stats[field]["disagree"] += 1
                disagreements.append(
                    {
                        "field": field,
                        "record": record,
                        "human_value": human_value,
                        "gemini_value": gemini_value,
                    }
                )
                if human_value is True and gemini_value is False:
                    human_true_gemini_false.append(disagreements[-1])

            if human_value is False:
                stats[field]["human_false"] += 1
                if gemini_value is False:
                    stats[field]["human_false_gemini_false"] += 1
            if human_value is True and gemini_value is False:
                stats[field]["human_true_gemini_false"] += 1

    rows = []
    for field in fields:
        total = stats[field]["total"]
        agree = stats[field]["agree"]
        disagree = stats[field]["disagree"]
        agree_pct = (agree / total * 100) if total else 0.0
        disagree_pct = (disagree / total * 100) if total else 0.0
        human_false = stats[field]["human_false"]
        human_false_gemini_false = stats[field]["human_false_gemini_false"]
        human_false_agree_pct = (
            human_false_gemini_false / human_false * 100
            if human_false
            else 0.0
        )
        rows.append(
            {
                "field": field,
                "label": FIELD_LABELS.get(field, field),
                "total": total,
                "agree": agree,
                "disagree": disagree,
                "agree_pct": agree_pct,
                "disagree_pct": disagree_pct,
                "human_false": human_false,
                "human_false_gemini_false": human_false_gemini_false,
                "human_false_agree_pct": human_false_agree_pct,
                "human_true_gemini_false": stats[field]["human_true_gemini_false"],
            }
        )

    total = sum(row["total"] for row in rows)
    agree = sum(row["agree"] for row in rows)
    disagree = sum(row["disagree"] for row in rows)
    human_false = sum(row["human_false"] for row in rows)
    human_false_gemini_false = sum(row["human_false_gemini_false"] for row in rows)
    human_true_gemini_false_count = sum(row["human_true_gemini_false"] for row in rows)
    rows.append(
        {
            "field": "RAZEM",
            "label": "RAZEM",
            "total": total,
            "agree": agree,
            "disagree": disagree,
            "agree_pct": (agree / total * 100) if total else 0.0,
            "disagree_pct": (disagree / total * 100) if total else 0.0,
            "human_false": human_false,
            "human_false_gemini_false": human_false_gemini_false,
            "human_false_agree_pct": (
                human_false_gemini_false / human_false * 100
                if human_false
                else 0.0
            ),
            "human_true_gemini_false": human_true_gemini_false_count,
        }
    )

    return rows, disagreements, human_true_gemini_false, {
        field: count for field, count in skipped_counts.items() if count
    }


def print_plain_table(
    headers: list[str],
    table_rows: list[list[str]],
    right_align: set[int] | None = None,
) -> None:
    if not table_rows:
        return
    right_align = right_align or set()

    widths = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in table_rows))
        for col_idx in range(len(headers))
    ]

    def align_cell(value: str, col_idx: int) -> str:
        if col_idx in right_align:
            return value.rjust(widths[col_idx])
        return value.ljust(widths[col_idx])

    header_line = " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
    separator = "-+-".join("-" * width for width in widths)

    print(header_line)
    print(separator)
    for row in table_rows:
        print(" | ".join(align_cell(row[idx], idx) for idx in range(len(row))))


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    headers = ["Pole", "Liczba ocen", "Zgodne", "Niezgodne", "% zgodne", "% niezgodne"]
    table_rows = [
        [
            row["label"],
            str(row["total"]),
            str(row["agree"]),
            str(row["disagree"]),
            f"{row['agree_pct']:.2f}",
            f"{row['disagree_pct']:.2f}",
        ]
        for row in rows
    ]
    print("Tabela 1. Zgodność ocen człowieka i Gemini")
    print_plain_table(headers, table_rows, right_align=set(range(1, len(headers))))


def print_false_analysis_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "Pole",
        "Człowiek False",
        "Gemini też False",
        "% zgodnych False",
        "Człowiek True/Gemini False",
    ]
    table_rows = [
        [
            row["label"],
            str(row["human_false"]),
            str(row["human_false_gemini_false"]),
            f"{row['human_false_agree_pct']:.2f}",
            str(row["human_true_gemini_false"]),
        ]
        for row in rows
    ]
    print("\nTabela 2. Analiza ocen False")
    print_plain_table(headers, table_rows, right_align=set(range(1, len(headers))))


def print_tables(rows: list[dict[str, Any]]) -> None:
    print_summary_table(rows)
    print_false_analysis_table(rows)


def write_disagreement_report(path: Path, disagreements: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for item in disagreements:
            record = item["record"]
            field = item["field"]
            human_field = f"r_{field}"
            gemini_field = f"gf_{field}"

            output_file.write(f"ID: {record.get('ID', '')}\n")
            output_file.write(f"Nazwa: {record.get('nazwa', '')}\n")
            output_file.write(f"Tom: {record.get('tom', '')}, strona: {record.get('strona', '')}\n")
            output_file.write(f"Pole: {field}\n")
            output_file.write("\nTekst hasła:\n")
            output_file.write(f"{record.get('text', '')}\n")
            output_file.write("\nWynik modelu GPT:\n")
            output_file.write(format_json(record.get(field)))
            output_file.write("\n\nOcena człowieka:\n")
            output_file.write(f"{human_field}: {format_bool(item['human_value'])}\n")
            output_file.write("\nOcena Gemini:\n")
            output_file.write(f"{gemini_field}: {format_bool(item['gemini_value'])}\n")
            output_file.write("\n" + "-" * 80 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Porównanie pól r_* i gf_* w pliku dane_testowe_llm.json."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Plik JSON z ocenami.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH), help="Plik TXT z niezgodnościami.")
    parser.add_argument(
        "--human-true-gemini-false-report",
        default=str(DEFAULT_HUMAN_TRUE_GEMINI_FALSE_REPORT_PATH),
        help="Plik TXT z przypadkami człowiek=True i Gemini=False.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data = load_json(Path(args.input))
    rows, disagreements, human_true_gemini_false, skipped_fields = compare(data, FIELDS)

    print_tables(rows)
    write_disagreement_report(Path(args.report), disagreements)
    write_disagreement_report(
        Path(args.human_true_gemini_false_report),
        human_true_gemini_false,
    )
    print(f"\nZapisano raport niezgodności: {args.report}")
    print(f"Liczba niezgodności: {len(disagreements)}")
    print(
        "Zapisano raport człowiek=True/Gemini=False: "
        f"{args.human_true_gemini_false_report}"
    )
    print(f"Liczba przypadków człowiek=True/Gemini=False: {len(human_true_gemini_false)}")

    if skipped_fields:
        print("\nPominięte porównania z powodu braku pary boolowskich pól r_*/gf_*:")
        for field, count in skipped_fields.items():
            print(f"- {FIELD_LABELS.get(field, field)}: {count}")


if __name__ == "__main__":
    main()
