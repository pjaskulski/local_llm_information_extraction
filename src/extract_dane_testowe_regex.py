"""Heurystyczna ekstrakcja danych testowych SGKP bez LLM.

Skrypt używa prostych reguł i wyrażeń regularnych. Nie zastępuje ekstrakcji LLM,
ale daje porównawczy baseline dla schematycznych haseł SGKP.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "test" / "dane_testowe.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "test" / "dane_testowe_regex.json"
BASE_FIELDS = ["nazwa", "text", "tom", "strona", "ID"]

REFERRAL_RE = re.compile(r"(^|[\s,;])ob\.?($|[\s,;])", re.IGNORECASE)
YEAR_RE = re.compile(r"(?:w|W)\s*(?:r\.\s*)?(\d{4})\s*r?\.?", re.IGNORECASE)
NUM_RE = r"(\d+(?:[.,'\u2019]\d+)?)"

LANG_PREFIXES = {
    "niem.": "niem.",
    "ros.": "ros.",
    "węg.": "węg.",
    "łac.": "łac.",
    "czes.": "czes.",
    "rus.": "rus.",
    "litew.": "litew.",
    "pol.": "pol.",
}

TYPE_PATTERNS = [
    (r"\bwieś rządowa\b", "wieś rządowa"),
    (r"\bwś rząd\.?\b", "wieś rządowa"),
    (r"\bwś\b|\bwieś\b", "wieś"),
    (r"\bfolw?\.?\b|\bfolwark\b", "folwark"),
    (r"\bmczko\b|\bmko\b|\bmiasteczko\b", "miasteczko"),
    (r"\bm\.\s*pow\.|\bmiasto\b|\bm\.\b", "miasto"),
    (r"\bos\.|\bosada\b", "osada"),
    (r"\bkol\.|\bkolonia\b", "kolonia"),
    (r"\bprzys\.|\bprzysiółek\b", "przysiółek"),
    (r"\bdobra\b", "dobra"),
    (r"\bjez\.|\bjezioro\b", "jezioro"),
    (r"\brz\.|\brzeka\b", "rzeka"),
    (r"\bpotok\b", "potok"),
    (r"\bgóra\b|\bgórny\b", "góra"),
    (r"\blas\b", "las"),
    (r"\bst\.\s*p\.", "stacja pocztowa"),
    (r"\bleśn\.|\bleśniczówka\b", "leśniczówka"),
    (r"\bmłyn\b", "młyn"),
]

RELIGION_RE = re.compile(
    rf"\b(rz[.-]?\s*kat\.?|rzym[.-]?\s*kat\.?|kat\.?|gr[.-]?\s*kat\.?|praw\.?|prawosł\.?|ew\.?|ewang\.?|żyd\.?|izrael\.?)\s*{NUM_RE}",
    re.IGNORECASE,
)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list):
        raise ValueError(f"Plik {path} nie zawiera listy rekordów JSON.")
    return data


def save_json_atomic(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp_file:
        json.dump(data, tmp_file, ensure_ascii=False, indent=4)
        tmp_name = tmp_file.name
    Path(tmp_name).replace(path)


def strip_parentheses(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def clean_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" \t\n\r,.;:")
    value = re.sub(r"\s+(?:w|we|na|od|do)$", "", value, flags=re.IGNORECASE)
    return value.strip(" \t\n\r,.;:")


def clean_author(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\n\r,;:")


def normalize_number(value: str) -> str:
    return value.replace("\u2019", ",").replace("'", ",")


def normalize_unit(unit: str) -> str:
    unit = unit.strip().lower()
    if unit.startswith("mór") or unit.startswith("morg") or unit == "mr":
        return "mr."
    if unit.startswith("dzies") or unit == "dz":
        return "dz."
    if unit.startswith("ha"):
        return "ha"
    return unit


def first_part(text: str, max_chars: int = 320) -> str:
    return strip_parentheses(text[:max_chars])


def extract_type(text: str) -> list[str] | None:
    part = first_part(text, 140)
    result: list[str] = []
    if REFERRAL_RE.search(part):
        result.append("odsyłacz")
    for pattern, label in TYPE_PATTERNS:
        if label == "rzeka" and re.search(r"\bnad\s+rz\.", part, re.IGNORECASE):
            continue
        if re.search(pattern, part, re.IGNORECASE) and label not in result:
            result.append(label)
    if "wieś rządowa" in result and "wieś" in result:
        result.remove("wieś")
    return result or None


def extract_variants(record: dict[str, Any]) -> list[dict[str, str]] | None:
    name = str(record.get("nazwa") or "")
    text = str(record.get("text") or "")
    if not name:
        return None

    part = first_part(text, 220)
    markers = [
        r"\bwś\b", r"\bwieś\b", r"\bfolw?\.?\b", r"\bpow\.", r"\bgm\.",
        r"\bpar[.,]\b", r"\bgub\.", r"\bob\.?\b", r"\bmczko\b", r"\bmko\b",
        r"\bmiasto\b", r"\bjez\.", r"\brz\.", r"\bpotok\b",
    ]
    marker_match = re.search("|".join(markers), part, re.IGNORECASE)
    head = part[: marker_match.start()] if marker_match else part
    if head.lower().startswith(name.lower()):
        head = head[len(name):]
    head = head.strip(" ,.;")
    if not head:
        return None

    variants: list[dict[str, str]] = []
    for raw_item in re.split(r",|\bal\.\b|\balbo\b|\bczyli\b", head):
        item = clean_value(raw_item)
        item = re.sub(r"^\d+\.\)\s*", "", item).strip()
        if not item or item.lower() == name.lower():
            continue
        if not re.search(r"[A-Za-zĄĆĘŁŃÓŚŻŹąćęłńóśżź]", item):
            continue
        if re.fullmatch(r"(?:st\.?\s*p\.?|wś|wieś|folw?\.?)", item, re.IGNORECASE):
            continue
        lang = "nieokr."
        for prefix, normalized in LANG_PREFIXES.items():
            if item.lower().startswith(prefix):
                lang = normalized
                item = clean_value(item[len(prefix):])
                break
        if item and item.lower() != name.lower():
            variants.append({"lang": lang, "wariant_nazwy": item})
    return variants or None


def extract_after_abbrev(text: str, abbrev_pattern: str) -> str | None:
    match = re.search(
        rf"{abbrev_pattern}\s*([^,.;()]+)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = clean_value(match.group(1))
    return value or None


def extract_admin(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    powiat = extract_after_abbrev(text, r"\bpow\.")
    if powiat:
        result["powiat_ocr"] = powiat

    gm_par = re.search(r"\bgm\.\s*i\s*par[.,]?\s*([^,.;()]+)", text, re.IGNORECASE)
    if gm_par:
        value = clean_value(gm_par.group(1))
        if value:
            result["gmina"] = value
            result["parafia_katolicka"] = value
    else:
        gmina = extract_after_abbrev(text, r"\bgm[.,]")
        if gmina:
            result["gmina"] = gmina

    gubernia = extract_after_abbrev(text, r"\bgub[.,]")
    if gubernia:
        result["gubernia"] = gubernia
    return result


def extract_parishes(record: dict[str, Any], result: dict[str, Any]) -> None:
    text = str(record.get("text") or "")
    name = str(record.get("nazwa") or "")

    catholic_patterns = [
        r"\bpar[.,]?\s+(?:kat\.?|katol\.?|rz[.-]?\s*kat\.?|rzym[.-]?\s*kat\.?)?\s*([^,.;()]+)",
        r"\bw\s+parafii\s+([^,.;()]+)",
    ]
    for pattern in catholic_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = clean_value(match.group(1))
            if value and not re.match(r"gr[.-]?\s*kat|ew|praw", value, re.IGNORECASE):
                result.setdefault("parafia_katolicka", value)
                break

    other: list[dict[str, str]] = []
    for match in re.finditer(
        r"\bpar[.,]?\s+(gr[.-]?\s*kat\.?|praw\.?|prawosł\.?|ew\.?|ewang\.?)\s*(?:w\s+)?([^,.;()]*)",
        text,
        re.IGNORECASE,
    ):
        wyznanie = clean_value(match.group(1))
        value = clean_value(match.group(2))
        if not value or value.lower().startswith("miejsc"):
            value = name
        if wyznanie and value:
            other.append({"wyznanie": wyznanie, "nazwa_parafii": value})

    if not other and re.search(r"cerkiew\s+par", text, re.IGNORECASE) and name:
        other.append({"wyznanie": "praw.", "nazwa_parafii": name})
    if other:
        result["parafia_inna"] = other


def extract_author(text: str) -> str | None:
    text = text.strip()
    match = re.search(
        r"([A-ZŁŚŻŹĆŃÓ][a-ząćęłńóśżź]+|(?:[A-ZŁŚŻŹĆŃÓ]\.\s*){1,3}[A-ZŁŚŻŹĆŃÓ]?[a-ząćęłńóśżź]*\.?)\s*$",
        text,
    )
    if not match:
        return None
    value = clean_author(match.group(1))
    if value.lower() in {"ob", "par", "gm", "pow", "gub"}:
        return None
    if len(value) < 2:
        return None
    return value


def year_for_position(text: str, pos: int) -> str:
    nearby = text[max(0, pos - 45):pos].lower()
    if re.search(r"\b(obecnie|teraz|teraz zaś|dziś|dzisiaj)\b", nearby):
        return "obecnie"
    last_year = None
    for match in YEAR_RE.finditer(text[:pos]):
        last_year = match.group(1)
    if last_year:
        return last_year
    return "obecnie"


def append_stat(stats: dict[str, list[dict[str, Any]]], field: str, date: str, number: str) -> None:
    item = {"data": date, "liczba": number}
    if not stats[field]:
        stats[field].append({"dotyczy": "główna miejscowość", "liczba": [item]})
        return
    if item not in stats[field][0]["liczba"]:
        stats[field][0]["liczba"].append(item)


def extract_statistics(text: str) -> dict[str, Any]:
    stats: dict[str, list[dict[str, Any]]] = {
        "l_mk_statystyka": [],
        "l_dm_statystyka": [],
    }
    for match in re.finditer(rf"{NUM_RE}\s*(dm\.?|domów|domy)\b", text, re.IGNORECASE):
        append_stat(stats, "l_dm_statystyka", year_for_position(text, match.start()), normalize_number(match.group(1)))
    for match in re.finditer(rf"{NUM_RE}\s*(mk\.?|mieszkańców|mieszk\.)\b", text, re.IGNORECASE):
        append_stat(stats, "l_mk_statystyka", year_for_position(text, match.start()), normalize_number(match.group(1)))

    return {field: values for field, values in stats.items() if values}


def extract_religion(text: str) -> list[dict[str, Any]] | None:
    items: list[dict[str, str]] = []
    for match in RELIGION_RE.finditer(text):
        wyznanie = clean_value(match.group(1))
        liczba = normalize_number(match.group(2))
        if wyznanie and liczba:
            items.append({"wyznanie_ocr": wyznanie, "liczba": liczba})
    if not items:
        return None
    seen = set()
    unique = []
    for item in items:
        key = (item["wyznanie_ocr"].lower(), item["liczba"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return [{"dotyczy": "główna miejscowość", "struktura_wyznaniowa": unique}]


def extract_owner(text: str) -> str | None:
    patterns = [
        r"(?:własność|własnością)\s+([^,.;]+)",
        r"(?:należy do|należał[ay]? do)\s+([^,.;]+)",
        r"(?:dziedzic|dziedziczka)\s+([A-ZŁŚŻŹĆŃÓ][A-Za-zĄĆĘŁŃÓŚŻŹąćęłńóśżź-]+)",
        r"([A-ZŁŚŻŹĆŃÓ][A-Za-zĄĆĘŁŃÓŚŻŹąćęłńóśżź-]+)\s+ma\s+\d+\s*(?:dz|dzies|mr|morg)",
    ]
    owners = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = clean_value(match.group(1))
            value = re.sub(r"\s+(?:ma|posiada).*$", "", value, flags=re.IGNORECASE)
            if value and not re.search(r"\bdóbr\b|^s\.?$|^s\.?\s*gm\b|^okr\b", value, re.IGNORECASE):
                owners.append(value)
    if re.search(r"\bwieś rządowa\b|\brządowe\b|\bskarbowa\b", text, re.IGNORECASE):
        owners.append("rząd/skarb państwa")
    unique = []
    seen = set()
    for owner in owners:
        key = owner.lower()
        if key not in seen:
            seen.add(key)
            unique.append(owner)
    return ", ".join(unique) if unique else None


def extract_land(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    text = str(record.get("text") or "")
    name = str(record.get("nazwa") or "miejscowość")
    land: list[dict[str, str]] = []

    total_patterns = [
        rf"(?:ziemi|ziemia|gruntu|gruntów|obszaru|przestrzeni)\s+{NUM_RE}\s*(ha|mr\.?|morg[óo]w|dz\.?|dziesięcin|dzies\.)",
        rf"{NUM_RE}\s*(ha|mr\.?|morg[óo]w|dz\.?|dziesięcin|dzies\.)\s+(?:ziemi|gruntu|obszaru)",
    ]
    for pattern in total_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            number, unit = match.group(1), match.group(2)
            item = {"type_of_ground": "obszar ogółem", "area_of_ground": f"{normalize_number(number)} {normalize_unit(unit)}"}
            if item not in land:
                land.append(item)

    ground_patterns = [
        (r"roli(?:\s+or\.?)?|rola(?:\s+orna)?|gr\.\s*or\.?|orne", "rola orna"),
        (r"ogrod[óo]w|ogr\.?", "ogrody"),
        (r"łąk|łąki", "łąki"),
        (r"pastw\.?|pastwisk", "pastwiska"),
        (r"lasu|lasów|lasy|boru", "lasy"),
        (r"nieużytk[óo]w|nieużyt\.?|nieuz\.?", "nieużytki"),
        (r"wody", "wody"),
        (r"włość\.?", "włość"),
    ]
    for ground_pattern, label in ground_patterns:
        pattern_a = rf"{ground_pattern}\s+{NUM_RE}\s*(ha|mr\.?|morg[óo]w|dz\.?|dziesięcin|dzies\.)?"
        pattern_b = rf"{NUM_RE}\s*(ha|mr\.?|morg[óo]w|dz\.?|dziesięcin|dzies\.)?\s+{ground_pattern}"
        for pattern in (pattern_a, pattern_b):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                number, unit = match_number_and_unit(match, text)
                if not number:
                    continue
                item = {"type_of_ground": label, "area_of_ground": f"{number} {unit}".strip()}
                if item not in land:
                    land.append(item)

    for match in re.finditer(
        rf"(?:dziedzic|dziedziczka)\s+[A-ZŁŚŻŹĆŃÓ][A-Za-zĄĆĘŁŃÓŚŻŹąćęłńóśżź-]+\s+ma\s+{NUM_RE}\s*(dz\.?|dziesięcin|dzies\.|mr\.?|morg[óo]w|ha)",
        text,
        re.IGNORECASE,
    ):
        item = {
            "type_of_ground": "własność dziedzica",
            "area_of_ground": f"{normalize_number(match.group(1))} {normalize_unit(match.group(2))}",
        }
        if item not in land:
            land.append(item)

    if not land:
        return None
    return [{"land_name": f"główna miejscowość {name}".strip(), "land": land}]


def infer_nearby_unit(text: str, pos: int) -> str:
    window = text[max(0, pos - 80): pos + 80]
    match = re.search(r"\b(ha|mr\.?|morg[óo]w|dz\.?|dziesięcin|dzies\.)\b", window, re.IGNORECASE)
    if not match:
        return ""
    return normalize_unit(match.group(1))


def match_number_and_unit(match: re.Match[str], text: str) -> tuple[str | None, str]:
    number = None
    unit = ""
    for group in match.groups():
        if not group:
            continue
        if re.fullmatch(NUM_RE, group):
            number = normalize_number(group)
        elif re.fullmatch(r"ha|mr\.?|morg[óo]w|dz\.?|dziesięcin|dzies\.", group, re.IGNORECASE):
            unit = normalize_unit(group)
    if number and not unit:
        unit = infer_nearby_unit(text, match.start())
    return number, unit


def prune_empty(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if value not in (None, "", [], {})
    }


def extract_record(source: dict[str, Any]) -> dict[str, Any]:
    text = str(source.get("text") or "")
    result = {field: source.get(field) for field in BASE_FIELDS}

    result["typ"] = extract_type(text)
    result["warianty_nazw"] = extract_variants(source)
    result.update(extract_admin(text))
    extract_parishes(source, result)
    result["autor"] = extract_author(text)
    result.update(extract_statistics(text))
    result["ludność_wyznanie"] = extract_religion(text)
    result["właściciel"] = extract_owner(text)
    result["własność_ziemska"] = extract_land(source)
    return prune_empty(result)


def process(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    data = load_json(input_path)
    if args.limit is not None:
        indexes = range(max(args.start, 0), min(len(data), max(args.start, 0) + args.limit))
    else:
        indexes = range(max(args.start, 0), len(data))

    output: list[dict[str, Any]] = []
    selected = set(indexes)
    for idx, record in enumerate(data):
        if idx in selected:
            output.append(extract_record(record))
        else:
            output.append({field: record.get(field) for field in BASE_FIELDS})

    save_json_atomic(output_path, output)
    print(f"Przetworzono rekordy: {len(selected)}")
    print(f"Zapisano wyniki w: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Heurystyczna ekstrakcja danych SGKP wyrażeniami regularnymi."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Plik wejściowy JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Plik wynikowy JSON.")
    parser.add_argument("--start", type=int, default=0, help="Indeks pierwszego rekordu.")
    parser.add_argument("--limit", type=int, default=None, help="Maksymalna liczba rekordów.")
    return parser


if __name__ == "__main__":
    process(build_arg_parser().parse_args())
