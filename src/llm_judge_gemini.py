"""LLM-as-a-Judge dla danych testowych SGKP z użyciem Gemini.

Skrypt weryfikuje pola wydobyte wcześniej przez GPT na podstawie tekstu hasła
i zapisuje wyniki oceny w polach z przedrostkiem ``gf_``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback dla środowisk bez python-dotenv
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "test" / "dane_testowe.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "test" / "dane_testowe_llm.json"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

TARGET_FIELDS = [
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

GF_FIELDS = [f"gf_{field}" for field in TARGET_FIELDS]
DONE_FIELD = "_gf_judge_done"
STATUS_CORRECT = "correct"
STATUS_INCORRECT = "incorrect"
STATUS_NOT_APPLICABLE = "not_applicable"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        field: {
            "type": "string",
            "enum": [STATUS_CORRECT, STATUS_INCORRECT, STATUS_NOT_APPLICABLE],
            "description": (
                "correct, jeżeli GPT poprawnie wydobył informację obecną w tekście; "
                "incorrect, jeżeli informacja jest błędna, nadmiarowa albo brakująca; "
                "not_applicable, jeżeli w tekście nie ma takiej informacji i GPT "
                "również jej nie podał"
            ),
        }
        for field in TARGET_FIELDS
    },
    "required": TARGET_FIELDS,
}

SYSTEM_PROMPT = """
Jesteś bezstronnym recenzentem danych historyczno-geograficznych SGKP.
Oceniasz wyłącznie zgodność danych GPT z podanym tekstem hasła. Ignorujesz
wszystkie pola ludzkiej weryfikacji i nie sugerujesz się nimi.
""".strip()

JUDGE_PROMPT = """
Zweryfikuj pola danych wydobytych przez GPT z hasła Słownika Geograficznego
Królestwa Polskiego.

Masz zwrócić JSON tylko z kluczami wymienionymi w sekcji POLA DO OCENY.

Dla każdego klucza zwróć jeden z trzech statusów:
- "correct": w tekście jest dana informacja i GPT poprawnie ją wydobył.
- "incorrect": GPT podał błędną informację, zmyślił informację, pomylił
  kategorię informacji albo pominął informację obecną w tekście.
- "not_applicable": w tekście nie ma danej informacji i GPT również jej nie
  podał. To nie jest pozytywne trafienie, tylko brak przedmiotu oceny.

Reguły oceny:
1. Oceniaj na podstawie pełnego tekstu hasła, nie na podstawie pól r_*.
2. Dopuszczaj drobne normalizacje fleksyjne, rozwinięcia skrótów i zmianę
   kolejności elementów listy, jeżeli sens i liczby są zgodne z tekstem.
3. Dla pól zagnieżdżonych sprawdź kompletność i przypisanie danych do
   właściwych części hasła, np. wieś, folwark, gmina.
4. Jeżeli GPT podał część poprawnych elementów, ale pominął istotny element
   albo dodał element nieobecny w tekście, oceń całe pole jako "incorrect".
5. Pole powiat_ocr odpowiada informacji o powiecie w tekście.
6. Pole właściciel dotyczy właściciela miejscowości, właściciela majątku albo posiadłości
   w XIX wieku; pomijaj wcześniejsze informacje historyczne. Pamiętaj też że chodzi o osoby właścicieli
   ewentualnie rodziny, firmy lub towarzystwa / instytucje, sformułowanie "należy do dóbr Byczynica" nie oznacza właściciela,
   lecz geograficzną, majątkową przynależność danego miejsca do jakiejś posiadłości.
7. Pole autor dotyczy wyłącznie podpisu autora na końcu hasła.
8. l_mk_statystyka dotyczy liczby mieszkańców, l_dm_statystyka liczby domów.
9. własność_ziemska dotyczy powierzchni i struktury gruntów.
10. ludność_wyznanie dotyczy liczby osób według wyznań.
11. Odsyłacz rozpoznawany jest zwykle po skrócie "ob."; jeżeli hasło jest tylko
    odsyłaczem, pozostałe pola powinny być puste, chyba że tekst podaje realne
    dane do oceny.
12. W przypadku pola Typ poprawną wartością może być wyrażenie złożone z paru słów, jeżeli w tekście jest np. wieś rządowa,
    a GPT rozpoznał "wieś" to nie jest w pełni poprawna wartość i należy ją oznaczyć jako False. Jeżeli w tekście jest wś i fol. a GPT
    zapisało tylko "wieś" to również nie jest pełna poprawna odpowiedź i należy oznaczyć ją jako False.

Zwróć wyłącznie JSON zgodny ze schematem. Nie dodawaj komentarzy.
""".strip()

EXTRACTION_REFERENCE = """
Kontekst pierwotnej ekstrakcji danych:

Model GPT miał wydobywać tylko informacje obecne w tekście hasła. Jeżeli tekst
nie zawierał danej informacji, pole miało zostać pominięte albo mieć wartość
null. Przy ocenie porównuj wynik GPT z tekstem źródłowym i z poniższymi
definicjami pól.

1. Dane podstawowe:
- typ: lista typów obiektu opisywanego przez hasło, np. "wieś", "folwark",
  "miasto", "rzeka", "jezioro", "góra", "osada". Jeżeli tekst zawiera skrót
  "ob." jako odsyłacz, zwykle poprawny typ to ["odsyłacz"], a pozostałe pola
  powinny być puste, chyba że hasło mimo odsyłacza podaje realne dane.
- warianty_nazw: lista obiektów {"lang": ..., "wariant_nazwy": ...}. Wariant
  musi różnić się od nazwy hasła. Jeżeli język nie jest podany, oczekiwane jest
  "nieokr.". Uwzględniaj aliasy, nazwy obcojęzyczne i dawne warianty podane
  zwykle na początku hasła. Nazwa po samym "ob." nie jest wariantem.
- powiat_ocr: nazwa powiatu z tekstu, zwykle po "pow.".
- gmina: nazwa gminy z tekstu, zwykle po "gm.".
- gubernia: nazwa guberni z tekstu, zwykle po "gub.".
- parafia_katolicka: nazwa parafii katolickiej/rzymskokatolickiej. Parafia
  bez określenia wyznania jest traktowana jako katolicka. Sam kościół bez
  informacji "par." albo "parafialny" nie wystarcza do ustalenia parafii.
- parafia_inna: lista obiektów {"wyznanie": ..., "nazwa_parafii": ...} dla
  parafii niekatolickich, np. prawosławnych, greckokatolickich, ewangelickich.
  Jeżeli tekst mówi o cerkwi/kościele parafialnym danego wyznania "w miejscu",
  parafią jest opisywana miejscowość.
- autor: inicjały albo nazwisko autora hasła znajdujące się na samym końcu
  tekstu hasła, np. "Br. Ch.", "F. S.", "Sulimierski". Ignoruj inicjały,
  nazwiska i osoby pojawiające się w środku tekstu, bo nie są autorami hasła.
  Jeżeli hasło nie kończy się podpisem autora, brak pola autor jest poprawnym
  brakiem przedmiotu oceny.

2. Właściciel:
- właściciel: właściciel miejscowości, majątku albo posiadłości w XIX wieku.
  Może to być osoba, rodzina, wielu właścicieli, skarb/rząd albo instytucja.
  Pomijaj informacje historyczne sprzed XIX wieku, np. średniowieczne lub
  XVI-wieczne nadania, jeżeli tekst zawiera późniejszą albo aktualną informację.

3. Statystyka:
- l_mk_statystyka: lista obiektów {"dotyczy": ..., "liczba": [...]}, gdzie
  "liczba" zawiera obiekty {"data": ..., "liczba": ...}. Dotyczy liczby
  mieszkańców oznaczanej w SGKP m.in. skrótem "mk.". Zachowuj rozróżnienie
  podmiotu danych, np. główna miejscowość, wieś, folwark, gmina. Dla danych
  bez roku, ale aktualnych dla hasła, akceptuj "obecnie".
- l_dm_statystyka: analogiczna lista dla liczby domów, zwykle skrót "dm.".
  Nie myl liczby domów z liczbą budynków innego typu, jeżeli tekst nie
  identyfikuje ich jako domów.

4. Własność ziemska:
- własność_ziemska: lista obiektów {"land_name": ..., "land": [...]}, gdzie
  "land" zawiera {"type_of_ground": ..., "area_of_ground": ...}. Pole dotyczy
  powierzchni i struktury gruntów: obszar ogółem, ziemia/rola orna, ogrody,
  łąki, pastwiska, lasy, nieużytki, wody itp. Jednostki mogą być zapisane jako
  ha, morgi/mr., dziesięciny/dzies. Jeżeli tekst osobno opisuje wieś, folwark
  lub kilka własności, powinny być osobne obiekty. Dopuszczaj normalizację nazw
  gruntów i separatorów dziesiętnych, jeżeli liczby i sens są zgodne.

5. Ludność według wyznań:
- ludność_wyznanie: lista obiektów {"dotyczy": ..., "struktura_wyznaniowa": [...]},
  gdzie struktura zawiera {"wyznanie_ocr": ..., "liczba": ...}. Oceniaj liczby
  osób według wyznań, np. rz.-kat., katol., gr.-kat., prawosł., ew., żyd.,
  izrael. Zachowuj oryginalny sens skrótu wyznania. Jeżeli dane dotyczą
  osobno wsi, folwarku, gminy lub parafii, wynik powinien to rozdzielać.

Ocena pola złożonego:
- Lista może być poprawna mimo innej kolejności elementów.
- Drobna różnica formy fleksyjnej, rozwinięcie skrótu albo ujednolicenie
  zapisu jest akceptowalne.
- Brak istotnego elementu, dodanie elementu nieobecnego w tekście, błędna
  liczba, błędny podmiot "dotyczy" albo pomylenie kategorii oznacza
  "incorrect".
""".strip()


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


def load_env_file(path: Path) -> None:
    if load_dotenv is not None:
        if path.exists():
            load_dotenv(dotenv_path=path)
        else:
            load_dotenv()
        return

    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def clean_record_for_judge(record: dict[str, Any]) -> dict[str, Any]:
    """Usuń pola oceny i zostaw tylko kontekst oraz dane GPT."""
    cleaned = {
        "nazwa": record.get("nazwa"),
        "text": record.get("text"),
        "tom": record.get("tom"),
        "strona": record.get("strona"),
        "ID": record.get("ID"),
    }
    for field in TARGET_FIELDS:
        if field in record:
            cleaned[field] = record[field]
    return cleaned


def make_response_schema(fields: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            field: RESPONSE_SCHEMA["properties"][field]
            for field in fields
        },
        "required": fields,
    }


def build_user_prompt(record: dict[str, Any], fields: list[str]) -> str:
    record_json = json.dumps(clean_record_for_judge(record), ensure_ascii=False, indent=2)
    fields_text = ", ".join(fields)
    return (
        f"{SYSTEM_PROMPT}\n\n{JUDGE_PROMPT}\n\n{EXTRACTION_REFERENCE}\n\n"
        f"POLA DO OCENY:\n{fields_text}\n\n"
        f"REKORD DO WERYFIKACJI:\n```json\n{record_json}\n```"
    )


def parse_response(text: str, fields: list[str]) -> dict[str, bool | None]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.removeprefix("```json").removeprefix("```").removesuffix("```")
        parsed = json.loads(stripped)

    if not isinstance(parsed, dict):
        raise ValueError("Odpowiedź modelu nie jest obiektem JSON.")

    result: dict[str, bool | None] = {}
    for field in fields:
        value = parsed.get(field)
        if value == STATUS_CORRECT:
            result[field] = True
        elif value == STATUS_INCORRECT:
            result[field] = False
        elif value == STATUS_NOT_APPLICABLE:
            result[field] = None
        else:
            raise ValueError(f"Brak poprawnego statusu pola {field!r}: {value!r}")
    return result


def verify_record(
    record: dict[str, Any],
    api_key: str,
    model: str,
    max_retries: int,
    retry_delay: float,
    request_timeout: float,
    fields: list[str],
) -> dict[str, bool | None]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Brak pakietu google-genai. Zainstaluj zależność, np. "
            "`pip install google-genai`, albo uruchom skrypt w środowisku "
            "projektu, w którym ten pakiet jest dostępny."
        ) from exc

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(request_timeout * 1000)),
    )
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=make_response_schema(fields),
    )
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=build_user_prompt(record, fields))])
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return parse_response(response.text or "", fields)
        except Exception as exc:  # noqa: BLE001 - zapisujemy kontekst błędu API/parsera
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_delay * (2**attempt))

    raise RuntimeError(f"Nie udało się zweryfikować rekordu: {last_error}") from last_error


def record_is_done(record: dict[str, Any], fields: list[str]) -> bool:
    if fields == TARGET_FIELDS and record.get(DONE_FIELD) is True:
        return True
    return all(f"gf_{field}" in record or f"r_{field}" not in record for field in fields)


def selected_indexes(
    data: list[dict[str, Any]],
    start: int,
    limit: int | None,
    force: bool,
    fields: list[str],
) -> list[int]:
    indexes = list(range(max(start, 0), len(data)))
    if limit is not None:
        indexes = indexes[:limit]
    if force:
        return indexes
    return [idx for idx in indexes if not record_is_done(data[idx], fields)]


def parse_fields(fields_arg: str) -> list[str]:
    if fields_arg.strip().lower() == "all":
        return TARGET_FIELDS
    fields = [field.strip() for field in fields_arg.split(",") if field.strip()]
    unknown = [field for field in fields if field not in TARGET_FIELDS]
    if unknown:
        raise ValueError(f"Nieznane pola do oceny: {', '.join(unknown)}")
    if not fields:
        raise ValueError("Lista pól do oceny jest pusta.")
    return fields


def process_records(args: argparse.Namespace) -> None:
    load_env_file(Path(args.env))
    if args.model == DEFAULT_MODEL and os.environ.get("GEMINI_MODEL"):
        args.model = os.environ["GEMINI_MODEL"]
    active_fields = parse_fields(args.fields)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Brak klucza API w zmiennej środowiskowej {args.api_key_env}.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists() and args.resume:
        data = load_json(output_path)
        print(f"Wznawiam pracę z pliku: {output_path}")
    else:
        data = load_json(input_path)
        print(f"Ładowanie danych wejściowych: {input_path}")

    indexes = selected_indexes(data, args.start, args.limit, args.force, active_fields)
    if not indexes:
        print("Brak rekordów do weryfikacji.")
        save_json_atomic(output_path, data)
        return

    print(
        f"Weryfikacja {len(indexes)} rekordów modelem {args.model} "
        f"w {args.workers} wątkach."
    )
    print(f"Pola do oceny: {', '.join(active_fields)}")

    completed = 0
    failures: list[tuple[int, str, str]] = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(
            verify_record,
            data[idx],
            api_key,
            args.model,
            args.max_retries,
            args.retry_delay,
            args.request_timeout,
            active_fields,
        ): idx
        for idx in indexes
    }

    aborted = False
    try:
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=args.stalled_timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                for future in pending:
                    idx = futures[future]
                    record_id = str(data[idx].get("ID", ""))
                    failures.append(
                        (
                            idx,
                            record_id,
                            f"Brak zakończonego zadania przez {args.stalled_timeout} s.",
                        )
                    )
                    future.cancel()
                save_json_atomic(output_path, data)
                executor.shutdown(wait=False, cancel_futures=True)
                aborted = True
                break

            for future in done:
                pending.discard(future)
                idx = futures[future]
                record_id = str(data[idx].get("ID", ""))
                try:
                    verdict = future.result()
                    for field, value in verdict.items():
                        gf_field = f"gf_{field}"
                        if value is None:
                            data[idx].pop(gf_field, None)
                        else:
                            data[idx][gf_field] = value
                    if active_fields == TARGET_FIELDS:
                        data[idx][DONE_FIELD] = True
                    completed += 1
                    print(f"[{completed}/{len(indexes)}] OK {record_id}")
                except Exception as exc:  # noqa: BLE001 - błąd pojedynczego rekordu nie kończy całości
                    failures.append((idx, record_id, str(exc)))
                    print(f"BŁĄD {record_id}: {exc}", file=sys.stderr)
                finally:
                    save_json_atomic(output_path, data)
    except KeyboardInterrupt:
        save_json_atomic(output_path, data)
        executor.shutdown(wait=False, cancel_futures=True)
        print(
            f"\nPrzerwano. Dotychczasowy wynik zapisano w: {output_path}",
            file=sys.stderr,
        )
        sys.exit(130)
    else:
        if not aborted:
            executor.shutdown(wait=True)

    if failures:
        print("\nRekordy z błędami:", file=sys.stderr)
        for idx, record_id, error in failures:
            print(f"- index={idx}, ID={record_id}: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Zapisano wyniki w: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weryfikacja dane_testowe.json przez Gemini jako LLM-as-a-Judge."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Plik wejściowy JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Plik wynikowy JSON.")
    parser.add_argument("--env", default=".env", help="Plik .env z kluczem Gemini.")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY", help="Nazwa zmiennej z kluczem API.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nazwa modelu Gemini.")
    parser.add_argument(
        "--fields",
        default="all",
        help="Pola do oceny: all albo lista po przecinku, np. autor lub typ,powiat_ocr.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Liczba równoległych wywołań API.")
    parser.add_argument("--start", type=int, default=0, help="Indeks pierwszego rekordu do oceny.")
    parser.add_argument("--limit", type=int, default=None, help="Maksymalna liczba rekordów do oceny.")
    parser.add_argument("--max-retries", type=int, default=3, help="Liczba ponowień po błędzie.")
    parser.add_argument("--retry-delay", type=float, default=3.0, help="Bazowe opóźnienie ponowień w sekundach.")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Limit czasu jednego żądania Gemini w sekundach.",
    )
    parser.add_argument(
        "--stalled-timeout",
        type=float,
        default=900.0,
        help="Limit czasu bez zakończonego rekordu; po nim skrypt zapisuje wynik i kończy pracę.",
    )
    parser.add_argument("--force", action="store_true", help="Przelicz także rekordy z istniejącymi polami gf_*.")
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Nie wznawiaj z istniejącego pliku wynikowego.",
    )
    parser.set_defaults(resume=True)
    return parser


if __name__ == "__main__":
    process_records(build_arg_parser().parse_args())
