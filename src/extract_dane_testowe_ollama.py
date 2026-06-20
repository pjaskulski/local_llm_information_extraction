"""Ekstrakcja danych testowych SGKP lokalnym modelem Ollama.

Domyślnie skrypt używa modelu Bielik:
SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "test" / "dane_testowe.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "test" / "dane_testowe_bielik.json"
DEFAULT_MODEL = "SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M"
DEFAULT_BASE_URL = "http://localhost:11434/v1"

BASE_FIELDS = ["nazwa", "text", "tom", "strona", "ID"]


class NameVarModel(BaseModel):
    lang: str | None = Field(
        None,
        description="Język wariantu nazwy, np. niem., ros., węg.; jeżeli brak, użyj nieokr.",
    )
    wariant_nazwy: str | None = Field(None, description="Wariant nazwy hasła.")


class ParafiaInnaModel(BaseModel):
    wyznanie: str | None = Field(None, description="Wyznanie parafii, np. praw., gr.-kat., ew.")
    nazwa_parafii: str | None = Field(None, description="Nazwa parafii.")


class LiczbaModel(BaseModel):
    data: str | None = Field(None, description="Rok albo obecnie.")
    liczba: str | None = Field(None, description="Liczba jako tekst.")


class StatystykaModel(BaseModel):
    dotyczy: str | None = Field(
        None,
        description="Czego dotyczą dane: główna miejscowość, wieś, folwark, gmina itd.",
    )
    liczba: list[LiczbaModel] | None = Field(None, description="Lista wartości liczbowych.")


class LandModel(BaseModel):
    type_of_ground: str | None = Field(
        None,
        description="Rodzaj gruntu, np. obszar ogółem, ziemia orna, łąki, lasy.",
    )
    area_of_ground: str | None = Field(None, description="Powierzchnia z jednostką.")


class LandOwnershipModel(BaseModel):
    land_name: str | None = Field(None, description="Czego dotyczy struktura gruntów.")
    land: list[LandModel] | None = Field(None, description="Lista rodzajów gruntów.")


class WyznanieModel(BaseModel):
    wyznanie_ocr: str | None = Field(None, description="Wyznanie w formie z tekstu.")
    liczba: str | None = Field(None, description="Liczba osób danego wyznania.")


class StrukturaWyznaniowaModel(BaseModel):
    dotyczy: str | None = Field(None, description="Czego dotyczą dane wyznaniowe.")
    struktura_wyznaniowa: list[WyznanieModel] | None = Field(
        None,
        description="Lista wyznań i liczebności.",
    )


class ExtractedEntryModel(BaseModel):
    typ: list[str] | None = Field(None, description="Lista typów hasła, np. wieś, folwark, wieś rządowa, miasto, kolonia.")
    warianty_nazw: list[NameVarModel] | None = Field(None, description="Warianty nazw.")
    powiat_ocr: str | None = Field(None, description="Powiat z tekstu.")
    gmina: str | None = Field(None, description="Gmina z tekstu.")
    gubernia: str | None = Field(None, description="Gubernia z tekstu.")
    parafia_katolicka: str | None = Field(None, description="Parafia katolicka.")
    parafia_inna: list[ParafiaInnaModel] | None = Field(None, description="Parafie niekatolickie.")
    właściciel: str | None = Field(None, description="Właściciel w XIX wieku.")
    autor: str | None = Field(None, description="Autor hasła z końca tekstu.")
    l_mk_statystyka: list[StatystykaModel] | None = Field(None, description="Liczba mieszkańców.")
    l_dm_statystyka: list[StatystykaModel] | None = Field(None, description="Liczba domów.")
    własność_ziemska: list[LandOwnershipModel] | None = Field(None, description="Struktura gruntów.")
    ludność_wyznanie: list[StrukturaWyznaniowaModel] | None = Field(
        None,
        description="Ludność według wyznań.",
    )


SYSTEM_PROMPT = """
Jesteś asystentem historyka specjalizującym się w analizie haseł Słownika
Geograficznego Królestwa Polskiego. Wydobywasz tylko informacje jawnie obecne
w podanym tekście hasła. Nie uzupełniasz danych z wiedzy zewnętrznej.
""".strip()

USER_PROMPT = """
Przeanalizuj hasło SGKP i wypełnij strukturę JSON.

Reguły ogólne:
- Zwróć wyłącznie dane obecne w tekście.
- Jeżeli informacji brak, zostaw pole jako null albo pustą listę.
- Dopuszczalne jest rozwijanie typowych skrótów, ale nie zgaduj.
- Nazwy miejscowości, powiatów, gmin, guberni i parafii zapisuj możliwie
  w mianowniku.
- Nie zostawiaj pustego wyniku, jeżeli w tekście są jawne skróty i dane:
  "wś" oznacza wieś, "folw." oznacza folwark, "st. p." oznacza stację
  pocztową, "pow." oznacza powiat, "gm." oznacza gminę, "gub." oznacza
  gubernię, "par." oznacza parafię, "dm." oznacza domy, "mk." oznacza
  mieszkańców.

Pola:
1. typ: typ hasła, czasem więcej niż jeden, zapisany w formie listy typów obiektu, np. wieś, folwark, miasto, miasteczko, osada,
   rzeka, jezioro, góra, dobra, kolonia np. ["wieś", "folwark"]. Jeżeli tekst jest odsyłaczem - w krótkim tekście hasła występuje
   skrót "ob.", typ to zwykle ["odsyłacz"].
2. warianty_nazw: aliasy, nazwy obcojęzyczne i dawne warianty nazw. Format:
   [{{"lang": "niem.", "wariant_nazwy": "..."}}]. Jeżeli język nie jest podany,
   użyj "nieokr.". Nazwa po samym "ob." nie jest wariantem.
3. powiat_ocr, gmina, gubernia: dane administracyjne z tekstu, zwykle po
   skrótach pow., gm., gub.
4. parafia_katolicka: parafia katolicka/rzymskokatolicka. Parafia bez podanego
   wyznania traktowana jest jako katolicka.
5. parafia_inna: parafie niekatolickie, np. prawosławne, greckokatolickie,
   ewangelickie. Format: [{{"wyznanie": "...", "nazwa_parafii": "..."}}].
6. właściciel: właściciel miejscowości, majątku lub posiadłości w XIX wieku.
   Pomiń wcześniejsze informacje historyczne, np. średniowieczne lub XVI-wieczne.
7. autor: tylko inicjały albo nazwisko autora na samym końcu hasła. Ignoruj
   osoby wymienione w treści.
8. l_mk_statystyka: liczby mieszkańców, zwykle skrót mk. Format listy:
   [{{"dotyczy": "główna miejscowość", "liczba": [{{"data": "obecnie", "liczba": "..."}}]}}].
   Rozdziel dane dla wsi, folwarku, gminy itp.
9. l_dm_statystyka: liczby domów, zwykle skrót dm., w takim samym formacie.
10. własność_ziemska: powierzchnia i struktura gruntów. Format:
    [{{"land_name": "...", "land": [{{"type_of_ground": "...", "area_of_ground": "..."}}]}}].
    Uwzględniaj obszar ogółem, rolę/ziemię orną, łąki, ogrody, pastwiska,
    lasy, nieużytki, wody itp.
11. ludność_wyznanie: liczby osób według wyznań. Format:
    [{{"dotyczy": "...", "struktura_wyznaniowa": [{{"wyznanie_ocr": "...", "liczba": "..."}}]}}].

Przykłady:
- Tekst: "1.) st. p., pow. ardatowski; gub. symbirska."
  Wynik powinien zawierać: typ=["stacja pocztowa"], powiat_ocr="ardatowski",
  gubernia="symbirska".
- Tekst: "Bieńki, niem. Bienken, wś i młyn, pow. ządzborski."
  Wynik powinien zawierać: typ=["wieś", "młyn"], powiat_ocr="ządzborski",
  warianty_nazw=[{{"lang": "niem.", "wariant_nazwy": "Bienken"}}].
- Tekst: "W 1827 r. było tu 26 dm., 153 mk.; teraz 34 dm., 272 mk."
  Wynik powinien zawierać osobno l_dm_statystyka i l_mk_statystyka z datami
  "1827" oraz "obecnie" np.

  "l_mk_statystyka": [
            {
                "dotyczy": "główna miejscowość",
                "liczba": [
                    {
                        "data": "1827",
                        "liczba": "153"
                    },
                    {
                        "data": "obecnie",
                        "liczba": "272"
                    }
                ]
            }
        ],

Hasło:
Nazwa: {nazwa}
Tom: {tom}
Strona: {strona}
ID: {entry_id}
Tekst:
{text}
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


def base_record(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in BASE_FIELDS}


def value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return True


def merge_result(record: dict[str, Any], extracted: ExtractedEntryModel) -> dict[str, Any]:
    result = base_record(record)
    dumped = extracted.model_dump(mode="json")
    for field, value in dumped.items():
        if value_present(value):
            result[field] = value
    result["_ollama_done"] = True
    return result


def record_is_done(record: dict[str, Any]) -> bool:
    return record.get("_ollama_done") is True


def build_prompt(record: dict[str, Any]) -> str:
    replacements = {
        "{nazwa}": str(record.get("nazwa", "")),
        "{tom}": str(record.get("tom", "")),
        "{strona}": str(record.get("strona", "")),
        "{entry_id}": str(record.get("ID", "")),
        "{text}": str(record.get("text", "")),
    }
    prompt = USER_PROMPT
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)
    return prompt.replace("{{", "{").replace("}}", "}")


def create_client(model: str, base_url: str):
    try:
        import instructor
    except ImportError as exc:
        raise RuntimeError(
            "Brak biblioteki instructor. Zainstaluj zależności: "
            "`pip install -r requirements.txt`."
        ) from exc

    return instructor.from_provider(
        f"ollama/{model}",
        mode=instructor.Mode.JSON,
        base_url=base_url,
    )


def extract_record(
    record: dict[str, Any],
    model: str,
    base_url: str,
    max_retries: int,
    timeout: float,
) -> ExtractedEntryModel:
    client = create_client(model, base_url)
    return client.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(record)},
        ],
        response_model=ExtractedEntryModel,
        max_retries=max_retries,
        timeout=timeout,
    )


def selected_indexes(
    data: list[dict[str, Any]],
    start: int,
    limit: int | None,
    force: bool,
) -> list[int]:
    indexes = list(range(max(start, 0), len(data)))
    if limit is not None:
        indexes = indexes[:limit]
    if force:
        return indexes
    return [idx for idx in indexes if not record_is_done(data[idx])]


def process(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists() and args.resume:
        data = load_json(output_path)
        print(f"Wznawiam z pliku: {output_path}")
    else:
        source = load_json(input_path)
        data = [base_record(record) for record in source]
        print(f"Ładowanie danych wejściowych: {input_path}")

    indexes = selected_indexes(data, args.start, args.limit, args.force)
    if not indexes:
        print("Brak rekordów do przetworzenia.")
        save_json_atomic(output_path, data)
        return

    print(f"Model Ollama: {args.model}")
    print(f"Endpoint Ollama: {args.base_url}")
    print(f"Rekordy do przetworzenia: {len(indexes)}")
    print(f"Wątki: {args.workers}")

    completed = 0
    failures: list[tuple[int, str, str]] = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(
            extract_record,
            data[idx],
            args.model,
            args.base_url,
            args.max_retries,
            args.timeout,
        ): idx
        for idx in indexes
    }

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
                break

            for future in done:
                idx = futures[future]
                record_id = str(data[idx].get("ID", ""))
                try:
                    extracted = future.result()
                    data[idx] = merge_result(data[idx], extracted)
                    completed += 1
                    print(f"[{completed}/{len(indexes)}] OK {record_id}")
                except Exception as exc:  # noqa: BLE001 - zapisujemy i idziemy dalej
                    failures.append((idx, record_id, str(exc)))
                    print(f"BŁĄD {record_id}: {exc}", file=sys.stderr)
                finally:
                    save_json_atomic(output_path, data)
    except KeyboardInterrupt:
        save_json_atomic(output_path, data)
        executor.shutdown(wait=False, cancel_futures=True)
        print(f"\nPrzerwano. Wynik częściowy zapisano w: {output_path}", file=sys.stderr)
        sys.exit(130)
    else:
        executor.shutdown(wait=True)

    if failures:
        print("\nRekordy z błędami:", file=sys.stderr)
        for idx, record_id, error in failures:
            print(f"- index={idx}, ID={record_id}: {error}", file=sys.stderr)
        sys.exit(1)

    print(f"Zapisano wyniki w: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ekstrakcja danych z dane_testowe.json lokalnym modelem Ollama."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Plik wejściowy JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Plik wynikowy JSON.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nazwa modelu Ollama.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Endpoint OpenAI-compatible Ollama.")
    parser.add_argument("--workers", type=int, default=1, help="Liczba równoległych wywołań.")
    parser.add_argument("--start", type=int, default=0, help="Indeks pierwszego rekordu.")
    parser.add_argument("--limit", type=int, default=None, help="Maksymalna liczba rekordów.")
    parser.add_argument("--max-retries", type=int, default=2, help="Liczba ponowień Instructor.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Timeout jednego rekordu w sekundach.")
    parser.add_argument(
        "--stalled-timeout",
        type=float,
        default=900.0,
        help="Limit czasu bez ukończonego rekordu.",
    )
    parser.add_argument("--force", action="store_true", help="Przelicz także rekordy już oznaczone jako gotowe.")
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Nie wznawiaj z istniejącego pliku wynikowego.",
    )
    parser.set_defaults(resume=True)
    return parser


if __name__ == "__main__":
    start_time = time.time()
    process(build_arg_parser().parse_args())
    elapsed = time.time() - start_time
    print(f"Czas wykonania: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
