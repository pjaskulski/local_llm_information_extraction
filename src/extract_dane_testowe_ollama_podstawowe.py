"""Etap 1: dane podstawowe z dane_testowe.json przez Ollama + Bielik."""
from __future__ import annotations

import argparse
import re
import time
from typing import List

from pydantic import BaseModel, Field

from ollama_extract_common import (
    DEFAULT_BIELIK_PATH,
    DEFAULT_SOURCE_PATH,
    add_common_args,
    finish_with_timer,
    run_extraction,
)
from pathlib import Path


DONE_FIELD = "_ollama_podstawowe_done"
REFERRAL_RE = re.compile(r"(^|[\s,;])ob\.?($|[\s,;])", re.IGNORECASE)


class NameVarModel(BaseModel):
    lang: str | None = Field(None, description="język wariantu nazwy, jeżeli podano np. niem., węg., jeżeli brak zapisz nieokr. - nieokreślony")
    wariant_nazwy: str | None = Field(None, description="wariant nazwy hasła (alias, nazwa w innym języku, nazwa występująca w dokumentach itp.)")


class ParafiaInnaModel(BaseModel):
    wyznanie: str | None = Field(None, description="Wyznanie parafii, np. praw., gr.-kat., ew.")
    nazwa_parafii: str | None = Field(None, description="Nazwa parafii.")


class DanePodstawoweModel(BaseModel):
    chain_of_thought: List[str] | None = Field(None,
                                               description="Kroki wyjaśniające prowadzące do ustalenia danych podstawowych dla hasła")
    typ: List[str] | None = Field(None,
                            description="Lista typów hasła - co hasło opisuje np. wieś, miasto, miasteczko, rzekę, górę, osiedle, krainę itp., dla hasła może występować więcej niż jeden typ")
    powiat: str | None = Field(None,
                               description="Nazwa powiatu w którym położona jest miejscowość")
    gmina: str | None = Field(None,
                              description="Nazwa gminy w której położona jest miejscowość")
    gubernia: str | None = Field(None,
                                 description="Nazwa guberni, do której należy miejscowość")
    parafia_katolicka: str | None = Field(None,
                                          description="Nazwa parafii katolickiej (rzymsko-katolickiej)")
    parafia_inna: List[ParafiaInnaModel] | None = Field(None,
                                     description="Lista parafii nie katolickich (np. prawosławnych, greko-katolickich, ewangelickich)")
    autor: str | None = Field(None,
                              description="Inicjały lub nazwisko autora hasła, występuje na końcu hasła, część haseł nie ma podanego autora.")
    warianty_nazw: List[NameVarModel] | None = Field(None,
                                                     description="Lista wariantów nazw (aliasów) dla hasła.")



# lista skrótów z SGKP
input_path = Path(__file__).resolve().parents[1] / 'dictionary' / 'prompt_sgkp_skroty.txt'
with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    skroty = [x.strip() for x in lines]

lista_skrotow = ', '.join(skroty)


SYSTEM_PROMPT = """
Jesteś asystentem historyka, specjalizującym się w badaniach historyczno - geograficznych,
ekspertem w analizie tekstów haseł Słownika Geograficznego Królestwa Polskiego (SGKP).
Wydobywasz tylko informacje jawnie obecne w tekście hasła. Nie używasz wiedzy zewnętrznej.
""".strip()

USER_PROMPT = """
Przeanalizuj poniższy tekst i wypełnij strukturę JSON zgodnie z podanymi polami i regułami.

    **KROKI POSTĘPOWANIA:**
    1.  **Przeanalizuj przekazany tekst hasła słownika SGKP:**
    2.  **Wypełnij pola:** Na podstawie swojej analizy, wypełnij pozostałe pola w strukturze JSON.

    **SZCZEGÓŁOWE REGUŁY EKSTRAKCJI:**

     **1. Typ Hasła (`typ`):**
    *   Określ, co hasło opisuje (np. wieś, miasto, folwark, rzeka, jezioro, góra), czasem może to być 
    więcej niż jedno określenie np. wieś, folwark.
    *   **Reguła specjalna:** Jeżeli tekst hasła zawiera skrót `ob.` (obacz) lub ' ob ', jest to odsyłacz
        (chyba, że skrót występuje w nawiasie wówczas nie oznacza to że hasło jest odsyłaczem). W takim przypadku wypełnij **tylko** pole `typ` wartością 'odsyłacz' (np. "typ": ["stacja pocztowa"]) i pozostaw resztę pól jako `null`. Pamiętaj wówczas, że nazwa po skrócie `ob.` to nazwa innego hasła, a nie wariant nazwy i należy ją pominąć.
        Wynik zapisz w formie listy np. ['wieś'] lub ['wieś', 'folwark'] w polu 'typ'.

    **2. Warianty Nazw (`warianty_nazw`):**
    *   Wyszukaj alternatywne lub obcojęzyczne nazwy hasła, podane zwykle na samym początku. Niekiedy warianty nazw podane są razem z datą, kiedy występowały. Wariant nazwy musi się różnić od nazwy hasła podanej na początku tekstu hasła. Czasem warianty haseł podane są po skrócie al. = alias.
    *   Zapisz język (np. `niem.`, `ros.`, `łac.`). Jeśli język nie jest podany, użyj wartości `nieokr.`.

    **3. Dane Administracyjne (`powiat`, `gmina`, `gubernia`):**
    *   Wyodrębnij te informacje z tekstu. Często występują po skrótach: `pow.`, `gm.`, `gub.`.
    *   Jeżeli hasło opisuje miasto powiatowe np. "Sławiska, m. pow.", albo "m. pow. gub. wołoskiej" to powiat jest właśnie opisywanym hasłem 'Sławiska'.

    **4. Parafie (`parafia_katolicka`, `parafia_inna`):**
    *   Postępuj według następującej logiki:
        *   **Krok 1:** Szukaj skrótu `par.` (parafia), uwaga mogą zdarzać się literówki np. `par,`
        *   **Krok 2:** Jeśli znajdziesz `par.` z określeniem wyznania: kat., katol., rz.-kat. lub bez określenia wyznania, przyjmij, że to `parafia_katolicka`.
        *   **Krok 3:** Jeśli znajdziesz `par.` z wyznaniem (np. `par. gr.-kat.`, `par. ewang.`), zapisz nazwę i wyznanie w polu `parafia_inna`.
        *   **Krok 4:** **JEŚLI NIE ZNAJDZIESZ SKRÓTU `par.`**, sprawdź, czy w tekście jest mowa o "kościele parafialnym" (kościół par.) lub "cerkwi parafialnej" (cerkiew par.). Jeśli tak, oznacza to, że parafia (odpowiednio katolicka lub inna) znajduje się w opisywanej miejscowości. Zapisz wówczas jako parafię nazwę miejscowości. Uwaga: Zwykła wzmianka o kościele lub cerkwi (bez słowa "parafialny", "par.") NIE JEST wystarczająca do ustalenia siedziby parafii.
        * Nazwę parafii katolickiej zapisz w polu 'parafia_katolicka', jeżeli w tekście znajdzie się parafia dla
        innego wyznania zapisz ją w polu 'parafia_inna' jako element listy w formie struktury np. [{{ "wyznanie": "nazwa wyznania", "nazwa_parafii": "nazwa miejscowości" }}] - zob. też przykłady niżej.

    **5. Autor (`autor`):**
    *   **Kluczowa reguła:** Autor to **TYLKO I WYŁĄCZNIE** inicjały lub nazwisko znajdujące się na **samym końcu** tekstu hasła (np. `Br. Ch.`, `F. S.`, `Sulimierski`).
    *   **ZIGNORUJ** wszelkie inicjały i nazwiska pojawiające się w środku tekstu, ponieważ dotyczą one postaci historycznych, a nie autorów hasła.
    * W przypadku haseł zbiorczych pole 'autor' wypełnij tylko raz dla hasła zbiorczego, a nie dla ostatniego pod-hasła z serii. Autor jest w takim przypadku wspólny dla hasła.

    **INFORMACJE POMOCNICZE:**
    *   W tekście mogą występować skróty. Oto lista najczęstszych: {lista_skrotow}.
    *   Uwzględniaj **TYLKO I WYŁĄCZNIE** dane pochodzące z dostarczonego tekstu hasła. Jeżeli informacja nie jest podana w tekście, nie wnioskuj na podstawie swojej wiedzy.
    *   Nazwy (miejscowości, gminy, powiaty, parafie) zapisuj w formie mianownika, inne jednostki np. hrabstwa, starostwa - pomiń.
    *   Jeżeli w tekście brak jakiejś informacji, pozostaw jej wartość jako `null`.

    ---
    **PRZYKŁAD:**

    **Hasło:** Bolkowce
    **Tekst hasła:** Bolkowce, niem. Bolkowitz, ros. Bolkovicje, mczko, pow. woliński, par. i poczta Więcko, par. gr.-kat. w miejscu, gm. Pastwiska w gub. lidzkiej. W 1800 r. był własnością Adama Lankckowskiego sędziego ziemskiego, ma 25 dm., 98 mk. Grunty orne, liczne sady, budynków z drewna 23, bud. mur. 2, na południu wsi staw rybny. Zabytkowy kościół z XVI w. św. Piotra i Pawła w centrum wsi. L. Doz.

    **Wynik w formie struktury JSON:**
    ```json
    {{
    "chain_of_thought": [
        "1. Analizuję warianty nazw: 'niem. Bolkowitz' i 'ros. Bolkovicje'.",
        "2. Identyfikuję typ miejscowości: 'mczko' to miasteczko, dodaję taką wartość do listy: ['miasteczki'] i zapisuję w polu 'typ'.",
        "3. Znajduję dane administracyjne: 'pow. woliński', 'gm. Pastwiska'. Gubernia: lidzka, w niej miejści się gmina więc także miejscowość.",
        "4. Analizuję parafie: 'par. Więcko' to parafia katolicka. 'par. gr.-kat. w miejscu' to parafia inna.",
        "5. Sprawdzam koniec tekstu w poszukiwaniu autora. Znajduję 'L. Doz.'."
    ],
    "warianty_nazw": [
        {{"lang": "niem.", "wariant_nazwy": "Bolkowitz"}},
        {{"lang": "ros.", "wariant_nazwy": "Bolkovicje"}}
    ],
    "typ": ["miasteczko"],
    "powiat": "woliński",
    "gmina": "Pastwiska",
    "gubernia": lidzka,
    "parafia_katolicka": "Więcko",
    "parafia_inna":
            [
                {{ "wyznanie": "gr.-kat.", "nazwa_parafii": "Bolkowce" }}
    ],
    "autor": "L. Doz."
    }}

Tekst:
{text}
""".strip()


def build_prompt(record: dict) -> str:
    return (
        USER_PROMPT
        .replace("{lista_skrotow}", lista_skrotow)
        .replace("{text}", str(record.get("text", "")))
    )


def strip_parentheses(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def is_referral_text(text: str) -> bool:
    text_without_parentheses = strip_parentheses(text)
    return bool(REFERRAL_RE.search(text_without_parentheses))


def enforce_referral_type(record: dict) -> dict:
    text = str(record.get("text", ""))
    if not is_referral_text(text):
        return record

    cleaned = dict(record)
    typ = cleaned.get("typ")
    if isinstance(typ, list):
        typ_values = [value for value in typ if isinstance(value, str) and value.strip()]
        if not any(value.strip().lower() == "odsyłacz" for value in typ_values):
            typ_values.insert(0, "odsyłacz")
        cleaned["typ"] = typ_values
    else:
        cleaned["typ"] = ["odsyłacz"]
    cleaned[DONE_FIELD] = True
    return cleaned


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etap 1: dane podstawowe przez Ollama/Bielik.")
    add_common_args(parser, DEFAULT_SOURCE_PATH, DEFAULT_BIELIK_PATH)
    return parser


if __name__ == "__main__":
    start_time = time.time()
    run_extraction(
        build_arg_parser().parse_args(),
        DanePodstawoweModel,
        SYSTEM_PROMPT,
        build_prompt,
        DONE_FIELD,
        "dane podstawowe",
        base_only=True,
        result_postprocessor=enforce_referral_type,
    )
    finish_with_timer(start_time)
