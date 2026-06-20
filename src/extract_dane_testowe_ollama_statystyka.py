"""Etap 2: statystyka i ludność według wyznań przez Ollama + Bielik."""
from __future__ import annotations

import argparse
import re
import time
from typing import List

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ollama_extract_common import (
    DEFAULT_BIELIK_PATH,
    add_common_args,
    finish_with_timer,
    run_extraction,
)
from pathlib import Path


DONE_FIELD = "_ollama_statystyka_done"
DM_RE = re.compile(r"\b(dm|dom|domów|domy)\b", re.IGNORECASE)
MK_RE = re.compile(r"\b(mk|mieszkań|mieszkańc)", re.IGNORECASE)


class LiczbaModel(BaseModel):
    data: str | None = Field(None, description="Data dla której podano liczbę, lub sformułowanie 'obecnie'")
    liczba: str | None = Field(None, description="Liczba jako tekst")


class StatystykaModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dotyczy: str | None = Field(None, description="Czego dotyczą dane: główna miejscowość, wieś, folwark, gmina")
    liczba: list[LiczbaModel] | None = Field(
        None,
        validation_alias=AliasChoices("liczba", "liczba_mieszkańców", "liczba_domów"),
        serialization_alias="liczba",
        description="Lista dat i wartości liczbowych",
    )


class WyznanieModel(BaseModel):
    wyznanie_ocr: str | None = Field(None, description="Nazwa wyznania, w takiej formie jak wystąpiła w tekście")
    liczba: str | None = Field(None, description="Liczba osób danego wyznania")


class StrukturaWyznaniowaModel(BaseModel):
    dotyczy: str | None = Field(None, description="Czego dotyczą dane wyznaniowe: główna miejscowość, inne miejsce opisane w haśle np. folwark, gmina")
    struktura_wyznaniowa: list[WyznanieModel] | None = Field(None, description="Lista wyznań i liczebności")


class StatystykaWyznanieModel(BaseModel):
    chain_of_thought: List[str] | None = Field(None, description="Kroki wyjaśniające prowadzące do ustalenia poszukiwanych danych dla hasła")
    l_mk_statystyka: list[StatystykaModel] | None = Field(None, description="Dane o liczbie mieszkańców podane w tekście hasła")
    l_dm_statystyka: list[StatystykaModel] | None = Field(None, description="Dane o liczbie domów podane w tekście hasła")
    ludność_wyznanie: list[StrukturaWyznaniowaModel] | None = Field(None, description="Dane o liczbie wyznawców podane w tekście hasła")


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
Twoim zadaniem jest precyzyjna ekstrakcja danych z podanego hasła.
    Przeanalizuj poniższy tekst i wypełnij strukturę JSON zgodnie z podanymi polami i regułami.

    **KROKI POSTĘPOWANIA:**
    1.  **Przemyśl analizę:** W polu `chain_of_thought` zapisz swoje rozumowanie krok po kroku, jak doszedłeś do poszczególnych wartości.
    2.  **Wypełnij pola:** Na podstawie swojej analizy, wypełnij pozostałe pola w strukturze JSON.

    **SZCZEGÓŁOWE REGUŁY EKSTRAKCJI:**

    Wyszukaj i zapisz w polach struktury JSON poszukiwane informacje.
    Uwzględniaj **TYLKO** dane dotyczące XIX wieku, starsze informacje historyczne - pomiń.

    **Liczba mieszkańców (pole 'l_mk_statystyka')**
    *   wyszukaj dane na temat liczby mieszkańców w danej miejscowości. Często hasła zawieraja takie informacje także o dodatkowych osadach, folwarkach, a nie tylko dla głównej miejscowości. Ustal czy znalezione dane dotyczą głównej miejscowości np. wsi, czy też właśnie np. folwarku. Zapisuj je wówczas osobno w odrębnych strukturach, każda z nich powinna mieć pole 'dotyczy' z wartością wskazującą czy liczba mieszkańców dotyczy głównej miejscowości czy innej osady, miejsca opisanego w treści hasła. Podobnie jeżeli podano dane na temat liczby mieszkańców dla całej gminy - zapisz te informacje osobno. Jeżeli w tekście są dane z różnych lat XIX wieku zapisz je jako osobne informacje, podając rok dla którego podano te informacje, lub określenie "obecnie" jeżeli rok nie jest podany, ale z kontekstu wynika że chodzi o dane aktualne w momencie pisania Słownika. Informacje o roku i liczbie zapisuj w polu 'liczba', które powinno zawierać jedną lub więcej struktur z datą i liczbą np. {{"data":"1827", "liczba": "650" }}.
    Jeżeli w tekście hasła są informacje o duszach rewizyjnych pomiń je, rejestruj tylko informacje mówiące o mieszkańcach.

    **Liczba domów (pole 'l_dm_statystyka')**
    *    wyszukaj dane na temat liczby domów w danej miejscowości. Często hasła zawieraja takie informacje o dodatkowych osadach, folwarkach, ustal czy znalezione dane dotyczą głównej miejscowości np. wsi, czy też właśnie np. folwarku. Zapisuj je wówczas osobno w odrębnych strukturach, każda z nich powinna mieć pole 'dotyczy' z wartością wskazującą czy liczba domów dotyczy głównej miejscowości czy innej osady, miejsca opisanego w treści hasła. Podobnie jeżeli podano dane na temat liczby domów dla całej gminy - zapisz te informacje osobno. Jeżeli w tekście są dane z różnych lat XIX wieku zapisz je jako osobne informacje, podając rok dla którego podano te informacje, lub określenie "obecnie" jeżeli rok nie jest podany, ale z kontekstu wynika że chodzi o dane aktualne w momencie pisania Słownika. Informacje o roku i liczbie zapisuj w polu 'liczba', które powinno zawierać jedną lub więcej struktur z datą i liczbą np. {{"data": "1827", "liczba": "650" }}.

    **WAŻNY FORMAT DLA PÓL STATYSTYCZNYCH:**
    *   W obu polach statystycznych używaj dokładnie tej samej nazwy listy wartości: `liczba`.
    *   Nie używaj nazw `liczba_mieszkańców` ani `liczba_domów`.
    *   Do `l_mk_statystyka` wpisuj wyłącznie wartości oznaczające mieszkańców: `mk.`, `mieszkańców`, `ludność`.
    *   Do `l_dm_statystyka` wpisuj wyłącznie wartości oznaczające domy: `dm.`, `domów`.
    *   Jeżeli w tekście jest tylko `obecnie 61 dm.` bez liczby `mk.`, zapisz 61 tylko w `l_dm_statystyka`, nie w `l_mk_statystyka`.
    *   Poprawny format to:
        {{
          "l_mk_statystyka": [
            {{
              "dotyczy": "główna miejscowość",
              "liczba": [
                {{"data": "obecnie", "liczba": "98"}}
              ]
            }}
          ],
          "l_dm_statystyka": [
            {{
              "dotyczy": "główna miejscowość",
              "liczba": [
                {{"data": "obecnie", "liczba": "25"}}
              ]
            }}
          ]
        }}

    **Struktura wyznaniowa (pole 'struktura_wyznaniowa')**
    *   wyszukaj dane na temat struktury wyznaniowej ludności w analizowanym haśle (liczby osób dla poszczególnych wyznań religijnych)
    *   określenia dot. wyznania ludności mogą być zapisane skrótem np. rz.-kat., gr.-kat., izrael. żyd. itp.
    *   jeżeli w tekście opisywane dane wyznaniowe dotyczące dodatkowych miejscowości (mp. opisana jest osobno wieś i folwark), lub terenu całej gminy, parafii, powiatu (i jest to wskazane wprost w tekście) - zapisz te informacje w osobnych strukturach
    *   zapisz dane w strukturze JSON "ludność_wyznanie", która przechowuje listę struktur z polami "dotyczy" (czego dotyczy struktura np. główna miejscowość, wieś, folwark), oraz "struktura_wyznaniowa", to drugie pole jest listą wyznań i liczebności wyznawców zapisanych w formie struktury typu:  {{"wyznanie_ocr": "rz. kat.", "liczba": "29"}}.



    **INFORMACJE POMOCNICZE:**
    * W tekście Słownika stosowano skróty: dm. - oznacza dom, domów, mk. - oznacza mieszkańca, mieszkańców, uwzględnij tylko informacje podane w taki sposób.
    *   W tekście mogą występować też inne skróty. Oto lista najczęstszych: {lista_skrotow}.
    *   Uwzględniaj **TYLKO I WYŁĄCZNIE** dane pochodzące z dostarczonego tekstu hasła.
    *   Jeżeli w tekście brak jakiejś informacji, pomiń daną kategorię informacji w wynikowej strukturze.
    
---
    **PRZYKŁAD:**

    **Hasło:** Bolkowce
    **Tekst hasła:** Bolkowce, niem. Bolkowitz, ros. Bolkovicje, mczko, pow. woliński, par. Więcko, par. gr.-kat. w miejscu, gm. Pastwiska w gub. lidzkiej, szkoła w Pustkowiu. W 1800 r. było własnością Adama Lankckowskiego sędziego ziemskiego. Ma 25 dm., 98 mk. Według szem. duch. z r. 1878 było w miejscu dusz rz.kat. 78, praw. 20. Grunty orne, liczne sady, budynków z drewna 23, bud. mur. 2, na południu wsi staw rybny oraz wiatrak i karczma. W pobliskiej dolinie mała huta szkła. Zabytkowy kościół z XVI w. św. Piotra i Pawła w centrum wsi. W 1860 r. Lanckowscy założyli tu mały przytułek dla włościan. L. Doz.

    **Wynik w formie struktury JSON:**
    ```json
    {{
    "chain_of_thought": [
        "Rozpoczynam analizę hasła 'Bolkowce' w celu ekstrakcji danych statystycznych o liczbie domów (dm.) i mieszkańców (mk.) zgodnie z zadaną strukturą.",
        "Przeszukuję tekst w poszukiwaniu wzorców zawierających skróty 'dm.' lub 'mk.' oraz powiązanych z nimi dat.",
        "Identyfikuję jedną kluczową frazę zawierającą dane statystyczne: '... Ma 25 dm., 98 mk.' - bez podaje daty, co oznacza że chodzi o okres 'obecnie' - w momencie powstawania hasła.",
        "Analizuję kontekst tej frazy. Odnosi się ona do głównej miejscowości 'Bolkowce', opisanej w haśle jako miasteczko ('mczko'). Nie ma tu mowy o oddzielnym folwarku czy innej części osady, więc wszystkie dane przypisuję do jednego podmiotu, który oznaczę jako 'główna miejscowość'.",
        "dla pola 'l_mk_statystyka' tworzę obiekt z polem `dotyczy` ustawionym na 'główna miejscowość' i pustą listą 'liczba' na dane liczbowe.",
        "dla pola 'l_dm_statystyka' tworzę obiekt z polem `dotyczy` ustawionym na 'główna miejscowość' i pustą listą 'liczba' na dane liczbowe.",
        "Z frazy '... Ma 25 dm., ...' ustalam liczbę domów = '25', tworzę obiekt z wartościami `data='obecnie'` i `liczba='25'`, a następnie dodaję go do listy `liczba`.",
        "Z tej samej frazy '... 98 mk.' ekstrahuję liczbę mieszkańców '98' i tworzę obiekt z wartościami `data='obecnie'` i `liczba='98'`, a następnie dodaję go do listy `liczba`.",
        "Zauważam również informację 'budynków z drewna 23, bud. mur. 2'. Jest to informacja o liczbie budynków, ale przedstawiona w formie podziału na typy, a nie jako łączna liczba 'domów' (dm.). Chociaż suma (23+2=25) odpowiada liczbie 'dm.', jest to inna forma danych. Aby zachować precyzję, decyduję się nie włączać tej informacji do struktury `l_dm_statystyka`, która jest przeznaczona dla danych oznaczonych jako 'dm.'.",
        "Przeglądam pozostałą część tekstu i stwierdzam, że nie zawiera ona żadnych dodatkowych danych o liczbie domów ani mieszkańców. Proces ekstrakcji danych statystycznych dla tego hasła został zakończony."
    ],
    "l_mk_statystyka": [
        {{
            "dotyczy": "główna miejscowość",
            "liczba": [
                {{"data": "obecnie", "liczba": "98"}}
            ]
        }}
    ],
    "l_dm_statystyka": [
        {{
            "dotyczy": "główna miejscowość",
            "liczba": [
                {{"data": "obecnie", "liczba": "25"}}
            ]
        }}
    ],
    "ludność_wyznanie": [
        {{
            "dotyczy": "główna miejscowość",
            "struktura_wyznaniowa": [
                {{
                    "wyznanie_ocr": "rz. kat.",
                    "liczba": "78"
                }},
                {{
                    "wyznanie_ocr": "praw.",
                    "liczba": "20"
                }}
            ]
        }}
    ]
    }}```

---    
Tekst:
{text}
""".strip()


def build_prompt(record: dict) -> str:
    return (
        USER_PROMPT
        
        .replace("{lista_skrotow}", lista_skrotow)
        .replace("{text}", str(record.get("text", "")))
    )


def remove_bad_unit_values(items: list[dict] | None, forbidden_pattern: re.Pattern) -> list[dict] | None:
    if not isinstance(items, list):
        return items

    cleaned_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        numbers = item.get("liczba")
        if not isinstance(numbers, list):
            continue
        cleaned_numbers = []
        for number_item in numbers:
            if not isinstance(number_item, dict):
                continue
            value = str(number_item.get("liczba", ""))
            if forbidden_pattern.search(value):
                continue
            cleaned_numbers.append(number_item)
        if cleaned_numbers:
            new_item = dict(item)
            new_item["liczba"] = cleaned_numbers
            cleaned_items.append(new_item)
    return cleaned_items or None


def clean_statystyka_units(record: dict) -> dict:
    cleaned = dict(record)
    l_mk = remove_bad_unit_values(cleaned.get("l_mk_statystyka"), DM_RE)
    l_dm = remove_bad_unit_values(cleaned.get("l_dm_statystyka"), MK_RE)

    if l_mk:
        cleaned["l_mk_statystyka"] = l_mk
    else:
        cleaned.pop("l_mk_statystyka", None)

    if l_dm:
        cleaned["l_dm_statystyka"] = l_dm
    else:
        cleaned.pop("l_dm_statystyka", None)

    return cleaned


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etap 2: statystyka i wyznania przez Ollama/Bielik.")
    add_common_args(parser, DEFAULT_BIELIK_PATH, DEFAULT_BIELIK_PATH)
    return parser


if __name__ == "__main__":
    start_time = time.time()
    run_extraction(
        build_arg_parser().parse_args(),
        StatystykaWyznanieModel,
        SYSTEM_PROMPT,
        build_prompt,
        DONE_FIELD,
        "statystyka i ludność według wyznań",
        base_only=False,
        result_postprocessor=clean_statystyka_units,
    )
    finish_with_timer(start_time)
