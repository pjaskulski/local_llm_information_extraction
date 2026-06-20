"""Etap 3: właściciel i własność ziemska przez Ollama + Bielik."""
from __future__ import annotations

import argparse
import time

from pydantic import BaseModel, Field

from ollama_extract_common import (
    DEFAULT_BIELIK_PATH,
    add_common_args,
    finish_with_timer,
    run_extraction,
)
from typing import List
from pathlib import Path


DONE_FIELD = "_ollama_wlasnosc_done"


class LandModel(BaseModel):
    type_of_ground: str | None = Field(None, description="Rodzaj gruntu, np. obszar ogółem, ziemia orna, łąki, lasy, nieużytki itp..")
    area_of_ground: str | None = Field(None, description="Powierzchnia gruntu z jednostką.")


class LandOwnershipModel(BaseModel):
    land_name: str | None = Field(None, description="Czego dotyczą informacje o własności ziemskiej: główna miejscowość, wieś, folwark - gdy w tekście haśle opisane są osobno różne części własności")
    land: list[LandModel] | None = Field(None, description="Lista rodzajów gruntów.")


class WlasnoscModel(BaseModel):
    chain_of_thought: List[str] = Field(
        [],
        description="Kroki wyjaśniające prowadzące do ustalenia struktury własności ziemskiej w danej miejscowości (haśle) lub w elemencie danego hasła jeżeli opisano osobno własność ziemską i strukturę gruntów dla wsi, folwarku itp."
        )
    właściciel: str | None = Field(None, description="Właściciel miejscowości/majątku (aktualny w czasie powstawania Słownika) - pomiń informacje historyczne sprzed XIX wieku")
    własność_ziemska: list[LandOwnershipModel] | None = Field(None, description="Lista własności ziemskich ze strukturą gruntów, występujących w analizowanym haśle")


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
    Uwzględniaj **TYLKO** dane dotyczące XIX wieku, pomiń starsze informacje historyczne.

    **Własność ziemska (pole 'własność_ziemska')**
    *  wyszukaj informacje o własności ziemskiej, w znaczeniu gruntów, które znajdują się w opisywanej miejscowości. Uwzględnij ogólną powierzchnię gruntów, oraz (jeżeli podano) dane szczegółowe np. ziemia orna, ogrody, łąki, lasy, pastwiska, nieużytki, zarośla itp. Powierzchnia gruntu może być podana w hektarach (ha) morgach (mr), arach (ar) lub innych jednostkach. Niekiedy podana jets tylko ogólna powierzchnia bez wyszczególnienia rodzaju gruntu, wówczas zapisz ta informację z etykieta 'obszar ogółem'. W treści haseł mogą znajdować się inne informacje np. o liczbie mieszkańców (mk) lub liczbie domów (dm), te informacje pomiń, ważna jest tylko struktura własności ziemskiej. W haśle mogą znajdować się informacje o kilku własnościach ziemskich, np. osobno o wsi i osobno o folwarku, zapisz je wówczas osobno, jako kolejne struktury na liście, w polu 'land_name' zapisz nazwę danej własności np. 'wieś Echtz', 'folwark Marienwalde'. Nie wpisuj imienia ani nazwiska właściciela w `type_of_ground` - sformułowanie 'Szlachcic Jerzmański ma 440 dz.' mówi o właścicielu, nie ma tu informacji  o typie gruntu.

    **Właściciel miejscowości/posiadłości (pole `właściciel`)**
    *   Zapisz właściciela/właścielkę lub jeżeli istnieje wielu - właścicieli miejscowości, majątku. Jeżeli podany jest właściciel tylko części gruntów również zapisz taką informację. Użyj tylko informacji wskazujących na posiadanie majątku w XIX wieku, wcześniejsze informacje historyczne zignoruj. Informacje bez daty, np. 'Rafał Mikołaj Sobiepolski ma 340 dz.', traktuj jako aktualne dla czasu powstawania Słownika. Właścicielem może być konkretna osoba, kilka osób, rodzina, fundusz lub firma, a także rząd lub skarb państwa. 
    Jeżeli tekst podaje osobę z czasownikiem „ma”, „posiada”, „należy do”, np. 'Szlachcic Jerzmański ma 440 dz.', zapisz osobę w polu `właściciel`. 

    **INFORMACJE POMOCNICZE:**
    *   W tekście mogą występować skróty. Oto lista najczęstszych: {lista_skrotow}.
    *   Uwzględniaj **TYLKO I WYŁĄCZNIE** dane pochodzące z dostarczonego tekstu hasła.
    *   Nazwy zapisuj w formie mianownika.
    *   Jeżeli w tekście brak jakiejś informacji, pomiń daną kategorię informacji w wynikowej strukturze.

    ---
    **PRZYKŁAD:**

    **Hasło:** Bolkowce
    **Tekst hasła:** Bolkowce, niem. Bolkowitz, ros. Bolkovicje, wś, pow. woliński, par. Więcko, par. gr.-kat. w miejscu, gm. Pastwiska w gub. lidzkiej, szkoła w Pustkowiu. W 1805 r. było własnością Adama Lankckowskiego sędziego ziemskiego, 120 dz. ma także podsędek Kumański. Ma 25 dm., 98 mk. Według szem. duch. z r. 1878 było w miejscu dusz rz.kat. 78, żyd. 20. Grunty orne, liczne sady: powierzchnia roli or. i ogrodów 516'27 ha., łąk 118'03, pastw. 119'41, boru 412.03, nieużytków 22'62, wody 0'61, razem 1188'97 ha. Budynków z drewna 23, bud. mur. 2, na południu wsi staw rybny oraz wiatrak i karczma. W pobliskiej dolinie mała huta szkła. Zabytkowy kościół z XVI w. św. Piotra i Pawła w centrum wsi. W 1860 r. Lanckowscy założyli tu mały przytułek dla włościan. L. Doz.

    **Wynik w formie struktury JSON:**
    ```json
    {{
    "chain_of_thought": [
         "Rozpoczynam analizę hasła 'Bolkowce' w celu ekstrakcji szczegółowych danych o strukturze własności ziemskiej. Celem jest wypełnienie zagnieżdżonej struktury 'własność_ziemska'.",
        "Przeszukuję tekst w poszukiwaniu słów kluczowych wskazujących na dane powierzchniowe, takich jak 'powierzchnia', 'rola', 'grunty', 'ha' (lub morgi).",
        "Identyfikuję kluczowe zdanie zawierające szczegółowy wykaz gruntów: 'powierzchnia roli or. i ogrodów 516'27 ha., łąk 118'03, pastw. 119'41, boru 412.03, nieużytków 22'62, wody 0'61, razem 1188'97 ha.'.",
        "Na podstawie kontekstu całego hasła, które opisuje 'Bolkowce' jako wieś ('wś'), ustalam, że te dane dotyczą głównej jednostki osadniczej. Tworzę główny obiekt w liście 'własność_ziemska' i ustawiam jego pole 'land_name' na 'wieś Bolkowce'.",
        "Inicjuję pustą listę 'land' wewnątrz tego obiektu, do której będę dodawał poszczególne typy gruntów.",
        "Przetwarzam pierwszy element z listy: 'roli or. i ogrodów 516'27 ha.'.",
        " - Ekstrahuję typ gruntu. Normalizuję formę gramatyczną ('roli') do mianownika, uzyskując 'rola orna i ogrody'.",
        " - Ekstrahuję powierzchnię. Identyfikuję liczbę '516'27' i jednostkę 'ha.'. Normalizuję format, zamieniając separator ' ' ' na przecinek, aby uzyskać '516,27 ha'.",
        " - Tworzę pierwszy obiekt {{'type_of_ground': 'rola orna i ogrody', 'area_of_ground': '516,27 ha'}} i dodaję go do listy 'land'.",
        "Przetwarzam kolejny element: 'łąk 118'03'. Normalizuję 'łąk' do 'łąki' oraz '118'03' do '118,03 ha'. Tworzę i dodaję odpowiedni obiekt.",
        "Przetwarzam 'pastw. 119'41'. Rozwijam skrót 'pastw.' do 'pastwiska'. Normalizuję '119'41' do '119,41 ha'. Tworzę i dodaję obiekt.",
        "Przetwarzam 'boru 412.03'. Normalizuję 'boru' do 'bór'. Dodaję semantyczne uzupełnienie '(las)', aby wynik był bardziej czytelny. Zauważam, że tutaj separatorem jest kropka, więc normalizuję '412.03' do '412,03 ha'. Tworzę i dodaję obiekt.",
        "Przetwarzam 'nieużytków 22'62'. Normalizuję 'nieużytków' do 'nieużytki'. Normalizuję '22'62' do '22,62 ha'. Tworzę i dodaję obiekt.",
        "Przetwarzam ostatni element: 'wody 0'61'. Normalizuję 'wody' do 'wody' i '0'61' do '0,61 ha'. Tworzę i dodaję obiekt.",
        "Zauważam na końcu fragment 'razem 1188'97 ha.'. Wprowadzam tą wartość jako 'obszar ogółem' i z wartością znormalizowaną 1188,97 ha.",
        "Weryfikuję, czy w tekście są inne dane o strukturze agrarnej. Nie znajduję.", 
        "Odnotowuję informacje o właścicielu ('była własnością Adama Lankckowskiego') która wskazuje na XIX wiek ('W 1805 r.') i zapisuję ją w polu 'właściciel' jako 'Adam Lankckowski'.",
        "Także kolejna informacja '120 dz. ma także podsędek Kumański' wskazuje na drugiego właściciela części ziem, dodaję więc do pola właściciel nazwisko 'Kumański' -> 'Adam Lankckowski, Kumański'",
        "Proces ekstrakcji danych o własności ziemskiej został zakończony. Lista 'land' zawiera 7 obiektów, co odpowiada liczbie typów gruntów oraz podsumowaniu ogółem, wymienionych w tekście źródłowym."
    ],
    "własność_ziemska": [
        {{
            "land_name": "wieś Bolkowce",
            "land": [
                {{
                    "type_of_ground": "rola orna i ogrody",
                    "area_of_ground": "516,27 ha"
                }},
                {{
                    "type_of_ground": "łąki",
                    "area_of_ground": "118,03 ha"
                }},
                {{
                    "type_of_ground": "pastwiska",
                    "area_of_ground": "119,41 ha"
                }},
                {{
                    "type_of_ground": "bór (las)",
                    "area_of_ground": "412,03 ha"
                }},
                {{
                    "type_of_ground": "nieużytki",
                    "area_of_ground": "22,62 ha"
                }},
                {{
                    "type_of_ground": "wody",
                    "area_of_ground": "0,61 ha"
                }},
                {{
                    "type_of_ground": "obszar ogółem",
                    "area_of_ground": "1188,97 ha"
                }}
            ]
        }}
    ],
    "właściciel": "Adam Lankckowski, Kumański"
    }}```

---
    **PRZYKŁAD:**

    **Hasło:** Bałtycka Wola
    **Tekst hasła:** Bałtycka Wola, wś, pow. krzemiński, par. Ewino, gm. Pastwiska w gub. lidzkiej, szkoła w Pustkowiu. Ma 15 dm., 38 mk. Grunty orne: powierzchnia roli or. 500'30 ha., łąk 18'00, pastw. 109'80. Większość gruntu, 300 dz. ma rodzina Otockich. L. M.

    **Wynik w formie struktury JSON:**
    ```json
    {{
        "właściciel": "Otoccy",
        "własność_ziemska": [
        {{
            "land_name": "główna miejscowość",
            "land": [
            {{"type_of_ground": "rola orna", "area_of_ground": "500,30 ha."}},
            {{"type_of_ground": "łąki", "area_of_ground": "18,00 ha."}},
            {{"type_of_ground": "pastwiska", "area_of_ground": "109,80 ha."}}
            ]
        }}
        ]
    }}
    ```
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etap 3: właściciel i własność ziemska przez Ollama/Bielik.")
    add_common_args(parser, DEFAULT_BIELIK_PATH, DEFAULT_BIELIK_PATH)
    return parser


if __name__ == "__main__":
    start_time = time.time()
    run_extraction(
        build_arg_parser().parse_args(),
        WlasnoscModel,
        SYSTEM_PROMPT,
        build_prompt,
        DONE_FIELD,
        "właściciel i własność ziemska",
        base_only=False,
    )
    finish_with_timer(start_time)
