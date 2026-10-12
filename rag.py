import os
import json
import sys
import pymupdf
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import weaviate
import weaviate.classes.config as wc
from groq import Groq
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5

MODEL_ODPOWIEDZI = "openai/gpt-oss-20b"
MODEL_SEDZIA = "openai/gpt-oss-120b"

KATALOG_DOKUMENTOW = Path(os.environ.get("DOKUMENTY_DIR", "dokumenty"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.exit("Brak klucza API.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""])

embed_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
llm_client = Groq(api_key=GROQ_API_KEY)

client = weaviate.connect_to_embedded()

collection_name = "DokumentyUczelniane"
if client.collections.exists(collection_name):
    client.collections.delete(collection_name)

collection = client.collections.create(
    name=collection_name,
    properties=[
        wc.Property(name="chunk_id", data_type=wc.DataType.TEXT),
        wc.Property(name="nazwa_pliku", data_type=wc.DataType.TEXT),
        wc.Property(name="chunk_index", data_type=wc.DataType.INT),
        wc.Property(name="tekst", data_type=wc.DataType.TEXT),
    ]
)

pliki_pdf = sorted(KATALOG_DOKUMENTOW.glob("*.pdf"))
if not pliki_pdf:
    client.close()
    sys.exit(f"Nie znaleziono plików PDF w folderze: {KATALOG_DOKUMENTOW.resolve()}")

for plik in pliki_pdf:
    dokument = pymupdf.open(plik)
    text = "\n".join(page.get_text() for page in dokument)
    chunks = text_splitter.split_text(text)
    wektory = embed_model.encode(chunks).tolist()

    with collection.batch.fixed_size(batch_size=200) as batch:
        for i, (chunk, wektor) in enumerate(zip(chunks, wektory)):
            chunk_id = f"{plik.name}_chunk_{i}"
            batch.add_object(
                properties={
                    "chunk_id": chunk_id,
                    "nazwa_pliku": plik.name,
                    "chunk_index": i,
                    "tekst": chunk
                },
                vector=wektor,
                uuid=generate_uuid5(chunk_id)
            )

    print(f"Zwektoryzowano {len(chunks)} fragmentów z pliku {plik.name}")


def search_vector(pytanie):
    wektor_pytania = embed_model.encode(pytanie).tolist()
    wyniki = collection.query.near_vector(near_vector=wektor_pytania, limit=5)
    return wyniki.objects


def get_document(document_id):
    wyniki = collection.query.fetch_objects(
        filters=Filter.by_property("nazwa_pliku").equal(document_id)
    )
    return {
        "nazwa_pliku": document_id,
        "liczba_fragmentow": len(wyniki.objects)
    }


BRAK_INFORMACJI = "W dokumentach nie ma wystarczających informacji, aby odpowiedzieć na to pytanie."

INSTRUKCJA = f"""Odpowiadasz na pytania studentów o zasady studiowania.
Korzystaj tylko z podanych fragmentów.

Fragmenty są oznaczone nagłówkami "=== FRAGMENT n ===".
Po każdym twierdzeniu podaj numer fragmentu WYŁĄCZNIE w formacie [n], np. [1] albo [2].
NIE pisz "[FRAGMENT 1]" ani "[FRAGMENT n]" - w nawiasie ma być sam numer, nic więcej.
Powołuj się wyłącznie na te numery. Numery paragrafów, ustępów i punktów
występujące w treści przepisów (np. "§ 21", "ust. 6", "10)")
NIE są numerami fragmentów.

Każde zdanie odpowiedzi musi zawierać konkretną treść merytoryczną - sam numer fragmentu
w nawiasach kwadratowych (np. samo "[2]") NIGDY nie jest wystarczającą odpowiedzią.

Jeśli fragmenty nie zawierają odpowiedzi, napisz dokładnie:
"{BRAK_INFORMACJI}"
"""

JUDGE_SYSTEM_PROMPT = """Jesteś modelem oceniającym odpowiedzi systemu RAG. Oceń odpowiedź wyłącznie na podstawie podanych fragmentów źródłowych. Nie korzystaj z wiedzy ogólnej. Sprawdź, czy odpowiedź: jest zgodna ze źródłami, nie zawiera halucynacji, odpowiada na pytanie użytkownika, zawiera właściwe źródła oraz poprawnie odmawia odpowiedzi, jeśli w dokumentach brakuje informacji. Zwróć wynik jako JSON z polami: correctness, faithfulness, source_quality, answer_completeness, hallucination_detected, comment. Przykład:
{
  "correctness": 3,
  "faithfulness": 3,
  "source_quality": 2,
  "answer_completeness": 2,
  "hallucination_detected": false,
  "comment": "Odpowiedź jest zgodna z dostarczonymi fragmentami, ale mogłaby dokładniej wskazać konkretny paragraf."
}

Proponowana skala:
0 - odpowiedź błędna lub nieoparta na źródłach
1 - odpowiedź częściowo poprawna
2 - odpowiedź w większości poprawna
3 - odpowiedź poprawna i dobrze uzasadniona źródłami"""

try:
    while True:
        pytanie = input("\nTwoje pytanie (wpisz 'koniec', aby wyjść): ").strip()
        if pytanie.lower() == "koniec":
            break
        if not pytanie:
            print("Pytanie nie może być puste.")
            continue

        wstepne_wyniki = search_vector(pytanie)

        kontekst = ""
        zrodla = set()
        znalezione_fragmenty_id = []

        for numer, obiekt in enumerate(wstepne_wyniki, start=1):
            wlasciwosci = obiekt.properties
            nazwa_pliku = wlasciwosci["nazwa_pliku"]
            tekst_fragmentu = wlasciwosci["tekst"]

            zrodla.add(nazwa_pliku)
            znalezione_fragmenty_id.append(wlasciwosci["chunk_id"])
            doc_info = get_document(nazwa_pliku)

            kontekst += f"=== FRAGMENT {numer} ===\n"
            kontekst += f"Metadane: Plik '{nazwa_pliku}' (całkowita wielkość: {doc_info['liczba_fragmentow']} fragmentów)\n"
            kontekst += f"Treść:\n{tekst_fragmentu}\n\n"

        print("\nWybrane fragmenty:\n" + kontekst)

        odpowiedz = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": INSTRUKCJA},
                {"role": "user", "content": f"Fragmenty:\n\n{kontekst}\nPytanie: {pytanie}"}
            ],
            model=MODEL_ODPOWIEDZI,
            temperature=0.1,
        )

        tresc_odpowiedzi = odpowiedz.choices[0].message.content

        trace = {
            "pytanie": pytanie,
            "strategia": "wyszukiwanie semantyczne (search_vector) w bazie wektorowej, wzbogacone o metadane dokumentu (get_document)",
            "wywolane_narzedzia": ["search_vector", "get_document"],
            "zapytania_do_retrievera": [pytanie],
            "znalezione_fragmenty_id": znalezione_fragmenty_id,
            "zrodla": sorted(zrodla),
            "oparta_na_zrodlach": BRAK_INFORMACJI not in tresc_odpowiedzi,
        }

        print("\nReasoning trace:", json.dumps(trace, ensure_ascii=False, indent=2))
        print("\nOdpowiedź:", tresc_odpowiedzi)
        print("Źródła:", ", ".join(sorted(zrodla)))

        judge_user_prompt = (
            f"Pytanie użytkownika:\n{pytanie}\n\n"
            f"Fragmenty źródłowe:\n\n{kontekst}\n\n"
            f"Odpowiedź systemu RAG:\n{tresc_odpowiedzi}\n\n"
            f"Reasoning trace:\n{json.dumps(trace, ensure_ascii=False)}"
        )

        score = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_user_prompt}
            ],
            model=MODEL_SEDZIA,
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        ocena = json.loads(score.choices[0].message.content)
        print("\nOcena odpowiedzi:")
        print(json.dumps(ocena, ensure_ascii=False, indent=2))
finally:
    client.close()
