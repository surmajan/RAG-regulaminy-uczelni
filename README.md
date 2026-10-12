# Asystent regulaminów uczelni (RAG)

System odpowiadający na pytania studentów o zasady studiowania na podstawie dokumentów PDF (regulaminów). Odpowiedzi generuje model językowy wyłącznie na podstawie znalezionych fragmentów, z numerami źródeł. Jeśli w dokumentach nie ma odpowiedzi, system o tym informuje. Jakość każdej odpowiedzi ocenia dodatkowo drugi model.

## Jak to działa

1. **Indeksowanie:** pliki PDF są dzielone na fragmenty (1100 znaków z nakładaniem 200), zamieniane na wektory modelem `paraphrase-multilingual-mpnet-base-v2` i zapisywane w bazie wektorowej Weaviate.
2. **Wyszukiwanie:** pytanie jest zamieniane na wektor, a baza zwraca 5 najbardziej podobnych fragmentów razem z metadanymi dokumentu.
3. **Odpowiedź:** model `gpt-oss-20b` (Groq API) odpowiada tylko na podstawie fragmentów i po każdym twierdzeniu podaje numer źródła, np. `[2]`.
4. **Ocena:** model `gpt-oss-120b` pełni rolę sędziego (LLM-as-a-judge) i ocenia odpowiedź w skali 0-3 pod kątem poprawności, zgodności ze źródłami, jakości źródeł i kompletności oraz wykrywa halucynacje.

Program wypisuje też ślad działania (*reasoning trace*): użytą strategię, wywołane narzędzia, identyfikatory fragmentów i źródła.

## Uruchomienie

```bash
pip install -r requirements.txt
```

1. Umieść pliki PDF z regulaminami w folderze `dokumenty/` (folder nie jest częścią repozytorium).
2. Ustaw klucz API do [Groq](https://console.groq.com):
   ```bash
   export GROQ_API_KEY="twój_klucz"
   ```
3. Uruchom:
   ```bash
   python rag.py
   ```
4. Wpisuj pytania w terminalu, a słowo `koniec` kończy program.

Przy pierwszym uruchomieniu pobierany jest model embeddingów i silnik Weaviate (wymaga internetu). Folder z dokumentami można zmienić zmienną `DOKUMENTY_DIR`.

## Technologie

Python, SentenceTransformers, Weaviate, LangChain (text splitters), PyMuPDF, Groq API
