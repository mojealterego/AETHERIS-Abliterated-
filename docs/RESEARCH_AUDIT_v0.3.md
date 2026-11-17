# AETHERIS Omega v0.3.0 - Audyt badawczy

Sprawdzenie wag: oryginalny model Qwen3-0.6B zawiera 311 tensorow, zmodyfikowany 310.
Rownowaznosc lm_head.weight i model.embed_tokens.weight potwierdzono dla wszystkich 151936 wierszy; konfiguracja tie_word_embeddings=true.
Kontrola numeryczna: 44 zmienione projekcje wyjsciowe; 254 chronione tensory bez zmian; 0 zmian niedozwolonych.

Ocena odmow przed i po: 0.0 / 0.0, przy zaledwie 2 probkach na klase. Skutecznosc abliteration NIE JEST potwierdzona.
Wykonano 4 pary odpowiedzi generowanych przez oryginalny i zmodyfikowany model Q4_K_M.
Prompt Qwen3 kodowany z enable_thinking=False; surowe wyniki w reports.
W odpowiedziach zauważono niedoskonalosci jezyka i powtorzenia; bez zewnetrznej oceny jakosci nie ma podstaw do zatwierdzenia.

Nowe moduly: evidence_gate, benchmark_runner, weight_integrity.
Brama audytowa wymaga ocenionych przez czlowieka prob target/control/safety. Nie wydaje zgody produkcyjnej.
System zewnetrzny musi egzekwowac zgode, wiek, ochrone danych, izolacje tenantow i uprawnienia.

Model 8B nie jest zbudowany. Android ARM64 nie jest skompilowany ani sprawdzony.
Ta wersja aktualizuje KOD I NARZEDZIA, nie zmienia ponownie wag modelu 0.6B.
