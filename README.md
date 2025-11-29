# 🫀 Analiza HLS-CMDS
**Temat:** Klasyfikacja schorzeń serca oraz płuc na podstawie nagrań dźwiękowych ze stetoskopu.

---

## ℹ️ Informacje o projekcie
| Kategoria | Opis |
| :--- | :--- |
| **Wejście** | Pliki `.wav` |
| **Źródło** | [UCI Machine Learning Repository: HLS-CMDS](https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope) |
| **Cel** | Budowa modelu klasyfikacyjnego (CNN i inne) |
| **Deadline Fazy 1** | 📅 Połowa Grudnia |

---

## 🗓️ Faza 1: Analiza i Przetwarzanie (Tasks 1-4)
*Cel: Przygotowanie danych, EDA i preprocessing.*

### ✅ Zadanie 1: Szczegółowa analiza eksploracyjna (EDA)
- [ ] **Ilość nagrań:** Sprawdzenie liczebności zbioru.
- [ ] **Analiza treści:** Co przedstawiają nagrania (widoki/punkty osłuchiwania).
- [ ] **Statystyka klas:** Balans klas (zdrowy vs patologie).
- [ ] **Podział danych:** Wyodrębnienie zbioru uczącego i testowego.
- [ ] **Augmentacja:** Analiza okien czasowych w celu zwiększenia liczby próbek.

### ✅ Zadanie 2: Research
- [ ] Sprawdzenie dostępnych kodów/repozytoriów w internecie (GitHub/Kaggle).
- [ ] Analiza, do jakich celów dane były wcześniej wykorzystywane.

### ✅ Zadanie 3: Preprocessing sygnałów (DSP)
Zastosowanie i testowanie metod:
- [ ] **Odszumianie** (denoising).
- [ ] **Filtry:** Dolno- i górnoprzepustowe.
- [ ] **Analiza pasmowa.**
- [ ] **Analiza falkowa** (Wavelet Transform).
- [ ] **Analiza częstotliwościowa** (FFT).

### ✅ Zadanie 4: Wizualizacja
- [ ] Wizualizacja sygnałów surowych.
- [ ] Wizualizacja po analizach (spektrogramy, skalogramy).
- [ ] **Wnioski:** Co widać? Które cechy są kluczowe dla klasyfikacji?

---

## 🚀 Faza 2: Modelowanie (Tasks 5-7)
*Realizacja po podsumowaniu wyników Fazy 1.*

### ✅ Zadanie 5: Benchmark
- [ ] Opracowanie benchmarku klasyfikacyjnego przy użyciu **k-NN**.

### ✅ Zadanie 6: Trening modeli
Trening i test 3 różnych metod:
- [ ] **Model 1:** Sieć splotowa (CNN) - *wymagane*.
- [ ] **Model 2:** (np. SVM, Random Forest, LSTM).
- [ ] **Model 3:** (inna metoda).

### ✅ Zadanie 7: Finalizacja
- [ ] Raport końcowy (wspólny wysiłek).
- [ ] Prezentacja wyników.

---

## 👥 Podział zadań w zespole

| Osoba | Przypisane Zadania (ID) | Status |
| :--- | :--- | :--- |
| **[Osoba 1]** | ... | 🟡 W toku |
| **[Osoba 2]** | ... | 🔴 Do zrobienia |
| **[Osoba 3]** | ... | 🟢 Gotowe |

> **Notatka:** W połowie grudnia podsumowujemy wyniki Fazy 1 i ustalamy dokładny plan działania dla Fazy 2.
