# 📊 Data Mining Project - Clasificare Text

Proiect de data mining pentru clasificarea articolelor din setul de date **20 Newsgroups** folosind multiple algoritmi de machine learning.

## 🎯 Obiectiv

Clasificarea documentelor text în categorii predefinite folosind 3 algoritmi diferiți:
- **Naive Bayes** (MultinomialNB)
- **Support Vector Machine** (SVM)
- **Random Forest**

## 📁 Structura Proiectului

```
data_mining_project/
├── data/
│   ├── raw/              # Date brute (20 Newsgroups complet)
│   └── processed/        # Date preprocesate și subset-uri selectate
├── scripts/
│   ├── download_20newsgroups.py  # Descărcare și export dataset
│   └── select_categories.py      # Selectare categorii pentru proiect
├── src/
│   ├── preprocessing.py          # Preprocesare text
│   ├── naive_bayes.py            # Algoritm Naive Bayes
│   ├── svm_classifier.py         # Algoritm SVM
│   ├── random_forest.py           # Algoritm Random Forest
│   └── evaluation.py             # Evaluare și comparație algoritmi
├── results/              # Rezultate, metrici, grafice
├── requirements.txt
└── README.md
```

## 🚀 Pași de Pornire

### 1. Creare Virtual Environment și Instalare Dependențe

**Opțiunea 1: Folosind scriptul de setup (recomandat)**
```bash
cd data_mining_project
chmod +x setup.sh
./setup.sh
```

**Opțiunea 2: Manual**
```bash
cd data_mining_project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Notă:** În viitor, când lucrezi la proiect, activează întotdeauna virtual environment-ul:
```bash
source venv/bin/activate
```

### 2. Descărcare Set de Date

```bash
python scripts/download_20newsgroups.py
```

Acest script va:
- Descărca setul de date 20 Newsgroups
- Exporta datele în `data/raw/20newsgroups_dataset.csv` și `.json`
- Genera statistici despre dataset

### 3. Selectare Categorii

Editează `scripts/select_categories.py` și modifică lista `selected_categories` cu categoriile dorite (5-6 categorii), apoi rulează:

```bash
python scripts/select_categories.py
```

**Categorii disponibile în 20 Newsgroups:**
- `alt.atheism`
- `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x`
- `misc.forsale`
- `rec.autos`, `rec.motorcycles`, `rec.sport.baseball`, `rec.sport.hockey`
- `sci.crypt`, `sci.electronics`, `sci.med`, `sci.space`
- `soc.religion.christian`, `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`, `talk.religion.misc`

**Exemplu de categorii diverse:**
```python
selected_categories = [
    'sci.space',           # Știință
    'rec.sport.hockey',    # Sport
    'comp.graphics',       # Tehnologie
    'talk.politics.mideast',  # Politică
    'rec.autos',           # Auto
    'sci.med'              # Medicină
]
```

### 4. Selectare Categorii și Rulare Proiect

**Selectează categoriile pentru proiect:**
```bash
python3 scripts/select_categories.py
```

Editează `scripts/select_categories.py` pentru a alege categoriile dorite (5-6 categorii).

**Rulează proiectul complet:**
```bash
python3 main.py
```

Acest script va:
- Preprocesa datele (tokenizare, stop words, stemming, vectorizare TF-IDF)
- Antrena cei 3 algoritmi (Naive Bayes, SVM, Random Forest)
- Evalua și compara rezultatele
- Genera grafice și rapoarte detaliate

**Rezultatele** vor fi salvate în `results/`:
- `algorithm_comparison.csv` - Tabel comparativ
- `algorithm_comparison.png` - Grafice comparație
- `detailed_results.json` - Metrici detaliate
- `confusion_matrices/` - Matrici de confuzie pentru fiecare algoritm

## 📊 Set de Date

**20 Newsgroups Dataset:**
- ~20,000 de documente
- 20 de categorii
- Text în engleză
- Format: text raw (fără headers/footers)

## 🔧 Tehnologii

- **Python 3.8+**
- **scikit-learn** - Machine learning
- **pandas** - Manipulare date
- **nltk** - Preprocesare text
- **matplotlib/seaborn** - Vizualizare

## 🖥️ Interfață Web (UI)

Proiectul include o interfață web simplă și frumoasă pentru clasificarea documentelor.

### Pași pentru UI:

1. **Antrenează și salvează modelele:**
```bash
python3 train_models.py
```

2. **Pornește serverul Flask:**
```bash
python3 app.py
```

3. **Accesează UI-ul în browser:**
```
http://localhost:5000
```

### Funcționalități UI:
- ✍️ Introducere text direct
- 📁 Upload fișiere (TXT, CSV, JSON)
- 📊 Rezultate pentru toți cei 3 algoritmi
- 📈 Comparație algoritmi cu metrici
- 🎨 Interfață modernă și responsive

## 📝 Note

- Toate datele sunt exportate local în `data/` pentru control complet
- Poți modifica categoriile selectate oricând
- Dataset-ul complet rămâne disponibil în `data/raw/` pentru experimente
- Modelele antrenate sunt salvate în `models/` pentru reuse

