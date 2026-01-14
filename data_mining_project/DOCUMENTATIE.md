# Documentatie Proiect Data Mining - Clasificare Text

## 1. Alegerea Problemei

### Problema aleasa: Clasificare Automata a Documentelor Text

Am ales sa rezolvam problema de **clasificare automata a documentelor text** in categorii predefinite. Aceasta este o problema fundamentala in data mining si machine learning, cu aplicatii practice importante.


### De ce aceasta problema?

1. **Relevanta**: Problema are aplicatii reale si utile
2. **Complexitate potrivita**: Ne permite sa implementam si sa comparam mai multi algoritmi
3. **Date disponibile**: Exista seturi de date standardizate pentru testare
4. **Evaluare clara**: Metrici de performanta bine definite (accuracy, precision, recall, F1-score)

### Obiectivele proiectului

- Implementarea a 3 algoritmi diferiti de clasificare text
- Compararea performantei algoritmilor folosind multiple metrici
- Analiza avantajelor si dezavantajelor fiecarui algoritm
- Formularea de concluzii despre ce algoritm functioneaza cel mai bine pentru aceasta problema

---

## 2. Alegerea Setului de Date

### Setul de date: 20 Newsgroups

Am ales setul de date **20 Newsgroups**, un dataset standard in machine learning pentru clasificare text.

### Caracteristici ale setului de date

- **Sursa**: Usenet newsgroups (forumuri de discutii)
- **Total categorii originale**: 20 categorii diverse
- **Categorii selectate pentru proiect**: 6 categorii
- **Total documente**: 5,701 documente
- **Distributie**: Echilibrata intre categorii

### Categoriile selectate

1. **comp.graphics** - Tehnologie - Grafica (955 documente)
2. **rec.autos** - Auto (937 documente)
3. **rec.sport.hockey** - Sport - Hockey (975 documente)
4. **sci.med** - Stiinta - Medicina (960 documente)
5. **sci.space** - Stiinta - Spatiu (955 documente)
6. **talk.politics.mideast** - Politica - Orientul Mijlociu (919 documente)

### De ce aceste categorii?

- **Diversitate**: Acoperim domenii diferite (tehnologie, sport, stiinta, politica)
- **Distinctie clara**: Categoriile sunt suficient de diferite pentru a permite clasificare corecta
- **Echilibru**: Distributia documentelor este relativ echilibrata
- **Dimensiune potrivita**: 6 categorii ofera complexitate suficienta fara a fi prea dificil

### Preprocesare date

- **Split train/test**: 80% antrenare (4,560 documente), 20% test (1,141 documente)
- **Vectorizare**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: 10,000 de caracteristici (cuvinte/cuvinte pereche)
- **Stemming**: Reducere cuvinte la radacina
- **Stop words**: Eliminare cuvinte comune (the, a, an, etc.)

---

## 3. Implementarea Algoritmilor de Data Mining

Am implementat **3 algoritmi diferiti** de clasificare text, fiecare cu abordari fundamentale diferite:

### 3.1. Naive Bayes

#### Ce este Naive Bayes?

Naive Bayes este un algoritm de clasificare probabilistica bazat pe teorema lui Bayes. Este unul dintre cei mai simpli si mai eficienti algoritmi pentru clasificare text.

#### Cum functioneaza?

**Teorema lui Bayes:**
```
P(clasa | document) = P(document | clasa) * P(clasa) / P(document)
```

**Presupunerea "Naive":**
Algoritmul presupune ca toate cuvintele sunt **independente** unele de altele. Desi aceasta presupunere nu este adevarata in realitate, simplifica mult calculele si functioneaza surprinzator de bine.

**Procesul:**
1. Calculeaza probabilitatea fiecarui cuvant pentru fiecare clasa
2. Calculeaza probabilitatea a priori a fiecarei clase
3. Pentru un document nou, inmulteste probabilitatile tuturor cuvintelor pentru fiecare clasa
4. Alege clasa cu cea mai mare probabilitate

**Laplace Smoothing:**
Pentru a evita problemele cand un cuvant nu apare in antrenare:
```
P(cuvant | clasa) = (count(cuvant, clasa) + alpha) / (count(clasa) + alpha * vocab_size)
```

#### Implementare

```python
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha)
```

**Parametrii:**
- `alpha=1.0`: Parametru de Laplace smoothing (valoare standard)

**Avantaje:**
- Foarte rapid la antrenare si predictie
- Functioneaza bine cu putine date
- Robust la noise
- Interpretabil

**Dezavantaje:**
- Presupunerea de independenta nu este realista
- Nu captureaza contextul sau ordinea cuvintelor

---

### 3.2. Support Vector Machine (SVM)

#### Ce este SVM?

Support Vector Machine este un algoritm puternic de machine learning care gaseste un hiperplan (limita de decizie) care separa cel mai bine clasele, maximizand marginea dintre ele.

#### Cum functioneaza?

**Conceptul de baza:**
- Gaseste un hiperplan care separa clasele
- Maximizeaza marginea (distanta) dintre cele mai apropiate puncte din clase diferite
- Punctele cele mai apropiate se numesc **support vectors**

**Kernel Trick:**
Pentru date care nu pot fi separate liniar, SVM foloseste un **kernel** care transforma datele intr-un spatiu dimensional mai mare unde devin separabile.

**Tipuri de kernel:**
- **Linear**: `K(x, y) = x · y` (folosit in proiect)
- **RBF**: Pentru date complexe neliniare
- **Polynomial**: Pentru relatii polinomiale

**Parametrul C (Regularizare):**
- **C mic**: Margine larga, toleranta la erori
- **C mare**: Margine ingusta, zero toleranta la erori

#### Implementare

```python
from sklearn.svm import SVC

class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, max_iter=1000):
        self.model = SVC(kernel=kernel, C=C, max_iter=max_iter, random_state=42)
```

**Parametrii:**
- `kernel='linear'`: Kernel liniar (optim pentru text sparse)
- `C=1.0`: Regularizare standard
- `max_iter=2000`: Numar maxim de iteratii pentru optimizare

**De ce kernel linear?**
- Datele TF-IDF sunt deja intr-un spatiu dimensional mare
- Kernel linear functioneaza excelent pentru date sparse
- Mult mai rapid decat RBF sau polynomial
- Rezultate comparabile sau mai bune

**Avantaje:**
- Performanta ridicata
- Eficient pe date sparse (TF-IDF)
- Robust la overfitting
- Memorie eficienta (foloseste doar support vectors)

**Dezavantaje:**
- Lent la antrenare (complexitate O(n²) sau O(n³))

---

### 3.3. Random Forest

#### Ce este Random Forest?

Random Forest este un algoritm de ensemble learning care combina predictiile a mai multor arbori de decizie pentru a obtine o performanta mai buna.

#### Cum functioneaza?

**Arbori de decizie:**
Un arbore de decizie ia decizii prin intrebari succesive:
- "Contine cuvantul 'hockey'?" → DA → "Contine 'puck'?" → DA → Clasa: Sport

**Bootstrap Aggregating (Bagging):**
1. Creeaza mai multi subset-uri aleatorii din date (cu inlocuire)
2. Antreneaza un arbore de decizie pe fiecare subset
3. Combina predictiile tuturor arborilor (vot majoritar)

**"Random" in Random Forest:**
- **Random sampling**: Fiecare arbore pe un subset aleator al datelor
- **Random features**: La fiecare nod, se alege un subset aleator de caracteristici

Aceasta randomizare reduce overfitting si creste robustetea.

#### Implementare

```python
from sklearn.ensemble import RandomForestClassifier

class RandomForestTextClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # paralelizare
        )
```

**Parametrii:**
- `n_estimators=100`: Numarul de arbori (compromis performanta/viteza)
- `max_depth=None`: Adancime maxima (None = pana cand toate frunzele sunt pure)
- `n_jobs=-1`: Foloseste toate core-urile CPU pentru antrenare paralela

**Avantaje:**
- Performanta buna
- Robust la overfitting (bagging si randomizare)
- Poate identifica feature importance
- Paralelizabil
- Nu necesita preprocesare complexa

**Dezavantaje:**
- Lent la antrenare (trebuie sa antreneze multi arbori)
- Ocupa multa memorie
- Interpretabilitate limitata
- Nu captureaza ordinea cuvintelor

---

## 4. Compararea Rezultatelor - Metrici de Performanta

Am evaluat cei 3 algoritmi folosind **multiple metrici de performanta**:

### 4.1. Metrici de Clasificare

#### Accuracy (Acuratete)
Proportia de predictii corecte din totalul de predictii.

#### Precision (Precizie)
Proportia de predictii corecte pozitive din totalul de predictii pozitive.
```
Precision = TP / (TP + FP)
```

#### Recall (Sensibilitate)
Proportia de cazuri pozitive corect identificate.
```
Recall = TP / (TP + FN)
```

#### F1-Score
Media armonica intre Precision si Recall.
```
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
```

#### Metrici Macro vs Weighted

- **Macro**: Media aritmetica a metricilor pentru fiecare clasa (trateaza toate clasele egal)
- **Weighted**: Media ponderata dupa numarul de exemple din fiecare clasa

### 4.2. Rezultate Obtinute

| Algoritm | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|----------|----------|-------------------|----------------|------------------|---------------------|------------------|---------------------|
| **Naive Bayes** | **0.9089** | **0.9112** | **0.9089** | **0.9083** | **0.9114** | **0.9089** | **0.9084** |
| **SVM** | 0.8966 | 0.8968 | 0.8964 | 0.8963 | 0.8968 | 0.8966 | 0.8965 |
| **Random Forest** | 0.8501 | 0.8557 | 0.8499 | 0.8497 | 0.8559 | 0.8501 | 0.8500 |

### 4.3. Metrici de Timp

| Algoritm | Timp Antrenare (s) | Timp Predictie (s) |
|----------|-------------------|-------------------|
| **Naive Bayes** | **0.0038** | **0.0012** |
| **SVM** | 7.5364 | 1.4539 |
| **Random Forest** | 0.5439 | 0.0288 |

### 4.4. Analiza Rezultatelor

#### Naive Bayes - Cel mai bun algoritm

**Performanta:**
- **Accuracy: 90.89%** - Cel mai bun
- **F1-Score: 90.83%** - Cel mai bun
- **Timp antrenare: 0.0038s** - Cel mai rapid (de ~2000x mai rapid decat SVM)
- **Timp predictie: 0.0012s** - Cel mai rapid (de ~1200x mai rapid decat SVM)

**De ce functioneaza atat de bine?**
- Presupunerea de independenta functioneaza surprinzator de bine pentru text
- TF-IDF vectorizarea creeaza date sparse pe care Naive Bayes le gestioneaza excelent
- Smoothing previne overfitting
- Algoritm optimizat pentru clasificare text

#### SVM - Performanta buna, viteza mai lenta

**Performanta:**
- **Accuracy: 89.66%** - Al doilea cel mai bun (doar 1.23% mai mic decat Naive Bayes)
- **F1-Score: 89.63%** - Al doilea cel mai bun
- **Timp antrenare: 7.54s** - Cel mai lent (dar acceptabil pentru dataset-ul nostru)
- **Timp predictie: 1.45s** - Mediu

**Observatii:**
- Performanta foarte apropiata de Naive Bayes
- Principalul dezavantaj este viteza de antrenare
- Pentru dataset-uri mai mari, timpul de antrenare ar putea deveni problematic

#### Random Forest - Performanta mai scazuta

**Performanta:**
- **Accuracy: 85.01%** - Al treilea (5.88% mai mic decat Naive Bayes)
- **F1-Score: 84.97%** - Al treilea
- **Timp antrenare: 0.54s** - Mediu (rapid, dar mai lent decat Naive Bayes)
- **Timp predictie: 0.029s** - Rapid

**De ce performanta mai scazuta?**
- Random Forest functioneaza mai bine pe date dense
- Datele TF-IDF sunt foarte sparse (multe zerouri)
- Nu captureaza semantica sau ordinea cuvintelor
- Naive Bayes si SVM sunt optimizate specific pentru text sparse

**Avantaj:**
- Poate identifica feature importance (care cuvinte sunt importante)
- Util pentru interpretare si feature selection

### 4.5. Confusion Matrices

Confusion matrices pentru fiecare algoritm arata:
- **Naive Bayes**: Cel mai putine confuzii intre categorii
- **SVM**: Confuzii similare cu Naive Bayes
- **Random Forest**: Mai multe confuzii, in special intre categorii similare

### 4.6. Comparatie Vizuala

Graficele de comparatie arata clar:
- **Accuracy**: Naive Bayes > SVM > Random Forest
- **F1-Score**: Naive Bayes > SVM > Random Forest
- **Timp antrenare**: Naive Bayes << Random Forest << SVM
- **Timp predictie**: Naive Bayes << Random Forest << SVM

---

## 5. Concluzii

### 5.1. Rezumat Performanta

**Cel mai bun algoritm: Naive Bayes**

Naive Bayes a demonstrat cea mai buna performanta pe toate metricile:
- **Accuracy: 90.89%** (cel mai bun)
- **F1-Score: 90.83%** (cel mai bun)
- **Viteza antrenare: 0.0038s** (cel mai rapid, de ~2000x mai rapid decat SVM)
- **Viteza predictie: 0.0012s** (cel mai rapid, de ~1200x mai rapid decat SVM)

### 5.2. Analiza Algoritmilor

#### Naive Bayes - Alegerea optima pentru clasificare text

**De ce este cel mai bun?**
1. **Performanta superioara**: Accuracy si F1-Score cele mai bune
2. **Viteza exceptionala**: Antrenare si predictie extrem de rapide
3. **Eficienta memorie**: Ocupa putina memorie
4. **Robustete**: Functioneaza bine pe date sparse (TF-IDF)
5. **Simplitate**: Usor de inteles si implementat

**Cand sa il folosim?**
- Clasificare text (email spam, categorizare documente)
- Cand ai nevoie de viteza
- Cand ai date sparse (TF-IDF, bag of words)
- Cand vrei un model interpretabil

#### SVM - Performanta buna, viteza mai lenta

**Caracteristici:**
- Performanta foarte apropiata de Naive Bayes (doar 1.23% diferenta)
- Viteza de antrenare mult mai lenta
- Functioneaza bine pe date sparse

**Cand sa il folosim?**
- Cand ai nevoie de performanta maxima si poti astepta antrenarea
- Pentru dataset-uri de dimensiune medie
- Cand vrei un model robust

#### Random Forest - Performanta mai scazuta pentru text

**Caracteristici:**
- Performanta mai scazuta (5.88% mai mic decat Naive Bayes)
- Viteza acceptabila
- Poate identifica feature importance

**Cand sa il folosim?**
- Cand ai nevoie de feature importance
- Pentru date dense (nu sparse)
- Cand datele au interactiuni complexe

### 5.3. Testare pe Articole Reale - Rezultate Obtinute

Pentru a testa algoritmii pe scenarii reale, am creat 3 articole de test cu caracteristici diferite si am obtinut urmatoarele rezultate:

---

#### Articol 1: Categorie Clara (Hockey)

**Continut:** Articol despre un meci de hockey intre Montreal Canadiens si Toronto Maple Leafs, cu detalii specifice sportului.

| Algoritm | Categorie Prezisa | Incredere | Timp Predicție (ms) | Observatii |
|----------|-------------------|-----------|---------------------|------------|
| **Naive Bayes** | Sport - Hockey | **99.74%** | 7.862 | Cea mai buna incredere reala, viteza excelenta |
| **SVM** | Sport - Hockey | 100.00% | 25.386 | Predictie corecta, viteza medie |
| **Random Forest** | Sport - Hockey | 97.00% | 64.449 | Predictie corecta, probabilitati reale |

**Rezumat:** Toti cei 3 algoritmi au identificat corect categoria **Sport - Hockey**. Pentru articole cu categorie clara, algoritmii functioneaza excelent si sunt de acord.

---

#### Articol 2: Topics Mixte (Tehnologie Spatiala si Medicina)

**Continut:** Articol care discuta despre cum tehnologia spatiala contribuie la progrese medicale, mentionand si aplicatii in industria auto.

| Algoritm | Categorie Prezisa | Incredere | Timp Predicție (ms) | Observatii |
|----------|-------------------|-----------|---------------------|------------|
| **Naive Bayes** | Stiinta - Spatiu | **77.12%** | 1.969 | Identifica spatiu bazat pe combinatii clare de cuvinte |
| **SVM** | Stiinta - Spatiu | 100.00% | 4.624 | Identifica pattern-uri care indica spatiu ca fiind dominanta |
| **Random Forest** | Stiinta - Medicina | 53.00% | 42.783 | Identifica medicina ca tema centrala bazat pe context |

**Rezumat:** Articolul mentioneaza multiple categorii. Naive Bayes si SVM identifica spatiu (bazat pe pattern-uri clare), iar Random Forest identifica medicina (bazat pe tema centrala). Algoritmii dau rezultate diferite bazate pe abordarea lor.

---

#### Articol 3: Povestioara cu Toate Categoriile

**Continut:** Povestioara care mentioneaza in mod egal toate categoriile: tehnologie, medicina, sport, spatiu, auto, politica.

| Algoritm | Categorie Prezisa | Incredere | Timp Predicție (ms) | Observatii |
|----------|-------------------|-----------|---------------------|------------|
| **Naive Bayes** | Stiinta - Spatiu | **47.06%** | 3.437 | Se bazeaza pe claritatea combinatiilor de cuvinte |
| **SVM** | Stiinta - Medicina | 100.00% | 17.469 | Identifica medicina ca tema centrala a povestii |
| **Random Forest** | Stiinta - Medicina | 32.00% | 30.247 | Analizeaza contextul si identifica medicina ca tema centrala |

**Rezumat:** Articolul mentioneaza toate categoriile in mod egal. Naive Bayes identifica spatiu (combinatii clare), iar SVM si Random Forest identifica medicina (tema centrala). Diferentele apar din abordarile diferite ale algoritmilor.

---

### 5.4. Observatie Importanta: SVM si Increderea de 100%

**Observatie:** In toate testele, SVM returneaza mereu **100% incredere** pentru predictie.

**De ce se intampla asta?**

SVM (SVC din scikit-learn) cu kernel='linear' **nu calculeaza probabilitati reale** pentru predictii. In implementarea noastra:

1. **SVC nu are `predict_proba` activat implicit**: Pentru a calcula probabilitati, SVC necesita parametrul `probability=True`, care activeaza Platt scaling (o metoda de calibrare a probabilitatilor).

2. **In codul nostru**: Cand SVM nu are `predict_proba` disponibil, codul seteaza:
   ```python
   prob_dict = {prediction_name: 1.0}  # 100% incredere
   ```

3. **De ce nu am activat `probability=True`?**
   - Platt scaling adauga overhead la antrenare si predictie
   - Pentru kernel linear, probabilitatile calibreate pot fi mai precise, dar necesita timp suplimentar
   - In practica, pentru clasificare text, predictia categoriei este mai importanta decat probabilitatea exacta

**Ce inseamna asta?**

- **Increderea de 100% nu inseamna ca SVM este sigur 100%**: Este doar o limitare tehnica - SVM nu ofera probabilitati reale fara Platt scaling
- **Predictia categoriei este corecta**: Chiar daca increderea este setata la 100%, categoria prezisa este rezultatul algoritmului SVM
- **Comparatia cu alti algoritmi**: Nu putem compara direct "increderea" SVM cu cea a Naive Bayes sau Random Forest, care ofera probabilitati reale

**Diferenta intre algoritmi:**

| Algoritm | Probabilitati | Metoda |
|----------|---------------|--------|
| **Naive Bayes** | Reale | Calculeaza probabilitati bazate pe frecventa cuvintelor |
| **SVM** | Nu (100% setat manual) | Nu calculeaza probabilitati fara Platt scaling |
| **Random Forest** | Reale | Media probabilitatilor din toti arborii |

**Implicatii pentru comparatie:**

- Pentru compararea "increderii", trebuie sa ne concentram pe **Naive Bayes si Random Forest**
- Pentru SVM, "increderea" de 100% este doar o valoare placeholder
- **Predictia categoriei** este cea care conteaza pentru SVM, nu nivelul de incredere

---

### 5.5. Concluzie Finala

Proiectul a demonstrat ca pentru problema de **clasificare automata a documentelor text**, **Naive Bayes** este alegerea optima, oferind:
- Cea mai buna performanta (90.89% accuracy)
- Cea mai rapida viteza (antrenare in 0.0038s)
- Cel mai bun echilibru intre performanta si eficienta
- Probabilitati reale si interpretabile

**Rezultatele testelor pe articole reale arata:**

1. **Pentru articole cu categorie clara**: Toti algoritmii functioneaza bine si sunt de acord
2. **Pentru articole cu topics mixte**: Algoritmii pot da rezultate diferite, bazate pe abordarea lor
3. **Pentru articole ambigue**: Diferentele intre algoritmi devin evidente:
   - Naive Bayes se bazeaza pe frecventa si claritatea cuvintelor
   - SVM si Random Forest identifica tema centrala si contextul

**Observatii importante:**

- **SVM ofera probabilitati doar cu Platt scaling**: Fara `probability=True`, SVM returneaza 100% incredere (valoare placeholder)
- **Naive Bayes si Random Forest ofera probabilitati reale**: Pot fi comparate direct pentru nivelul de incredere
- **Predictia categoriei este mai importanta decat probabilitatea**: Chiar daca SVM nu ofera probabilitati, predictia sa este corecta

SVM ofera performanta similara dar cu viteza mult mai lenta, iar Random Forest, desi versatil, nu este optim pentru date text sparse.

**Rezultatul proiectului confirma ca algoritmii simpli, optimizati pentru domeniul specific (text), pot depasi algoritmii mai complexi si generici.**

---

## 6. Bibliografie si Resurse

- **20 Newsgroups Dataset**: Standard dataset pentru clasificare text
- **scikit-learn**: Biblioteca Python pentru machine learning
- **TF-IDF**: Term Frequency-Inverse Document Frequency pentru vectorizare
- **Naive Bayes**: Algoritm probabilistic pentru clasificare
- **SVM**: Support Vector Machine pentru clasificare
- **Random Forest**: Ensemble learning cu arbori de decizie


