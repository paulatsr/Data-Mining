# Naive Bayes - Documentatie

## Ce este Naive Bayes?

Naive Bayes este un algoritm de clasificare probabilistica bazat pe teorema lui Bayes. Este unul dintre cei mai simpli si mai eficienti algoritmi de machine learning pentru clasificare text.

## Cum functioneaza?

### Teorema lui Bayes

Algoritmul se bazeaza pe teorema lui Bayes:

```
P(clasa | caracteristici) = P(caracteristici | clasa) * P(clasa) / P(caracteristici)
```

Unde:
- **P(clasa | caracteristici)** = probabilitatea ca un document sa apartina unei clase date caracteristicile sale
- **P(caracteristici | clasa)** = probabilitatea de a vedea aceste caracteristici in clasa respectiva
- **P(clasa)** = probabilitatea a priori a clasei
- **P(caracteristici)** = probabilitatea de a vedea aceste caracteristici (normalizare)

### Presupunerea "Naive"

Algoritmul se numeste "Naive" (naiv) pentru ca presupune ca toate caracteristicile (cuvintele) sunt **independente** unele de altele. In realitate, cuvintele dintr-un text nu sunt independente, dar aceasta presupunere simplifica mult calculele si functioneaza surprinzator de bine in practica.

### Pentru clasificare text

Pentru clasificare text, algoritmul:

1. **Calculeaza probabilitatile pentru fiecare cuvant:**
   - P(cuvant | clasa) = de cate ori apare cuvantul in documentele din clasa respectiva

2. **Calculeaza probabilitatea clasei:**
   - P(clasa) = numarul de documente din clasa / numarul total de documente

3. **Pentru un document nou:**
   - Inmulteste probabilitatile tuturor cuvintelor pentru fiecare clasa
   - Alege clasa cu cea mai mare probabilitate

### Formula pentru clasificare text

```
P(clasa | document) ∝ P(clasa) * ∏ P(cuvant_i | clasa)
```

Unde:
- ∏ = produsul (inmultirea) probabilitatilor tuturor cuvintelor
- ∝ = proportional cu (ignoram numitorul care este acelasi pentru toate clasele)

### Laplace Smoothing

Pentru a evita problemele cand un cuvant nu apare in antrenare, se foloseste **Laplace smoothing** (sau additive smoothing):

```
P(cuvant | clasa) = (count(cuvant, clasa) + alpha) / (count(clasa) + alpha * vocab_size)
```

Unde:
- **alpha** = parametru de smoothing (in proiect: alpha=1.0)
- **vocab_size** = numarul total de cuvinte unice

## Implementare in proiect

### Clasa: `NaiveBayesClassifier`

```python
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.model = MultinomialNB(alpha=alpha)
```

### Parametrii folositi

- **alpha=1.0**: Parametru de Laplace smoothing
  - Valori mai mari = mai mult smoothing (mai conservator)
  - Valori mai mici = mai putin smoothing (mai agresiv)
  - Valoarea 1.0 este o alegere standard

### MultinomialNB

Folosim `MultinomialNB` din scikit-learn pentru ca:
- Functioneaza bine cu date discrete (numar de aparitii ale cuvintelor)
- Este optimizat pentru clasificare text
- Suporta smoothing automat

## Avantaje

1. **Rapid la antrenare**: O(n) complexitate, unde n = numarul de documente
2. **Rapid la predictie**: Calculeaza doar probabilitati simple
3. **Functioneaza bine cu putine date**: Nu necesita mult training data
4. **Robust la noise**: Smoothing previne overfitting
5. **Interpretabil**: Poti vedea care cuvinte contribuie la fiecare clasa
6. **Nu necesita tuning complex**: Parametrii sunt simpli

## Dezavantaje

1. **Presupunerea de independenta**: Cuvintele nu sunt independente in realitate
2. **Nu captureaza contextul**: Nu tine cont de ordinea cuvintelor
3. **Poate fi influentat de cuvinte frecvente**: Stop words pot domina (dar le eliminam in preprocessing)

## Cand se foloseste?

- Clasificare text (email spam, sentiment analysis, categorizare documente)
- Clasificare cu multe caracteristici
- Cand ai nevoie de predictii rapide
- Cand ai putine date de antrenare
- Cand vrei un model interpretabil

## Rezultate in proiect

In proiectul nostru, Naive Bayes:
- **Accuracy**: ~90.89% (cel mai bun!)
- **Timp antrenare**: ~0.004 secunde (cel mai rapid!)
- **Timp predictie**: ~0.001 secunde (cel mai rapid!)

## Exemple de utilizare

```python
# Creare clasificator
nb = NaiveBayesClassifier(alpha=1.0)

# Antrenare
nb.train(X_train, y_train)

# Predictie
predictions = nb.predict(X_test)

# Evaluare
metrics = nb.evaluate(X_test, y_test)
```

## Concluzie

Naive Bayes este un algoritm excelent pentru clasificare text, oferind un echilibru perfect intre:
- **Performanta**: Accuracy ridicat
- **Viteza**: Antrenare si predictie foarte rapide
- **Simplitate**: Usor de inteles si implementat

Este de obicei primul algoritm de incercat pentru probleme de clasificare text!

