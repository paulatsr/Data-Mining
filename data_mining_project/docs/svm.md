# Support Vector Machine (SVM) - Documentatie

## Ce este SVM?

Support Vector Machine (SVM) este un algoritm de machine learning puternic pentru clasificare si regresie. Este deosebit de eficient pentru probleme de clasificare text, fiind capabil sa gaseasca limite de decizie complexe intre clase.

## Cum functioneaza?

### Conceptul de baza

SVM incearca sa gaseasca un **hiperplan** (o linie in spatiu multidimensional) care separa cel mai bine clasele. Hiperplanul optim este cel care maximizeaza **marginea** (distanta) dintre cele mai apropiate puncte din clase diferite.

### Support Vectors

Punctele cele mai apropiate de hiperplan se numesc **support vectors**. Acestea sunt esentiale pentru definirea limitei de decizie - daca le eliminam, limitele se schimba.

### Separare liniara vs neliniara

#### Separare liniara
Pentru date care pot fi separate liniar, SVM gaseste o linie dreapta care separa clasele.

#### Separare neliniara (Kernel Trick)
Pentru date care nu pot fi separate liniar, SVM foloseste un **kernel** care transforma datele intr-un spatiu dimensional mai mare unde devin separabile liniar.

### Tipuri de kernel

1. **Linear**: `K(x, y) = x · y`
   - Cel mai simplu si rapid
   - Functioneaza bine pentru date liniar separabile
   - Folosit in proiectul nostru

2. **RBF (Radial Basis Function)**: `K(x, y) = exp(-γ||x-y||²)`
   - Functioneaza bine pentru date complexe
   - Mai lent decat linear

3. **Polynomial**: `K(x, y) = (x · y + 1)^d`
   - Capteaza relatii polinomiale
   - Parametrul `d` controleaza gradul

### Parametrul C (Regularizare)

Parametrul **C** controleaza trade-off-ul intre:
- **Margine larga** (C mic) = mai putine erori de clasificare, dar margine mai ingusta
- **Clasificare corecta** (C mare) = margine mai ingusta, dar mai putine erori

```
C mic → margine larga, toleranta la erori
C mare → margine ingusta, zero toleranta la erori
```

### Soft Margin vs Hard Margin

- **Hard Margin**: Nu permite erori (C = infinit)
- **Soft Margin**: Permite erori pentru a gasi o solutie mai robusta (C finit)

## Implementare in proiect

### Clasa: `SVMClassifier`

```python
from sklearn.svm import SVC

class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, max_iter=1000):
        self.model = SVC(kernel=kernel, C=C, max_iter=max_iter, random_state=42)
```

### Parametrii folositi

- **kernel='linear'**: Folosim kernel liniar pentru viteza si eficienta
- **C=1.0**: Regularizare standard (echilibrat)
- **max_iter=1000**: Numar maxim de iteratii pentru optimizare
- **random_state=42**: Pentru reproducibilitate

### De ce kernel linear?

Pentru clasificare text cu TF-IDF:
- Datele sunt deja intr-un spatiu dimensional mare (10,000 features)
- Kernel linear functioneaza excelent pentru date sparse
- Este mult mai rapid decat RBF sau polynomial
- Rezultatele sunt comparabile sau mai bune

## Avantaje

1. **Performanta ridicata**: Accuracy bun pe multe tipuri de date
2. **Eficient pe date sparse**: Functioneaza excelent cu TF-IDF
3. **Memorie eficienta**: Foloseste doar support vectors pentru predictie
4. **Robust la overfitting**: Regularizare prin parametrul C
5. **Versatil**: Poate folosi diferite kernel-uri pentru date complexe

## Dezavantaje

1. **Lent la antrenare**: Complexitate O(n²) sau O(n³) in functie de kernel
2. **Nu scaleaza bine**: Devine foarte lent pe dataset-uri mari
3. **Sensibil la scaling**: Datele trebuie normalizate (dar TF-IDF face asta automat)
4. **Interpretabilitate limitata**: Mai greu de inteles decat Naive Bayes
5. **Parametrii sensibili**: C si kernel trebuie optimizati

## Cand se foloseste?

- Clasificare text (excelent pentru TF-IDF)
- Clasificare cu multe caracteristici
- Cand ai nevoie de performanta ridicata
- Cand datele sunt sparse (multe zerouri)
- Cand ai dataset-uri de dimensiune medie

## Rezultate in proiect

In proiectul nostru, SVM:
- **Accuracy**: ~89.66% (al doilea cel mai bun)
- **Timp antrenare**: ~7.4 secunde (cel mai lent)
- **Timp predictie**: ~1.45 secunde (mediu)

### De ce este mai lent?

SVM trebuie sa:
1. Calculeze distantele intre toate perechile de documente
2. Optimizeze functia obiectiv (problema de optimizare quadratica)
3. Gaseasca support vectors

Pentru 4,560 documente de antrenare, acest proces este costisitor computațional.

## Optimizari posibile

1. **LinearSVC**: Versiune optimizata pentru kernel linear (mai rapida)
2. **Reducere features**: Folosirea mai putine features (ex: 5,000 in loc de 10,000)
3. **Sampling**: Antrenare pe un subset al datelor
4. **Early stopping**: Oprire dupa un numar de iteratii

## Comparatie cu Naive Bayes

| Aspect | Naive Bayes | SVM |
|--------|-------------|-----|
| Viteza antrenare | Foarte rapid | Lent |
| Viteza predictie | Foarte rapid | Mediu |
| Accuracy | Foarte bun | Foarte bun |
| Interpretabilitate | Excelent | Limitata |
| Scalabilitate | Excelent | Limitata |

## Exemple de utilizare

```python
# Creare clasificator
svm = SVMClassifier(kernel='linear', C=1.0, max_iter=2000)

# Antrenare
svm.train(X_train, y_train)

# Predictie
predictions = svm.predict(X_test)

# Evaluare
metrics = svm.evaluate(X_test, y_test)
```

## Concluzie

SVM este un algoritm puternic pentru clasificare text, oferind:
- **Performanta ridicata**: Accuracy comparabil cu Naive Bayes
- **Robustete**: Functioneaza bine pe date complexe
- **Versatilitate**: Poate folosi diferite kernel-uri

Principalul dezavantaj este viteza de antrenare, dar pentru dataset-uri de dimensiune medie si cand ai nevoie de performanta maxima, SVM este o alegere excelenta!

