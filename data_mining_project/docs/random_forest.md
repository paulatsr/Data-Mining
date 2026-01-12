# Random Forest - Documentatie

## Ce este Random Forest?

Random Forest este un algoritm de machine learning bazat pe **ensemble learning** (invatare ansamblu). Combina predictiile a mai multor **arbori de decizie** pentru a obtine o performanta mai buna decat un singur arbore.

## Cum functioneaza?

### Arbori de decizie

Un **arbore de decizie** este un model care ia decizii prin intrebari succesive. Pentru text:
- "Contine cuvantul 'hockey'?" → DA → "Contine 'puck'?" → DA → Clasa: Sport
- "Contine cuvantul 'hockey'?" → NU → "Contine 'medical'?" → DA → Clasa: Medicina

### Bootstrap Aggregating (Bagging)

Random Forest foloseste **bagging**:
1. Creeaza mai multi subset-uri aleatorii din datele de antrenare (cu inlocuire)
2. Antreneaza un arbore de decizie pe fiecare subset
3. Combina predictiile tuturor arborilor (vot majoritar)

### "Random" in Random Forest

Algoritmul este "random" din doua motive:

1. **Random sampling**: Fiecare arbore este antrenat pe un subset aleator al datelor
2. **Random features**: La fiecare nod, se alege un subset aleator de caracteristici pentru split

Aceasta randomizare:
- Reduce overfitting
- Face arborii mai diversi
- Creste robustetea modelului

### Procesul de antrenare

```
1. Pentru fiecare arbore (i = 1 la n_estimators):
   a. Creeaza un bootstrap sample (subset aleator cu inlocuire)
   b. Antreneaza un arbore de decizie pe acest sample
   c. La fiecare nod, alege random un subset de features pentru split
   
2. Pentru predictie:
   a. Fiecare arbore face o predictie
   b. Se alege clasa cu cele mai multe voturi (majority voting)
```

### Split-uri in arbori

La fiecare nod, arborele alege:
- **Cea mai buna caracteristica** din subset-ul aleator
- **Cea mai buna valoare de split** care maximizeaza separarea claselor

Criterii comune:
- **Gini impurity**: Masoara cat de "impur" este un nod
- **Entropy**: Masoara incertitudinea
- **Information gain**: Cat de mult se reduce incertitudinea dupa split

## Implementare in proiect

### Clasa: `RandomForestTextClassifier`

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

### Parametrii folositi

- **n_estimators=100**: Numarul de arbori in padure
  - Mai multi arbori = mai buna performanta, dar mai lent
  - 100 este un compromis bun

- **max_depth=None**: Adancimea maxima a arborilor
  - None = arbori pot creste pana cand toate frunzele sunt pure
  - Limiteaza overfitting daca este setat

- **random_state=42**: Pentru reproducibilitate

- **n_jobs=-1**: Foloseste toate core-urile CPU pentru antrenare paralela

### De ce functioneaza pentru text?

Random Forest functioneaza bine pentru text pentru ca:
- Poate captura interactiuni complexe intre cuvinte
- Este robust la noise (cuvinte irelevante)
- Nu necesita normalizare (spre deosebire de SVM)
- Poate identifica feature-uri importante

## Avantaje

1. **Performanta buna**: Accuracy bun pe multe tipuri de date
2. **Robust la overfitting**: Bagging si randomizare reduc overfitting
3. **Nu necesita preprocesare complexa**: Functioneaza direct pe date
4. **Paralelizabil**: Poate antrena arborii in paralel
5. **Feature importance**: Poate identifica care cuvinte sunt importante
6. **Robust la missing values**: Poate gestiona date lipsa
7. **Nu necesita scaling**: Functioneaza pe date ne-normalizate

## Dezavantaje

1. **Lent la antrenare**: Trebuie sa antreneze multi arbori
2. **Memorie**: Ocupa multa memorie (stocheaza toti arborii)
3. **Interpretabilitate limitata**: Mai greu de inteles decat un singur arbore
4. **Nu captureaza ordinea**: Nu tine cont de ordinea cuvintelor
5. **Poate fi overfit pe date mici**: Necesita suficiente date

## Cand se foloseste?

- Clasificare cu multe caracteristici
- Cand ai nevoie de feature importance
- Cand datele au interactiuni complexe
- Cand vrei un model robust
- Cand ai resurse computaționale suficiente

## Rezultate in proiect

In proiectul nostru, Random Forest:
- **Accuracy**: ~85.01% (al treilea, dar inca bun)
- **Timp antrenare**: ~0.52 secunde (mediu)
- **Timp predictie**: ~0.029 secunde (rapid)

### De ce accuracy mai mic?

Pentru clasificare text cu TF-IDF:
- Datele sunt foarte sparse (multe zerouri)
- Random Forest functioneaza mai bine pe date dense
- Nu captureaza semantica (ordinea cuvintelor)
- Naive Bayes si SVM sunt optimizate pentru text sparse

## Feature Importance

Random Forest poate identifica care cuvinte sunt importante:

```python
# Dupa antrenare
feature_importance = rf.model.feature_importances_

# Top 10 cuvinte importante
top_features = np.argsort(feature_importance)[-10:]
```

Aceasta este o functionalitate utila pentru:
- Interpretarea modelului
- Feature selection
- Understanding domain-ului

## Comparatie cu alti algoritmi

| Aspect | Naive Bayes | SVM | Random Forest |
|--------|-------------|-----|---------------|
| Viteza antrenare | Foarte rapid | Lent | Mediu |
| Viteza predictie | Foarte rapid | Mediu | Rapid |
| Accuracy (text) | Excelent | Excelent | Bun |
| Interpretabilitate | Excelent | Limitata | Medie |
| Feature importance | Nu | Nu | Da |
| Scalabilitate | Excelent | Limitata | Buna |

## Optimizari posibile

1. **Grid Search**: Optimizare parametri (n_estimators, max_depth)
2. **Feature selection**: Folosirea doar a feature-urilor importante
3. **Early stopping**: Oprire daca performanta nu se imbunatateste
4. **Reducere n_estimators**: Folosirea mai putini arbori pentru viteza

## Exemple de utilizare

```python
# Creare clasificator
rf = RandomForestTextClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

# Antrenare
rf.train(X_train, y_train)

# Predictie
predictions = rf.predict(X_test)

# Evaluare
metrics = rf.evaluate(X_test, y_test)

# Feature importance
importance = rf.model.feature_importances_
```

## Concluzie

Random Forest este un algoritm versatil si robust care:
- **Functioneaza bine** pe multe tipuri de date
- **Ofera feature importance** pentru interpretare
- **Este paralelizabil** pentru viteza

Pentru clasificare text, Naive Bayes si SVM sunt de obicei mai bune, dar Random Forest ramane o alegere solida, mai ales cand ai nevoie de feature importance sau cand datele au interactiuni complexe!

