# Explicatie Rezultate Articole de Test

## Articol 3: Story cu toate categoriile

### Rezultate observate:
- **Naive Bayes**: Stiinta - Spatiu
- **SVM**: Stiinta - Medicina  
- **Random Forest**: Stiinta - Medicina

### De ce diferente?

Articolul mentioneaza atat cuvinte despre **spatiu** cat si despre **medicina**:

**Cuvinte despre spatiu:**
- "space exploration"
- "space stations" 
- "space science"
- "space missions"

**Cuvinte despre medicina:**
- "medical center"
- "medical imaging"
- "medical research"
- "medical expertise"
- "patient scans"
- "patient outcomes"
- "X-ray"
- "disease detection"
- "treatments"
- "patients"

### Analiza algoritmilor:

#### Naive Bayes - A dat "Stiinta - Spatiu"
- **De ce?** Naive Bayes numara frecventa cuvintelor si calculeaza probabilitati independente
- Articolul mentioneaza "space" de 4 ori in contexturi clare
- Cuvintele despre medicina sunt mai distribuite si mai generale
- Naive Bayes este sensibil la frecventa absoluta a cuvintelor
- Cuvintele despre spatiu apar in contexte mai clare si distincte

#### SVM si Random Forest - Au dat "Stiinta - Medicina"
- **De ce?** Acesti algoritmi iau in considerare si contextul si interactiunile
- Articolul are mai multe cuvinte legate de medicina (10+ vs 4 pentru spatiu)
- Contextul medical este mai prezent (Dr. Martinez, hospital, patient care)
- SVM si Random Forest pot identifica pattern-uri mai complexe
- Medicina este tema centrala a povestii (Dr. Martinez lucreaza in medical center)

### Concluzie:

**Articolul nu este clar** - mentioneaza multiple categorii in mod egal, ceea ce face clasificarea dificila.

**Diferentele apar din:**
1. **Frecventa cuvintelor**: Naive Bayes se bazeaza mai mult pe numarul de aparitii
2. **Context si pattern-uri**: SVM si Random Forest pot identifica contextul mai bine
3. **Distributia cuvintelor**: Medicina apare mai des, dar spatiul apare in contexte mai clare

Aceasta demonstreaza ca:
- **Naive Bayes** este mai sensibil la frecventa absoluta
- **SVM si Random Forest** sunt mai bune la identificarea contextului si pattern-urilor complexe
- Pentru texte ambigue, algoritmii pot da rezultate diferite bazate pe abordarea lor

