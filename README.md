# 📋 Task Manager

Un manager de task-uri simplu și eficient, construit în Python, care te ajută să îți organizezi activitățile zilnice.

## ✨ Caracteristici

- ✅ Adăugare task-uri cu priorități (high, medium, low)
- 📝 Listare task-uri (toate, pending, completed)
- ✓ Marcare task-uri ca finalizate
- 🗑️ Ștergere task-uri
- 📊 Statistici despre progresul task-urilor
- 💾 Persistență a datelor în fișier JSON
- 🎨 Interfață CLI prietenoasă

## 🚀 Instalare

Acest proiect folosește doar biblioteci standard Python, deci nu necesită instalarea de dependențe externe.

```bash
# Clonează sau descarcă proiectul
# Apoi rulează direct:
python3 task_manager.py
```

## 📖 Utilizare

### Comenzi disponibile:

- `add <descriere> [priority]` - Adaugă un task nou
  - Exemplu: `add Cumpără lapte high`
  - Prioritate: high, medium (implicit), low

- `list [status]` - Listează task-urile
  - `list` - toate task-urile
  - `list pending` - doar task-urile în așteptare
  - `list completed` - doar task-urile finalizate

- `complete <id>` - Marchează un task ca finalizat
  - Exemplu: `complete 1`

- `delete <id>` - Șterge un task
  - Exemplu: `delete 2`

- `stats` - Afișează statistici despre task-uri

- `help` - Afișează lista de comenzi

- `quit` - Ieșire din aplicație

### Exemple de utilizare:

```bash
> add Finalizează proiectul Python high
✅ Task adăugat cu succes! ID: 1

> add Citește documentația medium
✅ Task adăugat cu succes! ID: 2

> list
============================================================
ID    Status       Priority   Description
============================================================
1     ○ pending    🔴 high    Finalizează proiectul Python
2     ○ pending    🟡 medium  Citește documentația
============================================================

> complete 1
✅ Task 1 marcat ca finalizat!

> stats
========================================
Statistici Task-uri
========================================
Total: 2
Finalizate: 1
În așteptare: 1
Progres: 50.0%
========================================
```

## 📁 Structura Proiectului

```
.
├── task_manager.py    # Aplicația principală
├── requirements.txt   # Dependențe (gol - folosește doar stdlib)
├── README.md          # Documentație
└── tasks.json         # Fișier de date (generat automat)
```

## 🛠️ Tehnologii

- **Python 3.6+** - Limbajul de programare
- **JSON** - Pentru stocarea datelor
- **datetime** - Pentru gestionarea timpului

## 📝 Structura Datelor

Task-urile sunt stocate în format JSON cu următoarea structură:

```json
{
  "id": 1,
  "description": "Descrierea task-ului",
  "priority": "high",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00",
  "completed_at": null
}
```

## 🔧 Dezvoltare

Pentru a extinde funcționalitățile, poți:

1. Adăuga validări suplimentare
2. Implementa categorii pentru task-uri
3. Adăuga deadline-uri pentru task-uri
4. Integra cu servicii cloud
5. Crea o interfață grafică (GUI)

## 📄 Licență

Acest proiect este open source și disponibil pentru utilizare liberă.

## 👤 Autor

Creat cu ❤️ în Python

---

**Bucură-te de organizarea task-urilor! 🎉**

