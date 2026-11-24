# AskJack 💬

> Ce que j'ai appris dans ma formation OpenClassrooms **Data Scientist – Machine Learning** (13 projets) + mon parcours (professionnel et formation)

AskJack est un assistant **RAG** (Retrieval-Augmented Generation) qui répond aux questions sur mon parcours :
expérience professionnelle, formation, projets, stack technique et retours d'évaluateurs.
Il s'appuie **uniquement** sur mes documents (CV, fiches projets, présentations de soutenance, README…) et cite ses sources.

Askjack est aussi le Projet personnel technique réalisé dans le cadre du **Projet 13** de la formation OpenClassrooms.

---

## Deux modes

| Mode | Public | Comportement |
|---|---|---|
| `recrut` | recruteurs (app en ligne) | vouvoiement, 3e personne, réponses factuelles et concises |
| `candidat` | moi (en local) | tutoiement, analyse d'une annonce d'emploi + proposition de réponse |

Un **mode debug** (local uniquement) affiche l'intention détectée, les filtres, les chunks récupérés,
le prompt final, les tokens et les latences, et permet de changer de mode, de `top_k` et de LLM à la volée.
Ce **mode debug** permet de tester et finetuner AskJack afin de choisir les meilleurs paramètres

---

## Fonctionnement

```
question
   │
   ▼
1. Routage (get_intent)        1 appel LLM court, sortie JSON
   │   intent : question | annonce | salutation | hors_sujet | about_askjack | ...
   │   doc_types à cibler, n° de projets cités, compétences (si annonce)
   │
   ├── réponse toute faite ? (fixed_responses.yaml) ──► réponse directe, sans retrieval ni génération
   │
   ▼
2. Retrieval                   embedding de la question (mistral-embed) + index FAISS
   │   pré-filtre sur les métadonnées -> recherche vectorielle exacte -> complément hors filtre si < k
   │   annonce : 1 recherche par compétence extraite
   │   + fichiers forcés en tête de contexte (forcecontext)
   │
   ▼
3. Génération                  prompt système du mode + chunks + question -> LLM Mistral
   │
   ▼
réponse + sources (+ debug : intent, filtres, chunks, prompt, tokens, latences)
```

**Stack** : Python · LangChain · FAISS · Mistral AI (embeddings + LLM) · Streamlit ·
PyMuPDF4LLM / MarkItDown (conversion des documents en Markdown) · pandas

---

## Arborescence

```
.
├── app.py                          # application Streamlit
├── src/
│   └── rag.py                      # cœur RAG : config, routage, retrieval, génération, métriques
├── config.yaml                     # réglages de l'app (modèles, mode, UI, retrieval, chemins)
├── datasource_conf.yaml            # description du corpus (sources, doc_types, forcecontext...)
├── prompts.yaml                    # prompts : routeur, système par mode, gabarit utilisateur
├── fixed_responses.yaml            # réponses toutes faites par intent + messages d'incident
├── notebooks
│   └── 01_preparation_vectorisation.ipynb   # pipeline données : sources -> txt -> chunks -> index
│   └── 02_test_rag.ipynb                    # tests du RAG (routeur, retrieval, réponses)
├── data/
│   ├── ...                         # documents sources (CV, fiches, prezs, README, xlsx...)
│   ├── _txt/                       # textes extraits (arborescence miroir des sources, retouchables)
│   └── _chunks/chunks.jsonl        # chunks + métadonnées
└── index/faiss/                    # index FAISS (index.faiss + index.pkl) — seul élément utilisé au runtime
```

---

## Installation

```bash
git clone <url-du-depot>
cd <depot>
pip install -r requirements.txt
```

Créer un fichier `.env` à la racine (**jamais commité**) :

```
MISTRAL_API_KEY=...
HF_TOKEN=...              # facultatif (téléchargement du tokenizer, évite un warning)
ASKJACK_DEBUG=true        # local : active le panneau debug
ASKJACK_MODE=candidat     # local : surcharge app.mode de config.yaml
```

---

## Préparer les données (notebook 01)

1. Déposer les documents dans `data/` et les déclarer dans `datasource_conf.yaml` (glob, `doc_type`, `label`, `description`).
   Nommer les fichiers de projet avec le préfixe `Px_` / `Pxx_` (ex. `P5_Fiche_Deployer une API.docx`) : le n° de projet est extrait automatiquement.
2. Lancer `01_preparation_vectorisation.ipynb` :
   - **inventaire** : seules les sources nouvelles ou modifiées sont converties (`FORCE_ALL = True` pour tout reconvertir — écrase les retouches) ;
   - **extraction** : PDF → `pymupdf4llm`, docx → MarkItDown, xlsx → une ligne par ligne de tableau ;
   - les `.txt` de `data/_txt/` peuvent être **corrigés à la main** avant le chunking ;
   - **chunking** de tous les `.txt` (le nom du fichier est ajouté en en-tête de chaque chunk) puis **vectorisation** complète.

Options utiles par source (`datasource_conf.yaml`) :

| Option | Effet |
|---|---|
| `no_split: true` | le fichier forme un seul chunk |
| `forcecontext: true` | fichier toujours injecté en entier dans le contexte |
| `forcecontext: if_filter` | injecté si son `doc_type` est ciblé par le routeur |
| `forcecontext: if_filter_or_not_sure` | idem, ou si le routeur n'a pas d'avis (`not-sure`) |

---

## Lancer l'application

```bash
streamlit run app.py
```

> Windows / Git Bash : si une erreur `SSL_CERT_FILE` apparaît, lancer depuis PowerShell ou faire `unset SSL_CERT_FILE`.

---

## Déploiement (Streamlit Community Cloud)

1. Pousser le dépôt sur GitHub, **index `index/faiss/` inclus** (l'app ne fait pas tourner les notebooks).
2. Créer l'app sur Streamlit Cloud à partir du dépôt (`app.py`).
3. Dans **Settings → Secrets**, renseigner `MISTRAL_API_KEY` (et `HF_TOKEN`).
   Ne **pas** définir `ASKJACK_DEBUG` : le panneau debug n'existe alors pas, et rien dans l'interface ne permet de l'activer.

Garde-fous en production : limite de questions par session (`app.max_questions_per_session`),
clé API dédiée avec plafond de dépense mensuel, intents `hors_sujet` traités sans appel de génération.

---

## Configuration — fichiers YAML

| Fichier | Contenu |
|---|---|
| `config.yaml` | `app` (mode, debug, limite de questions), `candidate` (nom, surnom, titre), `ui` (textes par mode), `models` (embedding, llm, `llm_alternatives`, temperature, max_tokens), `retrieval` (top_k, k_per_skill, force_max_chars), `paths` |
| `datasource_conf.yaml` | `project_id_regex`, `sources` |
| `prompts.yaml` | `router`, `system.recrut`, `system.candidat`, `user_template` |
| `fixed_responses.yaml` | réponses par intent (texte, liste tirée au hasard, ou texte par mode), `default`, messages de réponse vide / tronquée |

Le modèle d'embedding (`models.embedding`) doit rester **le même** entre l'indexation et l'app.
