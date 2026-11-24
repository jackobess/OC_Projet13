# AskJack — backlog évolutions / fixes

Dernière mise à jour : 24/11/2026. Rien de décidé ici : ce sont des pistes à trier.

## Qualité des réponses / RAG
- [ ] Mettre à jour le CV avec les compétences de la formation (Data Science / ML) — aujourd'hui le CV ne parle que du parcours IT santé.
- [x] LLM de génération par défaut : `mistral-medium-latest` (fait le 24/11).
- [x] « Quels projets utilisent des compétences RAG ? » : P10 oublié avec small / top_k 10 ; correct (P9, P10, P13) avec medium / top_k 15 (24/11).
- [x] Ne plus citer de numéros de sources (« Sources 6 à 16 ») dans les réponses : règle ajoutée aux prompts système (24/11).
- [ ] Routeur sur un petit modèle dédié.
- [ ] Garde-fou code sur `project_ids` (ne garder que les n° réellement cités dans la question) — écarté pour l'instant (préférence : logique dans le prompt), à reconsidérer si le routeur régresse.
- [ ] Découpage par sections Markdown (MarkdownHeaderTextSplitter) + métadonnée `section` (ex. cibler le retour évaluateur).
- [ ] OCR des slides en image (RapidOCR pour pymupdf4llm) si des prezs sortent pauvres en texte.
- [ ] Renommer les fichiers de prez avec leur sujet (`P5_Prez_Deployer une API...`) : l'en-tête contextuel des chunks en profite.
- [ ] Afficher en debug le JSON brut du routeur (avant nettoyage) en plus de l'intent nettoyé.

## Fonctionnalités app
- [ ] Mémoire de conversation (reformuler la question avec l'historique avant le routeur → 1 appel LLM de plus).
- [ ] Streaming de la réponse.
- [ ] Mode candidat en ligne derrière un mot de passe (aujourd'hui : local uniquement).
- [ ] Journal persistant des questions/métriques en production (Neon / PostgreSQL) — aujourd'hui : en mémoire + export CSV.
- [ ] Estimation du coût par question (prix par modèle dans la config) — écarté pour l'instant.
- [ ] Mettre `project_catalog` en cache (recalculé à chaque question).
- [x] Sidebar : modèle, top_k, max_tokens et taille du corpus (fichiers / chunks) affichés sous le compteur (24/11).

## Technique / maintenance
- [ ] **Cache de config sur Streamlit Cloud** : `load_resources` (`st.cache_resource`) garde l'ancienne config après un push d'un YAML (le cache n'est vidé que si le code de la fonction change) → aujourd'hui il faut « Reboot app ». Fix : passer la date de modification des YAML (config, datasources, prompts, fixed_responses) en argument de la fonction en cache pour qu'il se vide tout seul.
- [ ] `st.components.v1.html` (scroll auto) est déprécié dans les versions récentes de Streamlit (OK avec la version figée 1.51) : à remplacer en cas de montée de version.
- [ ] Migrer `langchain-community` (en fin de vie) vers le package d'intégration FAISS autonome.
- [ ] `project_catalog` : utiliser `project_id_regex` de la config au lieu de la regex en dur `^[Pp]\d{1,2}_`.
- [ ] Contrôle de cohérence du modèle d'embedding : écrire `index/meta.json` à l'indexation, vérifier au démarrage de l'app.
- [ ] Notebook 01 : ignorer les `.txt` orphelins à l'étape 3 (source supprimée/renommée).
- [ ] Contournement `SSL_CERT_FILE` dans app.py (lancement depuis Git Bash) — optionnel.

## Évaluation / livrables
- [ ] Jeu de questions de non-régression (dont : « Quels projets utilisent des compétences RAG ? », Kubernetes, hors sujet, injection, annonce).
- [ ] Évaluation Ragas (Could de la roadmap).
- [ ] Mettre à jour le rapport de conduite de projet avec l'implémentation réelle.
- [x] README : URL du dépôt GitHub et de l'app (24/11).

## Divers
- [x] Scroll : après une nouvelle réponse, la vue est ramenée sur la dernière question/réponse (script JS, 24/11).
- [x] Questions exemples terminées par « ... » : ouvertes dans une zone de texte à compléter (Envoyer / Annuler) au lieu d'être envoyées (24/11).
