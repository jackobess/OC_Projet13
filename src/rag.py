"""AskJack — cœur RAG : config, routage (get_intent), retrieval, prompt, génération.
Utilisé par le notebook 02 et par l'app Streamlit.

Pipeline d'une question :
  1. get_intent  : 1 appel LLM court (JSON) -> intent (question|annonce), doc_types, project_ids, skills
  2. retrieve    : pré-filtre métadonnées -> recherche vectorielle exacte -> complément hors filtre si < k
                   (annonce : 1 recherche par compétence extraite, dédoublonnée)
  3. génération  : prompt système du mode + contexte -> réponse
"""
import json
import os
import random
import re
import time
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import yaml
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

MODES = ("recrut", "candidat")
INTENTS = ("question", "annonce")        # intents à traitement codé (pipeline RAG) ; les autres -> fixed_responses.yaml
FALLBACK_RESPONSE = "Sorry? I beg your pardon?"   # intent inconnu sans "default" dans fixed_responses.yaml
FALLBACK_EMPTY = "⚠️ Le modèle n'a renvoyé aucune réponse (raison : {finish_reason}). Merci de réessayer."
FALLBACK_TRUNCATED = "⚠️ Réponse tronquée : la limite de longueur (max_tokens) a été atteinte."


# ================================================================ config
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def fill(template: str, **values) -> str:
    """Remplace uniquement les {placeholders} connus -> les accolades JSON des prompts restent intactes."""
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def load_config(root: Path) -> dict:
    """Charge config.yaml (+ paths.datasources_file + paths.prompts_file) ; ASKJACK_MODE / ASKJACK_DEBUG surchargent app.mode / app.debug."""
    with open(Path(root) / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds_file = cfg.get("paths", {}).get("datasources_file")
    if ds_file:                                     # sources + project_id_regex dans un fichier à part
        with open(Path(root) / ds_file, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f))
    pr_file = cfg.get("paths", {}).get("prompts_file", "prompts.yaml")
    with open(Path(root) / pr_file, encoding="utf-8") as f:     # prompts : router, system.<mode>, user_template
        cfg["prompts"] = yaml.safe_load(f)
    fr_path = Path(root) / cfg.get("paths", {}).get("fixed_responses_file", "fixed_responses.yaml")
    cfg["fixed_responses"] = {}
    if fr_path.exists():                                         # facultatif
        with open(fr_path, encoding="utf-8") as f:
            cfg["fixed_responses"] = yaml.safe_load(f) or {}
    app = cfg.setdefault("app", {})
    app["mode"] = os.getenv("ASKJACK_MODE", app.get("mode", "recrut")).strip().lower()
    app["debug"] = _env_bool("ASKJACK_DEBUG", app.get("debug", False))
    if app["mode"] not in MODES:
        raise ValueError(f"mode inconnu : {app['mode']} (attendu : {MODES})")
    return cfg


def ui_texts(cfg: dict, mode: str | None = None) -> dict:
    """Textes d'interface du mode, avec {first_name}, {nickname}... remplacés."""
    mode = mode or cfg["app"]["mode"]
    fmt = lambda s: s.format(**cfg["candidate"]) if isinstance(s, str) else s
    return {k: ([fmt(x) for x in v] if isinstance(v, list) else fmt(v))
            for k, v in cfg.get("ui", {}).get(mode, {}).items()}


def source_catalog(cfg: dict) -> dict:
    """{doc_type: description} lu dans config.sources (description, sinon label)."""
    cat = {}
    for s in cfg["sources"]:
        cat.setdefault(s["doc_type"], s.get("description") or s["label"])
    return cat


# ================================================================ index
def load_vectorstore(cfg: dict, root: Path) -> FAISS:
    embeddings = MistralAIEmbeddings(model=cfg["models"]["embedding"])
    return FAISS.load_local(str(Path(root) / cfg["paths"]["index_dir"]), embeddings,
                            allow_dangerous_deserialization=True)


def project_catalog(vs: FAISS) -> dict:
    """{project_id: titre} déduit des noms de fichiers de l'index (fiche en priorité)."""
    cat = {}
    for doc_id in vs.index_to_docstore_id.values():
        m = vs.docstore.search(doc_id).metadata
        pid = m.get("project_id")
        if pid is None:
            continue
        title = re.sub(r"^[Pp]\d{1,2}_", "", Path(m["source_path"]).stem)
        if pid not in cat or m.get("doc_type") == "fiche":
            cat[pid] = title
    return dict(sorted(cat.items()))


def _match(meta: dict, filters: dict) -> bool:
    """Filtre : valeur scalaire -> égalité ; liste -> appartenance."""
    for f, v in filters.items():
        val = meta.get(f)
        if isinstance(v, (list, tuple, set)):
            if val not in v:
                return False
        elif val != v:
            return False
    return True


def retrieve(vs: FAISS, question: str, k: int, filters: dict | None = None,
             complete: bool = True, exclude: set | None = None, stats: dict | None = None):
    """Liste de (Document, distance L2², origine) — origine = 'filtre' | 'complément' | 'tous'.

    1. pré-filtre sur les métadonnées
    2. recherche vectorielle exacte sur les chunks filtrés (si <= k : on les prend tous, sans recherche)
    3. si < k résultats et complete=True : complément avec les chunks HORS filtre
    `exclude` : chunk_ids à ignorer (dédoublonnage multi-requêtes).
    `stats`   : dict cumulant les appels d'embedding (embed_calls, embed_chars, embed_s).
    """
    exclude = exclude or set()
    inside, outside = [], []
    for pos, doc_id in vs.index_to_docstore_id.items():
        meta = vs.docstore.search(doc_id).metadata
        if meta.get("chunk_id") in exclude:
            continue
        (inside if not filters or _match(meta, filters) else outside).append((pos, doc_id))

    q = None
    def search(ids, n):
        nonlocal q
        if q is None:
            t0 = time.perf_counter()
            q = np.array(vs.embeddings.embed_query(question), dtype="float32")
            if stats is not None:
                stats["embed_calls"] = stats.get("embed_calls", 0) + 1
                stats["embed_chars"] = stats.get("embed_chars", 0) + len(question)
                stats["embed_tok_est"] = round(stats["embed_chars"] / 3.5)   # estimation (non renvoyé par l'API)
                stats["embed_s"] = stats.get("embed_s", 0.0) + time.perf_counter() - t0
        vecs = np.stack([vs.index.reconstruct(pos) for pos, _ in ids])
        dist = ((vecs - q) ** 2).sum(axis=1)            # même métrique que FAISS IndexFlatL2
        return [(ids[i][1], float(dist[i])) for i in np.argsort(dist)[:n]]

    origin = "filtre" if filters else "tous"
    if not inside:
        found = []
    elif filters and len(inside) <= k:                  # pas de recherche : on prend tout
        found = [(d, 0.0) for _, d in inside]
    else:
        found = search(inside, k)
    hits = [(vs.docstore.search(d), s, origin) for d, s in found]

    if filters and complete and len(hits) < k and outside:
        hits += [(vs.docstore.search(d), s, "complément") for d, s in search(outside, k - len(hits))]
    return hits


def forced_hits(cfg: dict, vs: FAISS, doc_types: list | None, not_sure: bool = False) -> tuple[list, list]:
    """Chunks des sources marquées `forcecontext` (datasource_conf.yaml), en entier et dans l'ordre.
      forcecontext: true      -> toujours
      forcecontext: if_filter -> si le doc_type de la source est dans `doc_types` (filtre du routeur ou manuel)
      forcecontext: if_filter_or_not_sure -> idem, OU si le routeur a répondu "not-sure" (pas de filtre doc_type)
    Ignore les filtres (projet, doc_type) : c'est tout l'intérêt.
    Plafond cumulé : retrieval.force_max_chars (défaut 6000) -> sources au-delà ignorées (listées en 2e retour).
    """
    max_chars = cfg["retrieval"].get("force_max_chars", 6000)
    if isinstance(doc_types, str):                      # filtre manuel scalaire : doc_type="fiche"
        doc_types = [doc_types]
    globs = [s["glob"] for s in cfg["sources"]
             if s.get("forcecontext") is True
             or (s.get("forcecontext") in ("if_filter", "if_filter_or_not_sure")
                 and doc_types and s["doc_type"] in doc_types)
             or (s.get("forcecontext") == "if_filter_or_not_sure" and not_sure)]
    if not globs:
        return [], []

    by_source = {}
    for doc_id in vs.index_to_docstore_id.values():
        doc = vs.docstore.search(doc_id)
        sp = doc.metadata.get("source_path", "")
        if any(fnmatch(sp, g) for g in globs):
            by_source.setdefault(sp, []).append(doc)

    hits, skipped, total = [], [], 0
    for sp, docs in sorted(by_source.items()):
        docs.sort(key=lambda d: int(d.metadata["chunk_id"].rsplit("#", 1)[-1]))
        size = sum(len(d.page_content) for d in docs)
        if total + size > max_chars:
            skipped.append(sp)
            continue
        total += size
        hits += [(d, 0.0, "forcé") for d in docs]
    return hits, skipped


def split_content(content) -> tuple[str, str]:
    """(texte de réponse, raisonnement). Modèles classiques : content = str.
    Modèles de raisonnement : content = liste de blocs [{'type': 'thinking', ...}, {'type': 'text', 'text': ...}]."""
    if isinstance(content, str):
        return content, ""
    text, thinking = [], []

    def collect(block, into):
        if isinstance(block, str):
            into.append(block)
        elif isinstance(block, dict):
            if isinstance(block.get("text"), str):
                into.append(block["text"])
            th = block.get("thinking")
            if isinstance(th, str):
                into.append(th)
            elif isinstance(th, list):
                for sub in th:
                    collect(sub, into)

    for block in content or []:
        is_thinking = isinstance(block, dict) and block.get("type") == "thinking"
        collect(block, thinking if is_thinking else text)
    return "\n".join(text).strip(), "\n".join(thinking).strip()


# ================================================================ routage
def get_intent(question: str, cfg: dict, vs: FAISS, llm: ChatMistralAI) -> dict:
    """1 appel LLM court en JSON -> intention validée (valeurs inconnues ignorées)."""
    doc_cat, proj_cat = source_catalog(cfg), project_catalog(vs)
    system = fill(
        cfg["prompts"]["router"],
        doc_types="\n".join(f"- {t} : {d}" for t, d in doc_cat.items()),
        projects="\n".join(f"- {p} : {t}" for p, t in proj_cat.items()),
    )
    default = {"intent": "question", "doc_types": "not-sure", "project_ids": [], "skills": []}
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=question)],
                          temperature=0, max_tokens=300, response_format={"type": "json_object"})
        raw = json.loads(re.search(r"\{.*\}", split_content(resp.content)[0], re.S).group(0))
    except Exception as e:                              # routeur en panne -> pas de filtre
        return {**default, "error": repr(e)}

    intent = str(raw.get("intent") or "").strip().lower()    # gardé tel quel : les intents hors INTENTS -> fixed_responses
    dts = raw.get("doc_types")
    dts = [d for d in dts if d in doc_cat] if isinstance(dts, list) else []
    pids = [int(p) for p in raw.get("project_ids") or [] if str(p).isdigit() and int(p) in proj_cat]
    skills = [str(s).strip() for s in raw.get("skills") or [] if str(s).strip()]
    return {"intent": intent, "doc_types": dts or "not-sure", "project_ids": pids,
            "skills": skills if intent == "annonce" else [],
            "usage": getattr(resp, "usage_metadata", None)}


def intent_to_filters(intent: dict) -> dict | None:
    filters = {}
    if isinstance(intent["doc_types"], list):
        filters["doc_type"] = intent["doc_types"]
    if intent["project_ids"]:
        filters["project_id"] = intent["project_ids"]
    return filters or None


# ================================================================ prompt  (textes dans prompts.yaml)
def format_context(hits) -> str:
    blocks = []
    for i, (doc, *_) in enumerate(hits, 1):
        m = doc.metadata
        proj = f"P{m['project_id']}" if m.get("project_id") is not None else "Général"
        blocks.append(f"[Source {i}] {proj} · {m['label']} · {Path(m['source_path']).name}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_messages(cfg: dict, question: str, hits, mode: str) -> list:
    p = cfg["prompts"]
    system = fill(p["system"][mode], **cfg["candidate"])
    user = fill(p["user_template"], context=format_context(hits), question=question)
    return [SystemMessage(content=system), HumanMessage(content=user)]


# ================================================================ réponses toutes faites
def fixed_response(cfg: dict, intent: str, mode: str, **extra) -> str | None:
    """Réponse de fixed_responses.yaml pour cet intent, ou None si aucune.
    Entrée = texte | liste de textes (tirage au hasard) | {mode: texte | liste}.
    Placeholders : ceux de `candidate` + `extra` (ex. {finish_reason})."""
    entry = cfg.get("fixed_responses", {}).get(intent)
    if isinstance(entry, dict):
        entry = entry.get(mode)
    if isinstance(entry, list):
        entry = random.choice(entry) if entry else None
    return fill(entry, **cfg["candidate"], **extra) if isinstance(entry, str) and entry.strip() else None


# ================================================================ génération
def get_llm(cfg: dict, model: str | None = None) -> ChatMistralAI:
    """LLM de la config (models.llm) ou un autre modèle (ex. models.llm_alternatives), mêmes réglages."""
    m = cfg["models"]
    return ChatMistralAI(model=model or m["llm"], temperature=m["temperature"], max_tokens=m["max_tokens"])


def llm_choices(cfg: dict) -> list:
    """models.llm (défaut) en tête, puis models.llm_alternatives, sans doublon."""
    m = cfg["models"]
    return list(dict.fromkeys([m["llm"], *m.get("llm_alternatives", [])]))


def _tok(usage, key):
    return (usage or {}).get(key)


def ask(question: str, cfg: dict, vs: FAISS, llm: ChatMistralAI, mode: str | None = None,
        filters: dict | None = None, k: int | None = None, use_router: bool | None = None,
        router_llm: ChatMistralAI | None = None) -> dict:
    """Pipeline complet. `filters` explicites -> routeur court-circuité.
    `llm` sert à la génération ; `router_llm` au routage (défaut : `llm`).
    Retourne la réponse + tout ce qu'il faut pour le debug (intent, sources, prompt, métriques)."""
    t_start = time.perf_counter()
    mode = mode or cfg["app"]["mode"]
    rcfg = cfg["retrieval"]
    k = k or rcfg["top_k"]
    use_router = rcfg.get("use_router", True) if use_router is None else use_router
    router_llm = router_llm or llm
    m = {"model": getattr(llm, "model", None), "top_k": k}    # métriques

    intent = None
    if filters is None and use_router:
        t0 = time.perf_counter()
        intent = get_intent(question, cfg, vs, router_llm)
        m["router_s"] = time.perf_counter() - t0
        m["router_in"] = _tok(intent.get("usage"), "input_tokens")
        m["router_out"] = _tok(intent.get("usage"), "output_tokens")
        filters = intent_to_filters(intent)
    routed_filters = filters                     # avant ajustement annonce -> sert à forcecontext: if_filter

    def result(**kw):
        m["total_s"] = time.perf_counter() - t_start
        m["tot_in"] = sum(m.get(x) or 0 for x in ("router_in", "embed_tok_est", "gen_in"))   # 2. totaux
        m["tot_out"] = sum(m.get(x) or 0 for x in ("router_out", "gen_out"))
        for key, v in list(m.items()):
            if key.endswith("_s"):
                m[key] = round(v, 3)
        return {"mode": mode, "intent": intent, "metrics": m, **kw}

    # réponse toute faite : 1) entrée pour l'intent  2) intent sans traitement codé -> default -> fallback
    if intent:
        answer = fixed_response(cfg, intent["intent"], mode)
        if answer is None and intent["intent"] not in INTENTS:
            answer = fixed_response(cfg, "default", mode) or FALLBACK_RESPONSE
        if answer is not None:
            m["model"] = None                                   # pas de génération
            return result(answer=answer, thinking="", response_type="fixed", filters=None, forced_skipped=[],
                          sources=[], messages=[], usage=None)

    t0 = time.perf_counter()
    if intent and intent["intent"] == "annonce" and intent["skills"]:
        # 1 recherche par compétence, dédoublonnée (filtre projet éventuel conservé, pas de filtre doc_type)
        k_skill = rcfg.get("k_per_skill", 3)
        filters = {"project_id": filters["project_id"]} if filters and "project_id" in filters else None
        hits, seen = [], set()
        for skill in intent["skills"]:
            for doc, sc, origin in retrieve(vs, skill, k_skill, filters, exclude=seen, stats=m):
                seen.add(doc.metadata["chunk_id"])
                hits.append((doc, sc, f"skill:{skill}"))
    else:
        hits = retrieve(vs, question, k, filters, stats=m)

    # fichiers forcés : en tête du contexte, hors quota k, sans doublon avec le retrieval
    not_sure = bool(intent) and not isinstance(intent.get("doc_types"), list)   # routeur sans avis sur doc_type
    forced, forced_skipped = forced_hits(cfg, vs, (routed_filters or {}).get("doc_type"), not_sure)
    forced_ids = {d.metadata["chunk_id"] for d, *_ in forced}
    hits = forced + [h for h in hits if h[0].metadata["chunk_id"] not in forced_ids]
    m["retrieval_s"] = time.perf_counter() - t0 - m.get("embed_s", 0.0)   # hors embedding
    m["n_chunks"] = len(hits)

    messages = build_messages(cfg, question, hits, mode)
    t0 = time.perf_counter()
    resp = llm.invoke(messages)
    m["gen_s"] = time.perf_counter() - t0
    usage = getattr(resp, "usage_metadata", None)
    answer, thinking = split_content(resp.content)
    m["gen_in"], m["gen_out"] = _tok(usage, "input_tokens"), _tok(usage, "output_tokens")
    finish = (getattr(resp, "response_metadata", None) or {}).get("finish_reason") or "inconnue"
    m["finish_reason"] = finish                          # stop = fini normalement, length = coupé par max_tokens

    # réponse vide (ex. modèle « thinking » qui épuise max_tokens en raisonnant) ou tronquée -> on l'explique
    if not answer.strip():
        key = "reponse_vide_max_tokens" if finish == "length" else "reponse_vide"
        answer = (fixed_response(cfg, key, mode, finish_reason=finish)
                  or fixed_response(cfg, "reponse_vide", mode, finish_reason=finish)
                  or fill(FALLBACK_EMPTY, finish_reason=finish))
    elif finish == "length":
        answer += "\n\n" + (fixed_response(cfg, "reponse_tronquee", mode, finish_reason=finish)
                             or FALLBACK_TRUNCATED)

    return result(
        answer=answer,
        thinking=thinking,                                  # raisonnement (modèles « thinking »), sinon ""
        response_type="rag",
        filters=filters,
        forced_skipped=forced_skipped,                      # sources forcées écartées (force_max_chars)
        sources=[{"rank": i, "distance": round(float(sc), 3), "origin": o, **d.metadata,
                  "text": d.page_content} for i, (d, sc, o) in enumerate(hits, 1)],
        messages=messages,                                  # prompt final envoyé
        usage=usage,                                        # tokens in/out (génération)
    )
