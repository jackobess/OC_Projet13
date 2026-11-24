"""AskJack — app Streamlit.   Lancement : streamlit run app.py  (depuis la racine du repo)

Prod  : mode / debug lus dans config.yaml (ASKJACK_MODE / ASKJACK_DEBUG absents) -> pas de panneau debug.
Local : ASKJACK_DEBUG=true dans .env -> panneau debug dans la sidebar
        (on/off, mode recrut/candidat, top_k, LLM, journal des questions téléchargeable).
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")                         # local ; absent sur Streamlit Cloud

st.set_page_config(page_title="AskJack", page_icon="💬", layout="centered")

# Streamlit Cloud : les secrets -> variables d'environnement (MISTRAL_API_KEY, HF_TOKEN...)
try:
    for key, val in st.secrets.items():
        if isinstance(val, (str, int, float, bool)):
            os.environ.setdefault(key, str(val))
except Exception:                                  # pas de secrets.toml en local : normal
    pass

from src import rag  # noqa: E402  (après la mise en place de l'environnement)


# ------------------------------------------------------------------ ressources (chargées une fois)
@st.cache_resource
def load_resources():
    cfg = rag.load_config(ROOT)
    return cfg, rag.load_vectorstore(cfg, ROOT)


@st.cache_resource
def load_llm(model: str):
    return rag.get_llm(CFG, model)


CFG, VS = load_resources()
DEBUG_ALLOWED = CFG["app"]["debug"]                # décidé côté serveur uniquement
MAX_Q = CFG["app"].get("max_questions_per_session", 20)
DEFAULT_MODEL = CFG["models"]["llm"]

# ------------------------------------------------------------------ état de session
ss = st.session_state
ss.setdefault("history", [])                       # [{"question", "res"}]
ss.setdefault("log", [])                           # 1 ligne de métriques par question
ss.setdefault("n_questions", 0)
ss.setdefault("pending", None)                     # question exemple cliquée
ss.setdefault("debug_on", DEBUG_ALLOWED)
ss.setdefault("mode", CFG["app"]["mode"])
ss.setdefault("top_k", CFG["retrieval"]["top_k"])
ss.setdefault("model", DEFAULT_MODEL)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.metric("Questions", f"{ss.n_questions} / {MAX_Q}")

    if DEBUG_ALLOWED:
        st.divider()
        st.subheader("🛠️ Debug")
        ss.debug_on = st.toggle("Afficher le debug", value=ss.debug_on)
        ss.mode = st.radio("Mode", rag.MODES, index=rag.MODES.index(ss.mode), horizontal=True)
        ss.top_k = st.slider("top_k", 5, 30, ss.top_k)
        models = rag.llm_choices(CFG)
        ss.model = st.selectbox("LLM (génération)", models, index=models.index(ss.model))
        st.caption(f"Routeur : {DEFAULT_MODEL}")

        if st.button("🗑️ Nouvelle conversation"):
            ss.history, ss.n_questions = [], 0
            st.rerun()

        if ss.log:
            st.divider()
            st.subheader("📊 Journal")
            df_log = pd.DataFrame(ss.log)
            st.dataframe(df_log, height=220)
            st.download_button("⬇️ Télécharger CSV", df_log.to_csv(index=False, sep=";").encode("utf-8-sig"),
                               file_name=f"askjack_log_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")

    st.caption(
        f"Modèle : {ss.model if DEBUG_ALLOWED else DEFAULT_MODEL}  \n"
        f"top_k : {ss.top_k if DEBUG_ALLOWED else CFG['retrieval']['top_k']}  \n"
        f"max_tokens : {CFG['models']['max_tokens']}"
    )

DEBUG = DEBUG_ALLOWED and ss.debug_on
MODE = ss.mode if DEBUG_ALLOWED else CFG["app"]["mode"]

# ------------------------------------------------------------------ en-tête
ui = rag.ui_texts(CFG, MODE)
st.title(ui.get("title", "AskJack"))
if ui.get("subtitle"):
    st.caption(ui["subtitle"])
if ui.get("description"):
    st.write(ui["description"])

if ui.get("examples") and not ss.history:
    cols = st.columns(len(ui["examples"]))
    for col, ex in zip(cols, ui["examples"]):
        if col.button(ex):
            ss.pending = ex


# ------------------------------------------------------------------ affichage d'un échange
def source_label(s: dict) -> str:
    proj = f"P{s['project_id']}" if s.get("project_id") is not None else "Général"
    return f"{proj} · {s['label']} · {Path(s['source_path']).name}"


def show_debug(res: dict):
    with st.expander("🛠️ Debug"):
        st.write(f"**Type de réponse :** {res.get('response_type')} · **mode :** {res['mode']}")
        if res.get("intent"):
            st.write("**Intent :**", {k: v for k, v in res["intent"].items() if k != "usage"})
        st.write("**Filtres :**", res.get("filters"))
        if res.get("forced_skipped"):
            st.write("**Forcés écartés (taille) :**", res["forced_skipped"])
        st.write("**Métriques :**")
        st.dataframe(pd.DataFrame([res["metrics"]]), hide_index=True)
        if res["sources"]:
            st.write("**Chunks :**")
            st.dataframe(pd.DataFrame(res["sources"])[["rank", "origin", "distance", "doc_type",
                                                       "project_id", "chunk_id"]],
                         hide_index=True)
        if res.get("thinking"):
            st.write("**Raisonnement du modèle :**")
            st.code(res["thinking"], language=None)
        for msg in res["messages"]:
            st.write(f"**Prompt — {msg.type}**")
            st.code(msg.content, language=None)


def show_exchange(question: str, res: dict):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.markdown(res["answer"])
        if res.get("response_type") == "rag" and res["sources"]:
            with st.expander("📚 Sources"):
                for lbl in dict.fromkeys(source_label(s) for s in res["sources"]):
                    st.write("-", lbl)
        if DEBUG:
            show_debug(res)


for item in ss.history:
    show_exchange(item["question"], item["res"])

# ------------------------------------------------------------------ saisie
limit_reached = ss.n_questions >= MAX_Q
typed = st.chat_input("Votre question…" if MODE == "recrut" else "Ta question ou une annonce…",
                      disabled=limit_reached)
if limit_reached:
    st.info(f"Limite de {MAX_Q} questions atteinte pour cette session. Merci de votre visite !")

question = typed or ss.pending
ss.pending = None

if question and not limit_reached:
    ss.n_questions += 1
    t0 = time.perf_counter()
    try:
        with st.spinner("AskJack réfléchit…"):
            res = rag.ask(question, CFG, VS, load_llm(ss.model if DEBUG_ALLOWED else DEFAULT_MODEL),
                          mode=MODE, k=ss.top_k if DEBUG_ALLOWED else None,
                          router_llm=load_llm(DEFAULT_MODEL))
    except Exception as e:
        res = {"answer": "⚠️ Une erreur est survenue, merci de réessayer dans un instant.",
               "response_type": "error", "mode": MODE, "intent": None, "filters": None,
               "forced_skipped": [], "sources": [], "messages": [],
               "metrics": {"error": repr(e), "total_s": round(time.perf_counter() - t0, 3)}}

    ss.history.append({"question": question, "res": res})
    ss.log.append({"horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mode": MODE,
                   "question": question, "response_type": res.get("response_type"),
                   "intent": (res.get("intent") or {}).get("intent"), **res["metrics"]})
    st.rerun()
