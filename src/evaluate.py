"""AskJack — évaluation automatique sur un jeu de questions (tests/eval_questions.yaml).

Contrôles par test :
  - comportement (pass/fail) : intent attendu, type de réponse (rag | fixed), abstention si attendue
  - contenu (score 0-1)      : projets attendus cités + mots-clés attendus présents (rappel)
  - signal d'hallucination    : projets cités en trop (non pénalisé, affiché)
Un test passe si tout le comportement est OK et score >= pass_threshold (ou pas de score de contenu).
"""
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src import rag


def norm(text: str) -> str:
    """minuscules, sans accents, espaces normalisés -> comparaison tolérante"""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text.lower())


def cited_projects(answer: str) -> set:
    """n° de projets cités : P5, P 5, projet 5, projets 5..."""
    return {int(n) for n in re.findall(r"\b(?:p|projets?\s*)\s?(\d{1,2})\b", norm(answer))}


def score_test(test: dict, res: dict, settings: dict) -> dict:
    answer = norm(res["answer"])
    row = {}

    # ---- comportement
    checks = []
    if "expected_intent" in test:
        got = (res.get("intent") or {}).get("intent")
        row["intent_ok"] = got == test["expected_intent"]
        checks.append(row["intent_ok"])
    if "expected_response" in test:
        row["response_ok"] = res.get("response_type") == test["expected_response"]
        checks.append(row["response_ok"])
    if test.get("must_abstain"):
        row["abstain_ok"] = any(norm(m) in answer for m in settings.get("abstain_markers", []))
        checks.append(row["abstain_ok"])

    # ---- contenu
    scores = []
    if test.get("expected_projects"):
        expected, cited = set(test["expected_projects"]), cited_projects(res["answer"])
        row["proj_missing"] = sorted(expected - cited)
        row["proj_extra"] = sorted(cited - expected)          # signal d'hallucination
        row["proj_score"] = round(len(expected & cited) / len(expected), 2)
        scores.append(row["proj_score"])
    if test.get("expected_keywords"):
        groups = [g if isinstance(g, list) else [g] for g in test["expected_keywords"]]
        found = lambda v: re.search(r"(?<!\w)" + re.escape(norm(v)), answer)   # début de mot : "uk" ≠ "bulk"
        missing = [g[0] for g in groups if not any(found(v) for v in g)]
        row["kw_missing"] = missing
        row["kw_score"] = round(1 - len(missing) / len(groups), 2)
        scores.append(row["kw_score"])

    row["score"] = round(sum(scores) / len(scores), 2) if scores else None
    content_ok = row["score"] is None or row["score"] >= settings.get("pass_threshold", 0.8)
    row["passed"] = all(checks) and content_ok
    return row


def _mean(series: pd.Series, ndigits: int):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(s.mean()), ndigits) if len(s) else None


def run_eval(tests_path, cfg: dict, vs, llm, router_llm=None, k: int | None = None,
             out_dir=None) -> tuple[pd.DataFrame, dict]:
    """Passe tous les tests dans rag.ask -> (DataFrame des résultats, résumé). Exporte un CSV si out_dir."""
    with open(tests_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    settings, rows = spec.get("settings", {}), []

    for test in spec["tests"]:
        try:
            res = rag.ask(test["question"], cfg, vs, llm, mode=test.get("mode", "recrut"), k=k,
                          router_llm=router_llm)
        except Exception as e:
            res = {"answer": f"ERREUR : {e!r}", "response_type": "error", "intent": None, "metrics": {}}
        m = res.get("metrics", {})
        rows.append({
            "id": test["id"], "question": test["question"], "mode": test.get("mode", "recrut"),
            "intent": (res.get("intent") or {}).get("intent"), "response_type": res.get("response_type"),
            **score_test(test, res, settings),
            "tot_in": m.get("tot_in"), "tot_out": m.get("tot_out"), "total_s": m.get("total_s"),
            "model": m.get("model"), "top_k": m.get("top_k"),
            "answer": res["answer"], "reference": test.get("reference", ""),
            "note_manuelle": "",                             # à remplir : correct / partiel / faux
        })

    df = pd.DataFrame(rows)
    summary = {
        "tests": len(df),
        "passed": int(df["passed"].sum()),
        "pass_rate": round(float(df["passed"].mean()), 2),
        "mean_score": _mean(df["score"], 2),
        "mean_latency_s": _mean(df["total_s"], 2),
        "mean_tot_in": _mean(df["tot_in"], 0),
        "mean_tot_out": _mean(df["tot_out"], 0),
        "model": getattr(llm, "model", None), "top_k": k or cfg["retrieval"]["top_k"],
    }
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"eval_{datetime.now():%Y%m%d_%H%M}_{summary['model']}_k{summary['top_k']}.csv"
        df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
        summary["csv"] = str(path)
    return df, summary
