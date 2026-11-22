"""
TeluguShield AI — Streamlit Prototype (Final Bypass Version)
"""

import os, json, warnings, time, textwrap
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from captum.attr import IntegratedGradients
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# Sarvam AI import
try:
    from sarvamai import SarvamAI
    SARVAM_AVAILABLE = True
except ImportError:
    SARVAM_AVAILABLE = False

from config import (
    MBERT_MODEL_ID, MURIL_LOCAL_PATH, TELUGUBERT_LOCAL_PATH,
    TEST_DATA_PATH, TEXT_COLUMN, LABEL_COLUMN,
    LABEL_MAP, ID2LABEL, MAX_LENGTH,
    IG_STEPS, LIME_SAMPLES, EVAL_CACHE_PATH,
)

# =========================================================
st.set_page_config(page_title="TeluguShield AI", page_icon="🛡️", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SARVAM_API_KEY = st.secrets.get("SARVAM_API_KEY", "")

# =========================================================
# LANGUAGE DICTIONARY (Hindi / English)
# =========================================================
LANG_DICT = {
    "English": {
        "title": "🛡️ TeluguShield AI",
        "subtitle": "Telugu / code-mixed hate speech detection — mBERT + MuRIL + TeluguBERT",
        "detect_tab": "🔍 Detect",
        "eval_tab": "📊 Evaluation",
        "about_tab": "ℹ️ About",
        "enter_comment": "Enter Telugu / code-mixed comment",
        "comment_label": "Comment",
        "detect_btn": "🚀 Detect Hate Speech",
        "trans_title": "🌐 English Translation (for non-Telugu speakers)",
        "individual_preds": "### Individual Predictions",
        "ensemble": "### Ensemble",
        "majority": "Majority Voting",
        "soft": "Soft Voting",
        "ig_title": "### 🔴🟢 Token Highlighting (IG-based)",
        "ig_caption": "Red = supports predicted class. Green = opposes. Live Integrated Gradients.",
        "ig_chart_title": "### 📊 Integrated Gradients — Top Tokens",
        "lime_title": "### 🍋 LIME Explanations",
        "hate_contrib": "**HATE contributions**",
        "non_hate_contrib": "**NON-HATE contributions**",
        "eval_title": "Independent Test Set Evaluation",
        "run_eval": "▶️ Run Fresh Evaluation",
        "load_cache": "📦 Load Cached",
        "comp_title": "### 📋 Comparison",
        "chart_title": "### 📊 Accuracy vs Macro F1",
        "conf_title": "### 🧩 Confusion Matrices",
        "rep_title": "### 📄 Classification Reports"
    },
    "Hindi": {
        "title": "🛡️ तेलुगुशील्ड AI",
        "subtitle": "तेलुगु / कोड-मिक्स्ड हेट स्पीच डिटेक्शन — mBERT + MuRIL + TeluguBERT",
        "detect_tab": "🔍 पहचानें",
        "eval_tab": "📊 मूल्यांकन",
        "about_tab": "ℹ️ जानकारी",
        "enter_comment": "तेलुगु / कोड-मिक्स्ड कमेंट दर्ज करें",
        "comment_label": "कमेंट",
        "detect_btn": "🚀 हेट स्पीच पहचानें",
        "trans_title": "🌐 अंग्रेजी अनुवाद (गैर-तेलुगु लोगों के लिए)",
        "individual_preds": "### व्यक्तिगत भविष्यवाणियां",
        "ensemble": "### एन्सेम्बल (सामूहिक निर्णय)",
        "majority": "बहुमत वोटिंग",
        "soft": "सॉफ्ट वोटिंग",
        "ig_title": "### 🔴🟢 टोकन हाइलाइटिंग (IG-आधारित)",
        "ig_caption": "लाल = भविष्यवाणी का समर्थन करता है। हरा = विरोध करता है।",
        "ig_chart_title": "### 📊 इंटीग्रेटेड ग्रेडिएंट्स — टॉप टोकन",
        "lime_title": "### 🍋 LIME स्पष्टीकरण",
        "hate_contrib": "**हेट (HATE) योगदान**",
        "non_hate_contrib": "**नॉन-हेट (NON-HATE) योगदान**",
        "eval_title": "स्वतंत्र टेस्ट सेट मूल्यांकन",
        "run_eval": "▶️ नया मूल्यांकन चलाएं",
        "load_cache": "📦 कैश लोड करें",
        "comp_title": "### 📋 तुलना",
        "chart_title": "### 📊 सटीकता vs मैक्रो F1",
        "conf_title": "### 🧩 कन्फ्यूजन मैट्रिक्स",
        "rep_title": "### 📄 वर्गीकरण रिपोर्ट"
    }
}

# Sidebar for language selection
lang_choice = st.sidebar.selectbox("🌐 Select Language / भाषा चुनें", ["English", "Hindi"])
T = LANG_DICT[lang_choice]

# =========================================================
# MODEL LOADING (AutoConfig Bypass)
# =========================================================
def _resolve_model_dir(path):
    """Return a real local folder that contains config.json.
    Accepts a folder path OR a path to config.json; also looks one level down."""
    p = os.path.abspath(str(path).strip().strip('"'))
    if os.path.isfile(p):                      # user gave .../config.json
        p = os.path.dirname(p)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Folder not found: {p}")
    if os.path.isfile(os.path.join(p, "config.json")):
        return p
    for sub in sorted(os.listdir(p)):          # e.g. TeluguBERT/Best_Model
        sp = os.path.join(p, sub)
        if os.path.isdir(sp) and os.path.isfile(os.path.join(sp, "config.json")):
            return sp
    raise FileNotFoundError(f"config.json not found inside: {p}")

@st.cache_resource(show_spinner=False)
def _load_one(path, name):
    try:
        p = str(path).strip().strip('"')
        if name == "mBERT" or not os.path.exists(p):
            # Hugging Face repo ID (mBERT, and MuRIL / TeluguBERT on the cloud)
            tok = AutoTokenizer.from_pretrained(p)
            mdl = AutoModelForSequenceClassification.from_pretrained(p)
        else:
            # Local folder (laptop): pass the FOLDER (never the config.json file)
            model_dir = _resolve_model_dir(p)
            tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)

        mdl.to(DEVICE).eval()
        return {"name": name, "tokenizer": tok, "model": mdl, "ok": True, "error": None}
    except Exception as e:
        return {"name": name, "tokenizer": None, "model": None, "ok": False, "error": str(e)}

@st.cache_resource(show_spinner=False)
def load_all_models():
    return [
        _load_one(MBERT_MODEL_ID, "mBERT"),
        _load_one(MURIL_LOCAL_PATH, "MuRIL"),
        _load_one(TELUGUBERT_LOCAL_PATH, "TeluguBERT"),
    ]

# =========================================================
# INFERENCE & LABEL FIX
# =========================================================
def _needs_flip(model):
    try:
        id2label = getattr(model.config, "id2label", {}) or {}
        if isinstance(id2label, list):
            return len(id2label) > 0 and "hate" in str(id2label[0]).lower() and "non" not in str(id2label[0]).lower()
        elif isinstance(id2label, dict):
            t = {str(k): str(v).lower() for k, v in id2label.items()}
            return "0" in t and "hate" in t["0"] and "non" not in t["0"]
    except Exception:
        pass
    return False

@torch.no_grad()
def predict_proba(mdict, texts):
    tok, mdl = mdict["tokenizer"], mdict["model"]
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)
    probs = F.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()
    
    if _needs_flip(mdl):
        probs = probs[:, ::-1]
    if probs.shape[1] == 1:
        p = probs[:, 0]
        probs = np.stack([1 - p, p], axis=1)
    return probs

def predict_labels(mdict, texts):
    p = predict_proba(mdict, texts)
    return p.argmax(1), p

# =========================================================
# ENSEMBLES
# =========================================================
def majority_vote(preds_list):
    s = np.stack(preds_list, 1).sum(1)
    return (s >= 2).astype(int)

def soft_vote(probs_list):
    avg = np.mean(np.stack(probs_list, 0), 0)
    return avg.argmax(1), avg

# =========================================================
# INTEGRATED GRADIENTS & LIME
# =========================================================
def ig_explain(mdict, text, target_class):
    tok, mdl = mdict["tokenizer"], mdict["model"]
    flip = _needs_flip(mdl)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    pad  = tok.pad_token_id if tok.pad_token_id is not None else 0
    base = torch.full_like(ids, pad)

    emb = mdl.get_input_embeddings()

    def fwd(e):
        out = mdl(inputs_embeds=e, attention_mask=mask).logits
        if flip: out = torch.flip(out, dims=[1])
        return F.softmax(out, -1)

    ig = IntegratedGradients(fwd)
    attrs, _ = ig.attribute(
        inputs=emb(ids), baselines=emb(base),
        target=target_class, n_steps=IG_STEPS, return_convergence_delta=True,
    )
    scores = attrs.sum(-1).squeeze(0).detach().cpu().numpy()
    toks = tok.convert_ids_to_tokens(ids[0].cpu().numpy())
    out = []
    for t, s in zip(toks, scores):
        if t in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"): continue
        out.append((t.replace("##", ""), float(s)))
    return out

@st.cache_resource(show_spinner=False)
def _lime_explainer():
    return LimeTextExplainer(class_names=["NON-HATE", "HATE"])

def lime_explain(mdict, text, n=10):
    exp = _lime_explainer()
    def pred(xs): return predict_proba(mdict, list(xs))
    return exp.explain_instance(text, pred, num_features=n, num_samples=LIME_SAMPLES, labels=(0, 1))

def highlight_html(scores):
    if not scores: return "<i>No tokens.</i>"
    m = max(abs(s) for _, s in scores) or 1.0
    parts = []
    for t, s in scores:
        a = min(abs(s) / m, 1.0)
        bg = f"rgba(220,50,50,{0.15+0.55*a:.2f})" if s >= 0 else f"rgba(40,160,80,{0.15+0.55*a:.2f})"
        parts.append(f'<span style="background:{bg};padding:2px 4px;border-radius:4px;margin:1px;display:inline-block;">{t}</span>')
    return " ".join(parts)

# =========================================================
# METRICS & EVALUATION
# =========================================================
def _metrics(y, p):
    return {
        "accuracy":    float(accuracy_score(y, p)),
        "precision":   float(precision_score(y, p, average="macro", zero_division=0)),
        "recall":      float(recall_score(y, p, average="macro", zero_division=0)),
        "macro_f1":    float(f1_score(y, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
    }

@st.cache_data(show_spinner=False)
def run_eval(_models, path, tcol, lcol):
    if not os.path.exists(path):
        return None, 0
    if path.endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
        
    df = df.dropna(subset=[tcol, lcol]).reset_index(drop=True)
    texts = df[tcol].astype(str).tolist()
    y = np.array([LABEL_MAP.get(str(v).lower().strip(), -1) for v in df[lcol]])
    keep = y != -1
    texts = [t for t, k in zip(texts, keep) if k]
    y = y[keep]

    all_preds, all_probs, res = [], [], {}
    for m in _models:
        if not m["ok"]:
            res[m["name"]] = {"error": m["error"]}
            continue
        p, pr = predict_labels(m, texts)
        all_preds.append(p); all_probs.append(pr)
        res[m["name"]] = {
            "metrics": _metrics(y, p),
            "confusion": confusion_matrix(y, p).tolist(),
            "report": classification_report(y, p, target_names=["NON-HATE","HATE"], output_dict=True, zero_division=0),
        }
    if len(all_preds) == 3:
        for name, preds in [
            ("Majority Voting", majority_vote(all_preds)),
            ("Soft Voting",     soft_vote(all_probs)[0]),
        ]:
            res[name] = {
                "metrics": _metrics(y, preds),
                "confusion": confusion_matrix(y, preds).tolist(),
                "report": classification_report(y, preds, target_names=["NON-HATE","HATE"], output_dict=True, zero_division=0),
            }
    return res, int(len(y))

# =========================================================
# RESEARCH & INSIGHTS (teacher-facing, separate from live detection)
# All numbers below are the saved experimental results — do not edit.
# =========================================================
try:
    import plotly.graph_objects as pgo
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

R_MODELS = ["MuRIL", "TeluguBERT", "XLM-RoBERTa"]
R_PROPS = ["25%", "50%", "75%", "100%"]
R_SAMPLES = {"25%": 780, "50%": 1560, "75%": 2340, "100%": 3120}
R_VAL_F1 = {                      # validation Macro F1 (%)
    "MuRIL":       [68.69, 71.57, 76.04, 74.26],
    "TeluguBERT":  [68.31, 77.08, 76.68, 76.14],
    "XLM-RoBERTa": [70.53, 72.40, 75.67, 69.99],
}
R_TEST = {                        # independent test results (%)
    "MuRIL":       {"Accuracy": 66.67, "Macro F1": 66.62},
    "TeluguBERT":  {"Accuracy": 68.30, "Macro F1": 68.28},
    "XLM-RoBERTa": {"Accuracy": 65.03, "Macro F1": 64.63},
}
R_COLORS = {"MuRIL": "#7c3aed", "TeluguBERT": "#0d9488", "XLM-RoBERTa": "#f59e0b"}
R_PURPLE = "#6d28d9"

R_CSS = """
<style>
@keyframes rfade {from {opacity:0; transform:translateY(8px);} to {opacity:1; transform:none;}}
.r-hero {background:linear-gradient(135deg,#f5f3ff 0%,#ffffff 100%); border:1px solid #ddd6fe;
         border-left:6px solid #6d28d9; border-radius:14px; padding:26px 30px; margin-bottom:18px;
         animation:rfade .6s ease-out;}
.r-hero h1 {color:#4c1d95; font-size:2.0rem; margin:0 0 6px 0; line-height:1.25;}
.r-hero p {color:#5b21b6; font-size:1.15rem; margin:0; font-weight:500;}
.r-card {background:#ffffff; border:1px solid #e9e5f5; border-radius:12px; padding:16px 18px;
         box-shadow:0 2px 8px rgba(109,40,217,.08); animation:rfade .6s ease-out; height:100%;}
.r-card .lbl {color:#6b21a8; font-size:.9rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em;}
.r-card .val {color:#2e1065; font-size:1.9rem; font-weight:700; margin:4px 0;}
.r-card .sub {color:#444; font-size:.95rem;}
.r-note {background:#faf5ff; border-left:4px solid #8b5cf6; border-radius:8px; padding:12px 16px;
         color:#2e1065; font-size:1.02rem; margin:8px 0 18px 0;}
.r-warn {background:#fffbeb; border-left:4px solid #f59e0b; border-radius:8px; padding:12px 16px;
         color:#451a03; font-size:1.02rem; margin:8px 0 18px 0;}
.r-sec {color:#4c1d95; font-size:1.6rem; font-weight:700; margin:30px 0 8px 0;
        border-bottom:2px solid #ede9fe; padding-bottom:6px;}
.r-body {color:#1f2937; font-size:1.05rem; line-height:1.7;}
.r-body li {margin-bottom:6px;}
</style>
"""

def _card(label, value, sub=""):
    return (f'<div class="r-card"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')

def _sec(title):
    st.markdown(f'<div class="r-sec">{title}</div>', unsafe_allow_html=True)

def _note(text, warn=False):
    st.markdown(f'<div class="{"r-warn" if warn else "r-note"}">{text}</div>', unsafe_allow_html=True)

def _plotly_layout(fig, title, ytitle, ymin=60, ymax=80, xtitle=None):
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="#2e1065")),
        xaxis_title=xtitle, yaxis_title=ytitle,
        yaxis=dict(range=[ymin, ymax], gridcolor="#ede9fe", title_font=dict(size=15), tickfont=dict(size=14)),
        xaxis=dict(title_font=dict(size=15), tickfont=dict(size=14)),
        legend=dict(font=dict(size=14), orientation="h", y=-0.22),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", font=dict(color="#1f2937"),
        height=430, margin=dict(l=40, r=20, t=60, b=40), hovermode="x unified",
    )
    return fig

def render_research_tab():
    st.markdown(R_CSS, unsafe_allow_html=True)

    # ---------- Title ----------
    st.markdown(
        '<div class="r-hero"><h1>Telugu Hate Speech Detection – Research &amp; Experimental Analysis</h1>'
        '<p>Data Sufficiency Study and Comparative Evaluation of Transformer Models</p></div>',
        unsafe_allow_html=True,
    )

    df_val = pd.DataFrame(R_VAL_F1, index=R_PROPS)
    best = {m: (df_val[m].idxmax(), float(df_val[m].max())) for m in R_MODELS}

    # ---------- Key metric cards ----------
    c = st.columns(4)
    c[0].markdown(_card("Training pool", "3,120", "labelled samples"), unsafe_allow_html=True)
    c[1].markdown(_card("Validation set", "781", "same set for every run"), unsafe_allow_html=True)
    c[2].markdown(_card("Independent test", "489", "250 non-hate · 239 hate"), unsafe_allow_html=True)
    c[3].markdown(_card("Models compared", "3", "MuRIL · TeluguBERT · XLM-R"), unsafe_allow_html=True)

    # ---------- 1. Objective ----------
    _sec("1. Research Objective")
    st.markdown(
        '<div class="r-body">To investigate how different quantities of training data influence the '
        'performance of pretrained transformer models for Telugu hate speech detection.<br><br>'
        'The experiment investigates:<ul>'
        '<li>whether model performance improves consistently with increasing training data,</li>'
        '<li>whether performance varies across architectures, and</li>'
        '<li>whether using the entire available training dataset always provides the best validation performance.</li>'
        '</ul></div>', unsafe_allow_html=True)

    # ---------- 2. Methodology ----------
    _sec("2. Experimental Methodology")
    st.markdown(
        '<div class="r-body"><b>Models:</b> MuRIL, TeluguBERT, XLM-RoBERTa. All three are pretrained '
        'transformer models fine-tuned for binary Telugu hate speech classification (HATE / NON-HATE).</div>',
        unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.markdown(_card("Training", "3,120 samples",
                      "Data-sufficiency proportions: 25% = 780 · 50% = 1,560 · 75% = 2,340 · 100% = 3,120"),
                unsafe_allow_html=True)
    m2.markdown(_card("Validation", "781 samples",
                      "One fixed validation set used to compare all proportions. Used for model selection."),
                unsafe_allow_html=True)
    m3.markdown(_card("Independent test", "489 samples",
                      "Held-out set (250 non-hate, 239 hate), kept separate from training and validation."),
                unsafe_allow_html=True)
    st.write("")
    st.dataframe(
        pd.DataFrame({"Training proportion": R_PROPS,
                      "Training samples": [f"{R_SAMPLES[p]:,}" for p in R_PROPS]}),
        hide_index=True, use_container_width=True)
    _note("<b>Training</b> = data the model learns from. <b>Validation</b> = fixed set used to compare "
          "the data-proportion experiments. <b>Independent test</b> = unseen data used only for the final "
          "saved evaluation. Section 3 reports validation results; Section 4 reports test results.")

    # ---------- 3. Data sufficiency (validation) ----------
    _sec("3. Data Sufficiency Results (Validation Macro F1)")
    _note("These are <b>validation</b> scores on the 781-sample validation set — "
          "<b>not</b> independent test results.", warn=True)

    # best-score cards
    bc = st.columns(3)
    for col, m in zip(bc, R_MODELS):
        p, v = best[m]
        col.markdown(_card(f"{m} · best validation", f"{v:.2f}%",
                           f"highest observed at {p} training data ({R_SAMPLES[p]:,} samples)"),
                     unsafe_allow_html=True)
    st.write("")

    # 3a. line chart
    st.markdown("#### Macro F1 vs training data (line chart)")
    if PLOTLY_OK:
        fig = pgo.Figure()
        for m in R_MODELS:
            fig.add_trace(pgo.Scatter(
                x=R_PROPS, y=R_VAL_F1[m], mode="lines+markers", name=m,
                line=dict(color=R_COLORS[m], width=3), marker=dict(size=10),
                hovertemplate="%{y:.2f}%"))
            p, v = best[m]
            fig.add_trace(pgo.Scatter(
                x=[p], y=[v], mode="markers", showlegend=False, hoverinfo="skip",
                marker=dict(size=19, symbol="star", color=R_COLORS[m], line=dict(color="#2e1065", width=1.5))))
        _plotly_layout(fig, "Validation Macro F1 by training-data proportion", "Validation Macro F1 (%)",
                       xtitle="Training data used")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df_val, height=380)
        st.caption("Install plotly (pip install plotly) for the interactive version with highlighted peaks.")
    _note("<b>What this shows:</b> how each model's validation Macro F1 changes as more training data is used. "
          "The lines are not strictly increasing, and the ★ marks each model's best observed score.")

    # 3b. grouped bar chart
    st.markdown("#### Model comparison at each data proportion (grouped bar chart)")
    if PLOTLY_OK:
        fig = pgo.Figure()
        for m in R_MODELS:
            fig.add_trace(pgo.Bar(x=R_PROPS, y=R_VAL_F1[m], name=m, marker_color=R_COLORS[m],
                                 text=[f"{v:.2f}" for v in R_VAL_F1[m]], textposition="outside",
                                 hovertemplate="%{y:.2f}%"))
        _plotly_layout(fig, "Validation Macro F1 – models compared per proportion", "Validation Macro F1 (%)",
                       ymin=60, ymax=82, xtitle="Training data used")
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df_val, height=380)
    _note("<b>What this shows:</b> which architecture scores higher at each amount of training data. "
          "The ranking of models changes from one proportion to another.")

    # 3c. table
    st.markdown("#### Results table (validation Macro F1, %)")
    tbl = df_val.copy()
    tbl.index.name = "Training data"
    def _hl(col):
        return ["background-color:#ede9fe; color:#2e1065; font-weight:700" if v == col.max() else ""
                for v in col]
    st.dataframe(tbl.style.apply(_hl, axis=0).format("{:.2f}%"), use_container_width=True)
    _note("Highlighted cell = each model's highest observed validation Macro F1 "
          f"(MuRIL: {best['MuRIL'][1]:.2f}% at {best['MuRIL'][0]}, "
          f"TeluguBERT: {best['TeluguBERT'][1]:.2f}% at {best['TeluguBERT'][0]}, "
          f"XLM-RoBERTa: {best['XLM-RoBERTa'][1]:.2f}% at {best['XLM-RoBERTa'][0]}).")

    # ---------- 4. Independent test ----------
    _sec("4. Independent Test Results (saved final evaluation)")
    _note("These are <b>independent test</b> results on 489 unseen samples (250 non-hate, 239 hate). "
          "They are separate from the data-proportion validation experiment above.", warn=True)
    df_test = pd.DataFrame(R_TEST).T[["Accuracy", "Macro F1"]]
    if PLOTLY_OK:
        fig = pgo.Figure()
        for metric, colr in [("Accuracy", "#7c3aed"), ("Macro F1", "#c4b5fd")]:
            fig.add_trace(pgo.Bar(x=R_MODELS, y=df_test[metric], name=metric, marker_color=colr,
                                 text=[f"{v:.2f}%" for v in df_test[metric]], textposition="outside",
                                 hovertemplate="%{y:.2f}%"))
        _plotly_layout(fig, "Independent test set – Accuracy vs Macro F1", "Score (%)",
                       ymin=55, ymax=75)
        fig.update_layout(barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df_test, height=380)
    t_show = df_test.copy(); t_show.index.name = "Model"
    st.dataframe(t_show.style.format("{:.2f}%"), use_container_width=True)
    _note("<b>What this shows:</b> final performance on data never used in training or validation. "
          "TeluguBERT has the highest saved test Accuracy (68.30%) and Macro F1 (68.28%) of the three models.")

    # ---------- 5. Findings ----------
    _sec("5. Research Findings")
    st.markdown(
        '<div class="r-body"><ul>'
        '<li>Increasing training data did <b>not</b> produce a strictly monotonic increase in validation '
        'Macro F1 for every model.</li>'
        '<li>MuRIL achieved its highest validation Macro F1 at <b>75%</b> training data.</li>'
        '<li>TeluguBERT achieved its highest validation Macro F1 at <b>50%</b> training data.</li>'
        '<li>XLM-RoBERTa achieved its highest validation Macro F1 at <b>75%</b> training data.</li>'
        '<li>At 100% training data, all three models scored lower than their respective best observed '
        'validation scores.</li>'
        '<li>TeluguBERT achieved the highest saved independent test Accuracy and Macro F1 among the three models.</li>'
        '</ul>'
        'Together, these findings demonstrate <b>model-dependent performance variation with training-data '
        'quantity</b>. They do not show that less data is universally better.</div>',
        unsafe_allow_html=True)

    # ---------- 6. Limitations ----------
    _sec("6. Research Limitations")
    st.markdown(
        '<div class="r-body"><ol>'
        '<li>The experiment uses one Telugu hate-speech dataset, limiting generalization to other domains and platforms.</li>'
        '<li>The available training pool contains 3,120 samples.</li>'
        '<li>Validation scores fluctuate across training-data proportions; the experiment does not establish the '
        'exact cause of those fluctuations.</li>'
        '<li>Unless repeated runs with multiple random seeds were performed, sampling variability and statistical '
        'significance cannot be established.</li>'
        '<li>Further external evaluation is needed across Telugu dialects, spelling variations, code-mixed '
        'comments, and different social-media platforms.</li>'
        '<li>Classification performance does not independently establish fairness or linguistic correctness.</li>'
        '</ol></div>', unsafe_allow_html=True)

    # ---------- 7. Conclusion ----------
    _sec("7. Research Conclusion")
    _note("The experiment shows that the relationship between training-data quantity and model performance is "
          "not necessarily linear or identical across transformer architectures. The optimal observed validation "
          "data proportion differed across the three models. Therefore, data sufficiency should be examined "
          "empirically rather than assuming that using the maximum available data will always produce the best "
          "validation score.")


# =========================================================
# UI
# =========================================================
st.title(T["title"])
st.caption(T["subtitle"])

main_tab, research_tab = st.tabs(["🛡️ Hate Speech Detection", "🔬 Research & Insights"])

# Research tab is rendered first so it stays available even if a model fails to load
with research_tab:
    render_research_tab()

with main_tab:
    with st.spinner("Loading 3 models… ( may take a few minutes)"):
        MODELS = load_all_models()

    ok = [m for m in MODELS if m["ok"]]
    bad = [m for m in MODELS if not m["ok"]]
    for m in bad: st.error(f"❌ {m['name']} not loaded: {m['error']}")
    if not ok: st.stop()
    st.success(f"✅ {len(ok)}/3 models loaded on **{DEVICE.type.upper()}**")

    t1, t2, t3 = st.tabs([T["detect_tab"], T["eval_tab"], T["about_tab"]])

    # =========================================================
    # TAB 1 — DETECT
    # =========================================================
    with t1:
        st.subheader(T["enter_comment"])
        txt = st.text_area(T["comment_label"], height=120, placeholder="ఉదా: ఈ వీడియో చాలా బాగుంది...")
        go = st.button(T["detect_btn"], type="primary")

        if go and txt.strip():
            text = txt.strip()
        
            # --- TRANSLATION FEATURE (Sarvam AI) ---
            with st.expander(T["trans_title"], expanded=True):
                if not SARVAM_AVAILABLE:
                    st.error("SarvamAI library not installed. Please run: pip install sarvamai")
                else:
                    try:
                        client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
                        response = client.text.translate(
                            input=text,
                            source_language_code="auto",
                            target_language_code="en-IN"
                        )
                        st.info(f"**English:** {response.translated_text}")
                    except Exception as e:
                        st.warning(f"Translation failed: {e}")
            # ---------------------------

            st.markdown(T["individual_preds"])
            cols = st.columns(3)
            per = []
            for i, m in enumerate(MODELS):
                if not m["ok"]:
                    cols[i].error(f"{m['name']}: not loaded"); continue
                pr = predict_proba(m, [text])[0]
                p = int(pr.argmax()); lab = ID2LABEL[p]; conf = float(pr[p])
                per.append({"name": m["name"], "pred": p, "probs": pr})
                with cols[i]:
                    if lab == "HATE":
                        st.error(f"**{m['name']}**\n\n### 🔴 {lab}\nConf: **{conf:.1%}**")
                    else:
                        st.success(f"**{m['name']}**\n\n### 🟢 {lab}\nConf: **{conf:.1%}**")
                    st.caption(f"HATE={pr[1]:.3f} | NON-HATE={pr[0]:.3f}")

            if len(per) == 3:
                st.markdown(T["ensemble"])
                c1, c2 = st.columns(2)
                preds = [x["pred"] for x in per]
                maj = int(sum(preds) >= 2)
                (c1.error if maj else c1.success)(
                    f"**{T['majority']}**\n\n### {'🔴' if maj else '🟢'} {ID2LABEL[maj]}\n"
                    f"Votes: {preds.count(1)}/3 HATE"
                )
                avg = np.mean([x["probs"] for x in per], 0)
                sp = int(avg.argmax())
                (c2.error if sp else c2.success)(
                    f"**{T['soft']}**\n\n### {'🔴' if sp else '🟢'} {ID2LABEL[sp]}\n"
                    f"Avg HATE={avg[1]:.3f} | NON-HATE={avg[0]:.3f}"
                )
                target = maj

                st.markdown(T["ig_title"])
                st.caption(T["ig_caption"])
                for m in MODELS:
                    if not m["ok"]: continue
                    st.markdown(f"**{m['name']}** → `{ID2LABEL[target]}`")
                    try:
                        sc = ig_explain(m, text, target)
                        st.markdown(f'<div style="line-height:2.2;font-size:1.05rem;">{highlight_html(sc)}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"IG failed: {e}")

                st.markdown(T["ig_chart_title"])
                for m in MODELS:
                    if not m["ok"]: continue
                    with st.expander(f"IG — {m['name']}"):
                        try:
                            sc = ig_explain(m, text, target)
                            top = sorted(sc, key=lambda x: -abs(x[1]))[:15]
                            toks = [t for t, _ in top][::-1]
                            vals = [s for _, s in top][::-1]
                            cols_ = ["#c0392b" if v >= 0 else "#27ae60" for v in vals]
                            fig, ax = plt.subplots(figsize=(7, max(3, len(top)*0.35)))
                            ax.barh(toks, vals, color=cols_); ax.axvline(0, color="k", lw=0.8)
                            ax.set_xlabel(f"Attribution → {ID2LABEL[target]}")
                            st.pyplot(fig); plt.close(fig)
                        except Exception as e:
                            st.warning(f"Chart failed: {e}")

                st.markdown(T["lime_title"])
                for m in MODELS:
                    if not m["ok"]: continue
                    with st.expander(f"LIME — {m['name']}"):
                        try:
                            exp = lime_explain(m, text, n=12)
                            st.markdown(T["hate_contrib"])
                            for w, v in exp.as_list(label=1):
                                st.markdown(f"- {'🔴' if v>0 else '🟢'} `{w}` → **{v:+.4f}**")
                            st.markdown(T["non_hate_contrib"])
                            for w, v in exp.as_list(label=0):
                                st.markdown(f"- {'🟢' if v>0 else '🔴'} `{w}` → **{v:+.4f}**")
                        except Exception as e:
                            st.warning(f"LIME failed: {e}")
        elif go:
            st.warning("Please enter a comment first.")

    # =========================================================
    # TAB 2 — EVALUATION
    # =========================================================
    with t2:
        st.subheader(T["eval_title"])
        st.caption(f"`{TEST_DATA_PATH}` • ground truth: `{LABEL_COLUMN}`")

        a, b = st.columns(2)
        fresh = a.button(T["run_eval"], type="primary")
        cached_btn = b.button(T["load_cache"])

        res = None
        if fresh:
            with st.spinner("Running predictions..."):
                res, n = run_eval(tuple(MODELS), TEST_DATA_PATH, TEXT_COLUMN, LABEL_COLUMN)
                if res:
                    try:
                        with open(EVAL_CACHE_PATH, "w", encoding="utf-8") as f:
                            json.dump(res, f, indent=2, ensure_ascii=False)
                        st.success(f"✅ Evaluated {n} samples. Cache saved.")
                    except Exception as e:
                        st.warning(f"Cache save failed: {e}")
                else:
                    st.error("Dataset not found. Please check config.py path.")
                
        elif cached_btn and os.path.exists(EVAL_CACHE_PATH):
            with open(EVAL_CACHE_PATH, "r", encoding="utf-8") as f:
                res = json.load(f)
            st.success("📦 Loaded cached results.")

        if res:
            rows = []
            for name, r in res.items():
                if "metrics" not in r: continue
                m = r["metrics"]
                rows.append({
                    "Model": name,
                    "Accuracy": m["accuracy"],
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                    "Macro F1": m["macro_f1"],
                    "Weighted F1": m["weighted_f1"],
                })
            df = pd.DataFrame(rows)
            st.markdown(T["comp_title"])
            st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns if c != "Model"}), use_container_width=True, hide_index=True)

            st.markdown(T["chart_title"])
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(df)); w = 0.35
            ax.bar(x-w/2, df["Accuracy"], w, label="Accuracy", color="#3498db")
            ax.bar(x+w/2, df["Macro F1"], w, label="Macro F1", color="#e67e22")
            ax.set_xticks(x); ax.set_xticklabels(df["Model"], rotation=20, ha="right")
            ax.set_ylim(0, 1.05); ax.legend(); st.pyplot(fig); plt.close(fig)

            st.markdown(T["conf_title"])
            cc = st.columns(3)
            for i, (name, r) in enumerate(res.items()):
                if "confusion" not in r: continue
                with cc[i % 3]:
                    cm = np.array(r["confusion"])
                    fig, ax = plt.subplots(figsize=(3.2, 2.8))
                    ax.imshow(cm, cmap="Blues")
                    ax.set_xticks([0,1]); ax.set_yticks([0,1])
                    ax.set_xticklabels(["NH","H"]); ax.set_yticklabels(["NH","H"])
                    ax.set_xlabel("Pred"); ax.set_ylabel("Actual")
                    ax.set_title(name, fontsize=9)
                    for (j,k), v in np.ndenumerate(cm):
                        ax.text(k, j, str(v), ha="center", va="center", color="white" if v > cm.max()/2 else "black")
                    st.pyplot(fig); plt.close(fig)

            st.markdown(T["rep_title"])
            for name, r in res.items():
                if "report" not in r: continue
                with st.expander(f"{name}"):
                    st.dataframe(pd.DataFrame(r["report"]).transpose().round(4), use_container_width=True)

    # =========================================================
    # TAB 3 — ABOUT
    # =========================================================
    with t3:
        st.markdown(textwrap.dedent("""
    ### About TeluguShield AI
    **Inference-only prototype** — no training, no fine-tuning.

    | # | Model | Description |
    |---|-------|-------------|
    | 1 | mBERT | Multilingual BERT hate speech model |
    | 2 | MuRIL | Author's research checkpoint (HP-tuned) |
    | 3 | TeluguBERT | Author's research checkpoint (HP-tuned) |

    **Ensembles:** Majority Voting (2+ HATE → HATE), Soft Voting (avg probabilities)
    **Explainability:** Integrated Gradients (Captum, live) + LIME (live) + Red/Green highlighting
    **Evaluation:** genuine metrics on independent test set.
    """))

    st.markdown("---")
    st.caption("TeluguShield AI • Inference only • MURIL & TeluguBERT trained models")