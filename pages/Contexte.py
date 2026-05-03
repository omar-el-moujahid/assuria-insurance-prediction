import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="Contexte · AssurIA", page_icon="📋", layout="wide")

# ── CSS (même thème global) ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8e6e1; }
[data-testid="stSidebar"] { background: #16181f !important; border-right: 1px solid #2a2d3a; }
[data-testid="stSidebar"] * { color: #c9c7c0 !important; }
.kpi-card { background: linear-gradient(135deg,#1e2130,#252838); border:1px solid #2e3147; border-radius:12px; padding:20px 24px; text-align:center; }
.kpi-value { font-family:'DM Serif Display',serif; font-size:2.2rem; color:#f0a500; line-height:1.1; }
.kpi-label { font-size:0.78rem; color:#888; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }
.section-title { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#f0a500; }
.section-sub { color:#888; font-size:0.88rem; margin-bottom:1.2rem; }
.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; margin:2px; }
.badge-blue  { background:#1a3a5c; color:#5ab4ff; }
.badge-green { background:#1a3d2b; color:#4cde8a; }
.badge-gold  { background:#3d2e0a; color:#f0a500; }
.badge-red   { background:#3d1a1a; color:#ff6b6b; }
.var-card { background:#1a1c25; border:1px solid #2a2d3a; border-radius:10px; padding:14px 18px; margin-bottom:8px; }
.var-name { font-weight:600; color:#f0a500; font-size:0.9rem; }
.var-desc { color:#aaa; font-size:0.82rem; margin-top:3px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding-bottom:8px;'>
    <div class='section-title'>📋 Contexte du projet</div>
    <div class='section-sub'>Compagnie d'assurance · Prédiction de souscription automobile</div>
</div>
<hr style='border-color:#2a2d3a; margin-bottom:24px;'>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpis = [
    ("381 109", "Clients dans le dataset"),
    ("12.3 %",  "Taux de souscription"),
    ("11",      "Variables disponibles"),
    ("XGBoost", "Modèle retenu"),
]
for col, (val, label) in zip([c1, c2, c3, c4], kpis):
    col.markdown(f"""<div class='kpi-card'>
        <div class='kpi-value'>{val}</div>
        <div class='kpi-label'>{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Contexte métier ───────────────────────────────────────────────────────────
st.markdown("""
<div class='section-title' style='font-size:1.3rem;'>🏢 Mission</div>
<hr style='border-color:#2a2d3a; margin:8px 0 16px 0;'>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:#1a1c25; border-left:3px solid #f0a500; border-radius:0 10px 10px 0;
            padding:18px 22px; color:#ccc; font-size:0.92rem; line-height:1.7;'>
En tant que consultant Data Science au sein d'une compagnie d'assurance, notre mission est
d'<strong style='color:#f0a500;'>anticiper la probabilité</strong> que chaque client souscrive
un contrat d'assurance automobile au cours de l'année.<br><br>
L'objectif final est d'<strong style='color:#f0a500;'>optimiser la stratégie commerciale</strong>
en concentrant les actions marketing sur les clients les plus susceptibles d'être convertis,
tout en évitant de contacter inutilement ceux déjà certains de souscrire ou ceux très peu susceptibles de le faire.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Pipeline Data Science ─────────────────────────────────────────────────────
st.markdown("""
<div class='section-title' style='font-size:1.3rem;'>⚙️ Pipeline du projet</div>
<hr style='border-color:#2a2d3a; margin:8px 0 16px 0;'>
""", unsafe_allow_html=True)

steps = [
    ("🔍", "Exploration (EDA)",       "Analyse de la structure du dataset, détection des anomalies, étude des distributions."),
    ("📊", "Visualisation",           "Représentation graphique des tendances et relations entre variables."),
    ("🔧", "Feature Engineering",     "Création de tranches d'âge, variables d'interaction, encodage métier."),
    ("🤖", "Modélisation",            "XGBoost avec gestion du déséquilibre (scale_pos_weight, SMOTE), optimisation des hyperparamètres."),
    ("📈", "Évaluation",              "F1-score, ROC-AUC, matrice de confusion, seuil de décision optimisé."),
    ("🎯", "Production & Ciblage",    "Application du modèle sur les nouveaux clients, définition de la zone d'incertitude [0.3 – 0.7]."),
]

cols = st.columns(3)
for i, (icon, title, desc) in enumerate(steps):
    with cols[i % 3]:
        st.markdown(f"""
        <div class='var-card' style='border-left:3px solid #f0a500; margin-bottom:12px;'>
            <div class='var-name'>{icon} {title}</div>
            <div class='var-desc'>{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Description des variables ─────────────────────────────────────────────────
st.markdown("""
<div class='section-title' style='font-size:1.3rem;'>📐 Description des variables</div>
<hr style='border-color:#2a2d3a; margin:8px 0 16px 0;'>
""", unsafe_allow_html=True)

variables = [
    ("id_client",           "Identifiant unique du client.",                                             "blue"),
    ("genre",               "Genre du client (Male / Female).",                                          "blue"),
    ("age",                 "Âge du client.",                                                            "blue"),
    ("permis_conduire",     "0 = pas de permis · 1 = permis valide.",                                   "green"),
    ("code_regional",       "Code de la région de résidence du client.",                                 "blue"),
    ("ancien_assure",       "0 = jamais assuré · 1 = déjà assuré.",                                    "green"),
    ("age_vehicule",        "Âge du véhicule : < 1 an · 1-2 ans · > 2 ans.",                           "green"),
    ("vehicule_endommage",  "1 = véhicule déjà endommagé · 0 = intact.",                               "green"),
    ("prime_annuelle",      "Montant de la prime annuelle à payer (en €).",                             "gold"),
    ("canal_communication", "Canal de contact anonymisé (mail, téléphone, internet…).",                 "blue"),
    ("anciennete",          "Ancienneté du client en nombre de jours.",                                 "gold"),
    ("reponse_client",      "🎯 Variable cible : 0 = non intéressé · 1 = intéressé par la souscription.", "red"),
]

col_a, col_b = st.columns(2)
for i, (name, desc, badge) in enumerate(variables):
    target_col = col_a if i % 2 == 0 else col_b
    target_col.markdown(f"""
    <div class='var-card'>
        <div class='var-name'>
            <span class='badge badge-{badge}'>{name}</span>
        </div>
        <div class='var-desc' style='margin-top:6px;'>{desc}</div>
    </div>""", unsafe_allow_html=True)

