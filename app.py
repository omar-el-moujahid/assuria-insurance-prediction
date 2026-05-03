import streamlit as st

st.set_page_config(
    page_title="AssurIA — Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ---- Background ---- */
.stApp {
    background: #0f1117;
    color: #e8e6e1;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #16181f !important;
    border-right: 1px solid #2a2d3a;
}
[data-testid="stSidebar"] * {
    color: #c9c7c0 !important;
}

/* ---- KPI Cards ---- */
.kpi-card {
    background: linear-gradient(135deg, #1e2130 0%, #252838 100%);
    border: 1px solid #2e3147;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #f0a500;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.78rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ---- Section titles ---- */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #f0a500;
    margin-bottom: 0.2rem;
}
.section-sub {
    color: #888;
    font-size: 0.88rem;
    margin-bottom: 1.5rem;
}

/* ---- Badge ---- */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}
.badge-blue  { background:#1a3a5c; color:#5ab4ff; }
.badge-green { background:#1a3d2b; color:#4cde8a; }
.badge-gold  { background:#3d2e0a; color:#f0a500; }
.badge-red   { background:#3d1a1a; color:#ff6b6b; }

/* ---- Divider ---- */
.divider {
    border: none;
    border-top: 1px solid #2a2d3a;
    margin: 1.5rem 0;
}

/* ---- Plotly charts transparent ---- */
.js-plotly-plot .plotly, .js-plotly-plot .plotly div {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("""
#     <div style='padding: 8px 0 24px 0;'>
#         <div style='font-family:"DM Serif Display",serif; font-size:1.5rem; color:#f0a500;'>🛡️ AssurIA</div>
#         <div style='font-size:0.75rem; color:#555; margin-top:2px;'>Assurance Automobile · Dashboard</div>
#     </div>
#     <hr style='border-color:#2a2d3a; margin-bottom:20px;'>
#     """, unsafe_allow_html=True)

#     st.markdown("**Navigation**")
#     st.page_link("app.py",                              label=" Accueil")
#     st.page_link("pages/Contexte.py",                 label=" Contexte du projet")
#     st.page_link("pages/EDA.py",                      label=" Exploration des données")
#     st.page_link("pages/Modelisation.py",              label=" Modélisation")
#     st.page_link("pages/Decision.py",                  label=" Aide à la décision")
#     st.page_link("pages/Prediction_individuelle.py",   label=" Prédiction individuelle")

#     st.markdown("<hr style='border-color:#2a2d3a; margin-top:20px;'>", unsafe_allow_html=True)
#     st.markdown("<div style='font-size:0.7rem; color:#444;'>ENSIM · Projet Data Science 2025</div>",
#                 unsafe_allow_html=True)

# ── Home content ──────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 60px 0 20px 0; text-align:center;'>
    <div style='font-family:"DM Serif Display",serif; font-size:3.5rem; color:#f0a500; line-height:1.1;'>
        AssurIA
    </div>
    <div style='font-size:1.1rem; color:#888; margin-top:12px;'>
        Plateforme d'analyse prédictive · Souscription Assurance Automobile
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#2a2d3a; margin: 30px 0;'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<div class='kpi-card'>
        <div class='kpi-value'>381k</div>
        <div class='kpi-label'>Clients analysés</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class='kpi-card'>
        <div class='kpi-value'>12.3%</div>
        <div class='kpi-label'>Taux de souscription</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class='kpi-card'>
        <div class='kpi-value'>XGBoost</div>
        <div class='kpi-label'>Modèle utilisé</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#666; font-size:0.9rem; padding: 20px 0 40px 0;'>
    Utilisez la barre de navigation à gauche pour explorer le projet étape par étape.
</div>
""", unsafe_allow_html=True)