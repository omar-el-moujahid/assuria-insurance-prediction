import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Décision · AssurIA", page_icon="💡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8e6e1; }
[data-testid="stSidebar"] { background: #16181f !important; border-right: 1px solid #2a2d3a; }
[data-testid="stSidebar"] * { color: #c9c7c0 !important; }
.section-title { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#f0a500; }
.section-sub { color:#888; font-size:0.88rem; margin-bottom:1.2rem; }
.kpi-card { background:linear-gradient(135deg,#1e2130,#252838); border:1px solid #2e3147; border-radius:12px; padding:16px 20px; text-align:center; }
.kpi-value { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#f0a500; }
.kpi-label { font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }
.reco-card { background:#1a1c25; border:1px solid #2a2d3a; border-radius:12px; padding:18px 22px; margin-bottom:12px; }
.reco-title { color:#f0a500; font-weight:600; font-size:0.95rem; margin-bottom:6px; }
.reco-text { color:#aaa; font-size:0.85rem; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

TEMPLATE = "plotly_dark"

@st.cache_data
def load_clients_cibles():
    return pd.read_csv("Data/clients_cibles.csv")

@st.cache_data
def load_train():
    return pd.read_csv("Data/train_info.csv")

df_cibles = load_clients_cibles()
df_train  = load_train()

st.markdown("""
<div class='section-title'>💡 Aide à la décision</div>
<div class='section-sub'>Profil des clients à cibler · Recommandations marketing</div>
<hr style='border-color:#2a2d3a; margin-bottom:24px;'>
""", unsafe_allow_html=True)

st.markdown("### 👤 Profil moyen des clients dans la zone d'incertitude")

kpis = []
if "age" in df_cibles.columns:
    kpis.append((f"{df_cibles['age'].mean():.0f} ans", "Âge moyen"))
if "prime_annuelle" in df_cibles.columns:
    kpis.append((f"{df_cibles['prime_annuelle'].mean():,.0f} €", "Prime annuelle moyenne"))
if "ancien_assure" in df_cibles.columns:
    pct = (df_cibles["ancien_assure"] == 0).mean() * 100
    kpis.append((f"{pct:.0f}%", "Non anciennement assurés"))
if "vehicule_endommage" in df_cibles.columns:
    v = df_cibles["vehicule_endommage"]
    pct_end = (v == "Yes").mean() * 100 if v.dtype == object else (v == 1).mean() * 100
    kpis.append((f"{pct_end:.0f}%", "Véhicule endommagé"))

cols = st.columns(len(kpis) if kpis else 4)
for col, (val, label) in zip(cols, kpis):
    col.markdown(f"""<div class='kpi-card'><div class='kpi-value'>{val}</div><div class='kpi-label'>{label}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📊 Comparaison : clients ciblés vs population globale")

compare_cols = [c for c in ["age", "prime_annuelle", "anciennete"] if c in df_cibles.columns and c in df_train.columns]
sel_col = st.selectbox("Variable à comparer", compare_cols)

fig_comp = go.Figure()
fig_comp.add_trace(go.Histogram(x=df_train[sel_col], name="Population globale",
    marker_color="#5ab4ff", opacity=0.6, nbinsx=40, histnorm="percent"))
fig_comp.add_trace(go.Histogram(x=df_cibles[sel_col], name="Clients ciblés",
    marker_color="#f0a500", opacity=0.7, nbinsx=40, histnorm="percent"))
fig_comp.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    template=TEMPLATE, xaxis_title=sel_col, yaxis_title="% clients",
    legend=dict(orientation="h", y=1.1), margin=dict(t=20))
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("<hr style='border-color:#2a2d3a; margin:24px 0;'>", unsafe_allow_html=True)
st.markdown("### 📋 Répartition des variables catégorielles (clients ciblés)")

cat_available = [c for c in ["genre", "age_vehicule", "vehicule_endommage", "ancien_assure"] if c in df_cibles.columns]
if cat_available:
    cat_sel = st.selectbox("Variable catégorielle", cat_available)
    vc = df_cibles[cat_sel].value_counts(normalize=True).reset_index()
    vc.columns = [cat_sel, "proportion"]
    vc["proportion_pct"] = (vc["proportion"] * 100).round(1)
    fig_bar = px.bar(vc, x=cat_sel, y="proportion_pct", color=cat_sel,
                     color_discrete_sequence=["#f0a500","#5ab4ff","#4cde8a","#ff6b6b"],
                     text="proportion_pct", template=TEMPLATE)
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, yaxis_title="%", margin=dict(t=20))
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("<hr style='border-color:#2a2d3a; margin:24px 0;'>", unsafe_allow_html=True)
st.markdown("### 🧭 Recommandations marketing")

recos = [
    ("🎯 Priorité : véhicule endommagé + non assuré",
     "Concentrez les efforts sur les clients dont le véhicule a été endommagé et qui n'ont jamais souscrit. Ce profil est le plus prédictif d'une conversion."),
    ("📞 Adapter le canal selon l'âge",
     "Les clients plus jeunes sont accessibles via des canaux digitaux. Les profils plus âgés répondent mieux au contact téléphonique direct."),
    ("💶 Sensibilité à la prime",
     "Les clients avec une prime dans la moyenne (~30 000 €) sont les plus souvent dans la zone d'incertitude. Des offres personnalisées peuvent faire la différence."),
    ("⏱️ Ancienneté faible = opportunité",
     "Les clients récents sont encore en phase de découverte. C'est le bon moment pour proposer un contrat automobile complémentaire."),
    ("🚗 Véhicule de 1 à 2 ans = profil idéal",
     "Les propriétaires de véhicules relativement récents et endommagés ont la plus forte propension à souscrire. Ciblez cette tranche en priorité."),
]
for title, text in recos:
    st.markdown(f"""<div class='reco-card'><div class='reco-title'>{title}</div><div class='reco-text'>{text}</div></div>""", unsafe_allow_html=True)