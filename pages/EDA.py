import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="EDA · AssurIA", page_icon="🔍", layout="wide")

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
.kpi-value { font-family:'DM Serif Display',serif; font-size:2rem; color:#f0a500; }
.kpi-label { font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

TEMPLATE = "plotly_dark"

@st.cache_data
def load_train():
    return pd.read_csv("Data/train_info.csv")

df = load_train()
df["reponse_client_str"] = df["reponse_client"].astype(str)

st.markdown("""
<div class='section-title'>🔍 Exploration des données</div>
<div class='section-sub'>Analyse descriptive du dataset d'entraînement (train_info.csv)</div>
<hr style='border-color:#2a2d3a; margin-bottom:24px;'>
""", unsafe_allow_html=True)

with st.expander("🎛️ Filtres globaux", expanded=False):
    fc1, fc2, fc3 = st.columns(3)
    genre_sel = fc1.multiselect("Genre", df["genre"].unique(), default=list(df["genre"].unique()))
    veh_sel   = fc2.multiselect("Âge du véhicule", df["age_vehicule"].unique(), default=list(df["age_vehicule"].unique()))
    age_range = fc3.slider("Tranche d'âge", int(df["age"].min()), int(df["age"].max()),
                           (int(df["age"].min()), int(df["age"].max())))

df_f = df[df["genre"].isin(genre_sel) & df["age_vehicule"].isin(veh_sel) & df["age"].between(*age_range)]

k1, k2, k3, k4 = st.columns(4)
taux = df_f["reponse_client"].mean() * 100
for col, val, label in zip(
    [k1, k2, k3, k4],
    [f"{len(df_f):,}", f"{taux:.1f}%", f"{df_f['age'].mean():.0f} ans", f"{df_f['prime_annuelle'].mean():,.0f} €"],
    ["Clients filtrés", "Taux souscription", "Âge moyen", "Prime moyenne"]
):
    col.markdown(f"""<div class='kpi-card'><div class='kpi-value'>{val}</div><div class='kpi-label'>{label}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🎯 Distribution de la variable cible")
tc1, tc2 = st.columns([1, 2])

with tc1:
    counts = df_f["reponse_client"].value_counts().reset_index()
    counts.columns = ["reponse", "nb"]
    counts["label"] = counts["reponse"].map({0: "Non intéressé", 1: "Intéressé"})
    fig_pie = px.pie(counts, values="nb", names="label", color="label",
                     color_discrete_map={"Non intéressé": "#5ab4ff", "Intéressé": "#f0a500"},
                     hole=0.5, template=TEMPLATE)
    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

with tc2:
    st.markdown("""<div style='background:#1a1c25; border-left:3px solid #f0a500; border-radius:0 10px 10px 0;
                padding:16px 20px; color:#ccc; font-size:0.88rem; line-height:1.7; margin-top:30px;'>
    <strong style='color:#f0a500;'>Déséquilibre des classes détecté.</strong><br>
    Environ <strong>12 %</strong> des clients sont intéressés par la souscription (classe minoritaire).<br>
    Ce déséquilibre a nécessité des métriques adaptées (F1-score, ROC-AUC) et des techniques de rééquilibrage (SMOTE, scale_pos_weight).
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#2a2d3a; margin:24px 0;'>", unsafe_allow_html=True)
st.markdown("### 📊 Variables catégorielles")
cat_col = st.selectbox("Choisir une variable", ["genre", "age_vehicule", "vehicule_endommage", "ancien_assure"])

fig_cat = px.histogram(df_f, x=cat_col, color="reponse_client_str", barmode="group",
                       color_discrete_map={"0": "#5ab4ff", "1": "#f0a500"},
                       labels={"reponse_client_str": "Réponse client"}, template=TEMPLATE, text_auto=True)
fig_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title="Réponse (0=Non · 1=Oui)")
st.plotly_chart(fig_cat, use_container_width=True)

st.markdown("<hr style='border-color:#2a2d3a; margin:24px 0;'>", unsafe_allow_html=True)
st.markdown("### 📈 Distribution des variables numériques")
num_sel = st.selectbox("Choisir une variable numérique", ["age", "prime_annuelle", "anciennete", "code_regional", "canal_communication"])

nc1, nc2 = st.columns(2)
with nc1:
    fig_hist = px.histogram(df_f, x=num_sel, color="reponse_client_str", nbins=40, barmode="overlay", opacity=0.7,
                            color_discrete_map={"0": "#5ab4ff", "1": "#f0a500"}, template=TEMPLATE)
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title=f"Distribution — {num_sel}")
    st.plotly_chart(fig_hist, use_container_width=True)

with nc2:
    fig_box = px.box(df_f, x="reponse_client_str", y=num_sel, color="reponse_client_str",
                     color_discrete_map={"0": "#5ab4ff", "1": "#f0a500"}, template=TEMPLATE)
    fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", title=f"Boxplot — {num_sel}", showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("<hr style='border-color:#2a2d3a; margin:24px 0;'>", unsafe_allow_html=True)
st.markdown("### 🔗 Matrice de corrélation (Spearman)")

num_for_corr = ["age", "permis_conduire", "code_regional", "ancien_assure",
                "vehicule_endommage", "prime_annuelle", "canal_communication", "anciennete", "reponse_client"]
df_corr = df_f[num_for_corr].copy()
if df_corr["vehicule_endommage"].dtype == object:
    df_corr["vehicule_endommage"] = df_corr["vehicule_endommage"].map({"Yes": 1, "No": 0})

corr = df_corr.corr(method="spearman").round(2)
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", template=TEMPLATE)
fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=30))
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("""<div style='background:#1a1c25; border-left:3px solid #5ab4ff; border-radius:0 10px 10px 0;
            padding:14px 18px; color:#aaa; font-size:0.84rem; line-height:1.6;'>
<strong style='color:#5ab4ff;'>Lecture :</strong>
La variable <code>ancien_assure</code> présente la corrélation la plus forte avec la réponse client (−0.34).
Une corrélation notable est aussi observée entre <code>age</code> et <code>canal_communication</code> (−0.58).
</div>""", unsafe_allow_html=True)