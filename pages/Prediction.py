import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Prédiction individuelle · AssurIA", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8e6e1; }
[data-testid="stSidebar"] { background: #16181f !important; border-right: 1px solid #2a2d3a; }
[data-testid="stSidebar"] * { color: #c9c7c0 !important; }
.section-title { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#f0a500; }
.section-sub { color:#888; font-size:0.88rem; margin-bottom:1.2rem; }
.result-card { border-radius:16px; padding:28px 32px; text-align:center; margin-bottom:20px; }
.result-value { font-family:'DM Serif Display',serif; font-size:3.5rem; line-height:1; }
.result-label { font-size:0.85rem; text-transform:uppercase; letter-spacing:0.1em; margin-top:8px; }
.input-section { background:#1a1c25; border:1px solid #2a2d3a; border-radius:12px; padding:20px 24px; margin-bottom:16px; }
.input-title { color:#f0a500; font-size:0.82rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return {
        "model":         joblib.load("model.pkl"),
        "onehot":        joblib.load("onehot.pkl"),
        "ordinal":       joblib.load("ordinal.pkl"),
        "minmax":        joblib.load("minmax.pkl"),
        "robust":        joblib.load("robust.pkl"),
        "standard":      joblib.load("standard.pkl"),
        "mean_canal":    joblib.load("mean_canal.pkl"),
        "mean_region":   joblib.load("mean_region.pkl"),
        "columns_order": joblib.load("columns_order.pkl"),
        "standard_cols": joblib.load("standard_cols.pkl"),
        "minmax_cols":   joblib.load("minmax_cols.pkl"),
        "robust_cols":   joblib.load("robust_cols.pkl"),
    }

@st.cache_data
def load_train():
    return pd.read_csv("Data/train_info.csv")

# ── Charger pipeline + détecter les vraies catégories ────────────────────────
pipe = load_pipeline()
onehot = pipe["onehot"]

# Récupérer les catégories réelles apprises par le OneHotEncoder
# categories_[0] = genre, [1] = vehicule_endommage, [2] = ancien_assure, etc.
cats = onehot.categories_
genre_cats          = list(cats[0])   # ex: ['Female', 'Male'] ou ['0', '1']
veh_endommage_cats  = list(cats[1])
ancien_assure_cats  = list(cats[2])

# Pour age_vehicule → OrdinalEncoder
ordinal_cats = list(pipe["ordinal"].categories_[0])

# Canaux et régions connues
mean_canal  = pipe["mean_canal"]
mean_region = pipe["mean_region"]
canaux_connus  = sorted(mean_canal.index.tolist())
regions_connues = sorted(mean_region.index.tolist())

# ── Interface ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='section-title'>🔮 Prédiction individuelle</div>
<div class='section-sub'>Renseignez le profil d'un client pour estimer sa probabilité de souscription</div>
<hr style='border-color:#2a2d3a; margin-bottom:24px;'>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1])

with left:
    st.markdown("<div class='input-section'><div class='input-title'>👤 Profil client</div>", unsafe_allow_html=True)
    genre           = st.selectbox("Genre", genre_cats)
    age             = st.slider("Âge", 18, 90, 35)
    permis_conduire = st.selectbox("Permis de conduire", [1, 0], format_func=lambda x: "Oui" if x else "Non")
    # ancien_assure : utiliser les vraies catégories
    ancien_assure   = st.selectbox("Déjà assuré ailleurs ?", ancien_assure_cats,
                                   format_func=lambda x: "Oui" if str(x) in ["1","Yes"] else "Non")
    anciennete      = st.slider("Ancienneté (jours)", 0, 300, 100)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='input-section'><div class='input-title'>🚗 Véhicule & Contrat</div>", unsafe_allow_html=True)
    age_vehicule       = st.selectbox("Âge du véhicule", ordinal_cats)
    vehicule_endommage = st.selectbox("Véhicule endommagé ?", veh_endommage_cats,
                                      format_func=lambda x: "Oui" if str(x) in ["1","Yes"] else "Non")
    prime_annuelle     = st.number_input("Prime annuelle (€)", min_value=1000, max_value=100000, value=30000, step=500)
    canal_communication = st.selectbox("Canal de communication", canaux_connus)
    code_regional       = st.selectbox("Code régional", regions_connues)
    st.markdown("</div>", unsafe_allow_html=True)

predict_btn = st.button("🔮 Lancer la prédiction", type="primary", use_container_width=True)

if predict_btn:
    try:
        client = pd.DataFrame([{
            "genre": genre,
            "age": age,
            "permis_conduire": permis_conduire,
            "code_regional": code_regional,
            "ancien_assure": ancien_assure,
            "age_vehicule": age_vehicule,
            "vehicule_endommage": vehicule_endommage,
            "prime_annuelle": prime_annuelle,
            "canal_communication": canal_communication,
            "anciennete": anciennete
        }])

        # Feature engineering identique au notebook
        client["tranche_age"] = pd.cut(client["age"],
            bins=[0,30,40,50,60,70,80,100],
            labels=["20-30","30-40","40-50","50-60","60-70","70-80","80+"])

        client["tranche_age_age_vehicule"]         = client["tranche_age"].astype(str) + "_" + client["age_vehicule"].astype(str)
        client["tranche_age_vehicule_endommage"]   = client["tranche_age"].astype(str) + "_" + client["vehicule_endommage"].astype(str)
        client["tranche_age_ancien_assure"]        = client["tranche_age"].astype(str) + "_" + client["ancien_assure"].astype(str)
        client["age_vehicule_vehicule_endommage"]  = client["age_vehicule"].astype(str) + "_" + client["vehicule_endommage"].astype(str)
        client["age_vehicule_ancien_assure"]       = client["age_vehicule"].astype(str) + "_" + client["ancien_assure"].astype(str)
        client["vehicule_endommage_ancien_assure"] = client["vehicule_endommage"].astype(str) + "_" + client["ancien_assure"].astype(str)

        onehot_cols = [
            "genre","vehicule_endommage","ancien_assure","tranche_age",
            "tranche_age_age_vehicule","tranche_age_vehicule_endommage",
            "tranche_age_ancien_assure","age_vehicule_vehicule_endommage",
            "age_vehicule_ancien_assure","vehicule_endommage_ancien_assure"
        ]

        encoded    = pipe["onehot"].transform(client[onehot_cols])
        encoded_df = pd.DataFrame(encoded, columns=pipe["onehot"].get_feature_names_out(onehot_cols))

        client["age_vehicule"] = pipe["ordinal"].transform(client[["age_vehicule"]])
        client["canal_communication_encoded"] = client["canal_communication"].map(pipe["mean_canal"]).fillna(pipe["mean_canal"].mean())
        client["code_regional_encoded"]       = client["code_regional"].map(pipe["mean_region"]).fillna(pipe["mean_region"].mean())

        client = pd.concat([client.drop(columns=onehot_cols), encoded_df], axis=1)
        client.drop(columns=["code_regional","canal_communication"], inplace=True, errors="ignore")

        minmax_cols   = pipe["minmax_cols"]
        robust_cols   = pipe["robust_cols"]
        standard_cols = [c for c in client.columns if c not in list(minmax_cols) + list(robust_cols)]

        client[minmax_cols]   = pipe["minmax"].transform(client[minmax_cols])
        client[robust_cols]   = pipe["robust"].transform(client[robust_cols])
        client[standard_cols] = pipe["standard"].transform(client[standard_cols])

        client.columns = client.columns.str.replace(r'[\[\]<>]', '_', regex=True)
        train_cols = pd.Index(pipe["columns_order"]).str.replace(r'[\[\]<>]', '_', regex=True)
        client = client.reindex(columns=train_cols, fill_value=0)

        import xgboost as xgb
        dmat  = xgb.DMatrix(client)
        proba = float(pipe["model"].predict(dmat)[0])

        # ── Résultat ──────────────────────────────────────────────────────────
        st.markdown("<hr style='border-color:#2a2d3a; margin:24px 0;'>", unsafe_allow_html=True)
        st.markdown("### 📊 Résultat")

        res1, res2 = st.columns([1, 2])

        with res1:
            if proba >= 0.7:
                color, icon, message, bg = "#4cde8a", "🟢", "Très susceptible de souscrire", "#0e2a1a"
            elif proba >= 0.3:
                color, icon, message, bg = "#f0a500", "🟡", "Incertain — à contacter", "#2a1e00"
            else:
                color, icon, message, bg = "#ff6b6b", "🔴", "Peu susceptible de souscrire", "#2a0e0e"

            st.markdown(f"""<div class='result-card' style='background:{bg}; border:2px solid {color};'>
                <div class='result-value' style='color:{color};'>{proba*100:.1f}%</div>
                <div class='result-label' style='color:{color};'>{icon} {message}</div>
            </div>""", unsafe_allow_html=True)

        with res2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"color": color, "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#555"},
                    "bar":  {"color": color},
                    "bgcolor": "#1a1c25",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30],   "color": "#2a0e0e"},
                        {"range": [30, 70],  "color": "#2a1e00"},
                        {"range": [70, 100], "color": "#0e2a1a"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": proba * 100}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220,
                                    margin=dict(t=10, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        if proba >= 0.7:
            conseil = "✅ Ce client est très probablement intéressé. Un contact proactif est recommandé avec une offre personnalisée."
        elif proba >= 0.3:
            conseil = "⚡ Ce client est dans la zone d'incertitude : une action commerciale ciblée peut faire basculer sa décision. C'est le profil prioritaire à contacter."
        else:
            conseil = "⏭️ Ce client est peu susceptible de souscrire actuellement. Inutile de mobiliser des ressources commerciales sur ce profil."

        st.markdown(f"""<div style='background:#1a1c25; border-left:3px solid {color}; border-radius:0 10px 10px 0;
                    padding:14px 18px; color:#ccc; font-size:0.9rem; line-height:1.6;'>{conseil}</div>""",
                    unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.exception(e)