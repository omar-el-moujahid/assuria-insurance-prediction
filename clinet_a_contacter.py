#!/usr/bin/env python
# coding: utf-8

# In[72]:


# PIPELINE POUR NOUVELLE BASE (clients_a_contacter)
import pandas as pd
import joblib
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)


# In[116]:


# Charger objets
model = joblib.load("model.pkl")
onehot = joblib.load("onehot.pkl")
ordinal = joblib.load("ordinal.pkl")
minmax_scaler = joblib.load("minmax.pkl")
robust_scaler = joblib.load("robust.pkl")
standard_scaler = joblib.load("standard.pkl")
train_cols = joblib.load("columns_order.pkl")
standard_cols = joblib.load("standard_cols.pkl")
minmax_cols = joblib.load("minmax_cols.pkl")
robust_cols = joblib.load("robust_cols.pkl")


# In[144]:


#  Charger nouveaux clients
df_new = pd.read_csv("Data/clients_a_contacter.csv")


# In[145]:


# Feature engineering (IDENTIQUE)

df_new["tranche_age"] = pd.cut(
    df_new["age"],
    bins=[0, 30, 40, 50, 60, 70, 80, 100],
    labels=["20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"]
)

df_new["tranche_age_age_vehicule"] = df_new["tranche_age"].astype(str) + "_" + df_new["age_vehicule"].astype(str)
df_new["tranche_age_vehicule_endommage"] = df_new["tranche_age"].astype(str) + "_" + df_new["vehicule_endommage"].astype(str)
df_new["tranche_age_ancien_assure"] = df_new["tranche_age"].astype(str) + "_" + df_new["ancien_assure"].astype(str)
df_new["age_vehicule_vehicule_endommage"] = df_new["age_vehicule"].astype(str) + "_" + df_new["vehicule_endommage"].astype(str)
df_new["age_vehicule_ancien_assure"] = df_new["age_vehicule"].astype(str) + "_" + df_new["ancien_assure"].astype(str)
df_new["vehicule_endommage_ancien_assure"] = df_new["vehicule_endommage"].astype(str) + "_" + df_new["ancien_assure"].astype(str)


# In[146]:


# Encodage

onehot_cols = [
    "genre",
    "vehicule_endommage",
    "ancien_assure",
    "tranche_age",
    "tranche_age_age_vehicule",
    "tranche_age_vehicule_endommage",
    "tranche_age_ancien_assure",
    "age_vehicule_vehicule_endommage",
    "age_vehicule_ancien_assure",
    "vehicule_endommage_ancien_assure"
]
encoded = onehot.transform(df_new[onehot_cols])
encoded_df = pd.DataFrame(encoded, columns=onehot.get_feature_names_out(onehot_cols))

df_new["age_vehicule"] = ordinal.transform(df_new[["age_vehicule"]])



# In[147]:


# Encodage des variables à forte cardinalité
mean_canal = joblib.load("mean_canal.pkl")
mean_region = joblib.load("mean_region.pkl")
df_new["canal_communication_encoded"] = df_new["canal_communication"].map(mean_canal)
df_new["code_regional_encoded"] = df_new["code_regional"].map(mean_region)


# In[149]:


df_new.head()   


# In[150]:


df_new = pd.concat([df_new.drop(columns=onehot_cols), encoded_df], axis=1)


# In[151]:


for col in df_new.columns:
    if df_new[col].dtype == "object":
        print(f"Colonne '{col}' is still object type")
    


# In[152]:


df_new.shape


# In[153]:


df_new.drop(columns=["id_client","code_regional","canal_communication"], inplace=True)


# In[154]:


df_new.head()


# In[124]:


df_new.shape


# In[155]:


df_new.drop(columns=["age"], inplace=True)
df_new["age"]=df_first["age"]


# In[156]:


df_new.head()


# In[126]:


df_new.drop(columns=["age"], inplace=True)
df_first = pd.read_csv("Data/clients_a_contacter.csv")
df_new["age"] = df_first["age"]


# In[157]:


df_new.head()


# In[158]:


# # alignement
# df_new = df_new.reindex(columns=train_cols, fill_value=0)

# # 🔥 recalcul correct
# standard_cols = [
#     col for col in train_cols
#     if col not in minmax_cols + robust_cols
# ]

minmax_cols = ["age", "permis_conduire"]

# Colonnes pour RobustScaler (avec outliers)
robust_cols = ["prime_annuelle"]

# Colonnes restantes → StandardScaler
# On enlève celles déjà utilisées
standard_cols = [
    col for col in df_new.columns
    if col not in minmax_cols + robust_cols
]

# scaling
df_new[minmax_cols] = minmax_scaler.transform(df_new[minmax_cols])
df_new[robust_cols] = robust_scaler.transform(df_new[robust_cols])
df_new[standard_cols] = standard_scaler.transform(df_new[standard_cols])


# In[159]:


# Nettoyer noms colonnes
df_new.columns = df_new.columns.str.replace('[\\[\\]<>]', '_', regex=True)


# In[164]:


import xgboost as xgb

dnew = xgb.DMatrix(df_new)
proba = model.predict(dnew)


# In[165]:


# Sélection marketing
df_new["proba"] = proba
df_new["prediction"] = (df_new["proba"] > 0.7).astype(int)


# In[167]:


# Sauvegarde
clients_cibles = df_new[(df_new["proba"] >= 0.3) & (df_new["proba"] <= 0.7)]
clients_cibles.to_csv("clients_cibles.csv", index=False)


# In[168]:


clients_cibles.head()


# In[169]:


df_new["proba"].describe()


# In[171]:


df_new["proba"].head()


# In[172]:


df_original = pd.read_csv("Data/clients_a_contacter.csv")

df_original["proba"] = df_new["proba"]
df_original["prediction"] = df_new["prediction"]


# In[173]:


clients_cibles = df_original[
    (df_original["proba"] > 0.3) & (df_original["proba"] < 0.7)
]


# In[174]:


clients_cibles.to_csv("clients_cibles.csv", index=False)


# In[175]:


clients_cibles.head()


# In[176]:


clients_cibles["prediction"].value_counts()


# In[ ]:




