#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime
from datetime import date

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[192]:


df = pd.read_csv("Data/train_info.csv")
df_second = pd.read_csv("Data/clients_a_contacter.csv")


# # Partie 1 : EDA Exploration des données  

# In[19]:


df.head()


# In[20]:


df.columns


# In[21]:


df.info()


# In[22]:


numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
print("Numerical columns:", numerical_columns)


# In[23]:


df.describe()


# #### la plus part des client entre 20 - 50
# #### la majourite de 0.997 sont des permis les autres propablement des erreur 812/381109
# #### 46% des client sont deja assures 
# #### mouene des prime_annuelle envirant de 30564 entre 2630 et 39400 
# #### personne n a encore depase le 300j d anciennte 
# #### moins que 13% qui veux faire l insscription 

# In[26]:


df.duplicated().sum()


# In[27]:


df.isna().sum()


# #### Le dataset ne contient ni valeurs manquantes ni doublons, ce qui indique une bonne qualité des données.

# In[28]:


df.shape


# In[29]:


numerical_columns.remove("id_client")


# In[30]:


len(numerical_columns)


# In[31]:


fig, axes = plt.subplots(3, 3, figsize=(12, 10))  # largeur, hauteur en pouces
axes = axes.flatten()

for i, col in enumerate(numerical_columns[:9]):
    axes[i].boxplot(df[col], labels=[col])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


# #### donc le prime annuelle ce lui qui contient plus de decalage et des valeurs aberrantes </br> possible parceque on plus de type d assurence et peux changer depend de type de voiture 

# In[32]:


# pour prime annuelle
Q1 = df["prime_annuelle"].quantile(0.25)
Q3 = df["prime_annuelle"].quantile(0.75)
IQR = Q3 - Q1
born_inf = Q1 - 1.5 * IQR
born_sup = Q3 + 1.5 * IQR
print(f"Bornes pour prime_annuelle : [{born_inf}, {born_sup}]")


# In[33]:


df.loc[(df["prime_annuelle"] < born_inf) | (df["prime_annuelle"] > born_sup)].shape


# In[35]:


fig, axes = plt.subplots(3, 3, figsize=(12, 10))  
axes = axes.flatten()

for i, col in enumerate(numerical_columns[:9]):
    axes[i].hist(df[col] , bins=20)
    axes[i].set_title(col + " Distribution")

plt.tight_layout()
plt.show()


# In[36]:


sns.countplot(x=df["reponse_client"])
plt.title("Distribution de la variable cible")
plt.show()

df["reponse_client"].value_counts(normalize=True)


# In[37]:


# Variables catégorielles
categorical_columns = df.select_dtypes(include='object').columns
print("Categorical columns:", categorical_columns)


# In[38]:


for col in categorical_columns:
    print(df[col].value_counts())


# In[39]:


fg , axes = plt.subplots(1, 3 , figsize=(15, 5))  
for i , col in enumerate(categorical_columns):
    sns.countplot(x=df[col], ax=axes[i])
    axes[i].set_title(col)


# In[40]:


df.describe(include="object")


# In[41]:


categorical_columns


# In[42]:


df[['genre', 'age_vehicule', 'vehicule_endommage','reponse_client']]


# In[44]:


df.groupby(['genre', 'age_vehicule', 'vehicule_endommage'])['reponse_client'].value_counts(normalize=True).unstack().fillna(0)


# In[45]:


# on peux ingorer pour le moment le genre 
df.groupby([ 'age_vehicule', 'vehicule_endommage'])['reponse_client'].value_counts(normalize=True).unstack().fillna(0)


# #### Plus le véhicule est ancien et endommagé, plus la probabilité que le client réponde positivement augmente.

# In[46]:


fg , axes = plt.subplots(1, 3 , figsize=(15, 5))
for i, col in enumerate(categorical_columns):
    sns.countplot(x=df[col], hue=df["reponse_client"], ax=axes[i])
    axes[i].set_title(col)


# #### Cette visualisation confirme que les clients ayant un véhicule endommagé sont significativement plus représentés parmi ceux ayant répondu positivement.
# 
# La variable vehicule_endommage montre une forte influence sur la réponse du client. Les clients ayant un véhicule déjà endommagé présentent une proportion significativement plus élevée de réponses positives comparé à ceux dont le véhicule n’est pas endommagé. Cette variable apparaît donc comme un facteur clé dans la décision du client.

# ### Visualisation des variables quantitatives

# In[47]:


len(numerical_columns)


# In[48]:


fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(numerical_columns):
    sns.kdeplot(data=df, x=col, hue="reponse_client", ax=axes[i], fill=True)
    axes[i].set_title(col + " Distribution par reponse_client")


# #### Le fait d’être déjà assuré influence fortement la réponse : les clients non assurés sont beaucoup plus susceptibles de répondre positivement.
# #### La prime annuelle présente une forte asymétrie et des valeurs aberrantes. Son influence sur la réponse client semble limitée.
# 
# #### en generale . Parmi les variables quantitatives, le statut d’assurance (ancien_assure) apparaît comme le facteur le plus discriminant. En revanche, des variables comme permis_conduire, code_regional et anciennete montrent peu d’influence sur la réponse du client. Certaines variables comme l’âge et le canal de communication présentent une influence modérée.
# 

# In[49]:


# identifier correlations entre les variables numériques entre eux et la variable cible
matrix_corr = df[numerical_columns].corr(method="pearson")
plt.figure(figsize=(10, 8))
sns.heatmap(matrix_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matrice de corrélation")


# #### La matrice de corrélation montre que la variable la plus liée à la réponse du client est le fait d’être déjà assuré, avec une corrélation négative modérée (-0.34), indiquant que les clients non assurés sont plus susceptibles de répondre positivement. </br> Les autres variables présentent des corrélations faibles avec la cible, ce qui suggère une influence limitée.Par ailleurs, une corrélation notable est observée entre l’âge et le canal de communication (-0.58), indiquant que le choix du canal dépend fortement de l’âge du client.

# In[50]:


sns.scatterplot(data=df, x="age", y="prime_annuelle", hue="reponse_client")


# In[51]:


sns.scatterplot(data=df, x="age", y="anciennete", hue="reponse_client")


# In[52]:


sns.boxplot(data=df, x="age", y="vehicule_endommage", hue="reponse_client")


# L’analyse de l’âge en fonction de l’état du véhicule et de la réponse client montre que l’âge a une influence modérée lorsque le véhicule est endommagé, les clients plus âgés étant légèrement plus susceptibles de répondre positivement. En revanche, lorsque le véhicule n’est pas endommagé, cette influence devient négligeable.

# In[53]:


sns.boxplot(data=df, x="anciennete", y="vehicule_endommage", hue="reponse_client")


# L’analyse de l’ancienneté en fonction de l’état du véhicule et de la réponse client montre des distributions très similaires entre les différents groupes. Cela suggère que l’ancienneté n’a pas d’impact significatif sur la décision du client.

# In[54]:


sns.countplot(x=df["age_vehicule"], hue=df["reponse_client"])


# In[55]:


sns.countplot(x=df["vehicule_endommage"], hue=df["reponse_client"])


# In[56]:


df.head()


# In[57]:


df["ancien_assure"].value_counts()


# In[58]:


df["reponse_client"].value_counts()


# #### L’analyse de la variable cible révèle un fort déséquilibre entre les classes, avec une proportion largement majoritaire de clients n’ayant pas répondu positivement. Ce déséquilibre peut biaiser les performances du modèle en favorisant la classe majoritaire. Il sera donc nécessaire d’adopter des métriques adaptées et éventuellement des techniques de rééquilibrage afin d’améliorer la prédiction de la classe minoritaire.

# In[59]:


df["code_regional"].unique()


# In[60]:


df["canal_communication"].unique()


# In[61]:


df_second.head()


# In[62]:


df.shape


# In[63]:


df_second.shape


# # Feature Engineering

# In[64]:


df["age"].unique()


# In[65]:


df["genre"].unique()
df["genre"].value_counts()


# ##### La variable genre a été encodée en binaire afin de la rendre exploitable par les modèles de machine learning, avec 1 pour les hommes et 0 pour les femmes. 

# In[193]:


from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
df["genre_labeled"] = lab.fit_transform(df["genre"])


# In[194]:


df.head()   


# In[195]:


df["genre_labeled"].value_counts()    


# In[202]:


df.drop(columns=["genre_labeled"], inplace=True)


# In[196]:


df["age"].describe()


# ##### creer tranche age de 20 a 30  -> 1
# 30 , 40 -> 2 
# ...
# 70 , 80 -> 6
# > 80 -> 7

# In[197]:


tranch ={20:1, 30:2,40:3 ,50:4 ,60:5, 70:6, 80:7}


# In[198]:


df["tranche_age"] = pd.cut(
    df["age"],
    bins=[0, 30, 40, 50, 60, 70, 80 , 100],
    labels=["20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"]
)


# In[199]:


df.head()


# In[203]:


df.isna().sum()


# In[204]:


df["age_vehicule"].value_counts()


# In[205]:


df["vehicule_endommage"].value_counts()


# In[206]:


df["ancien_assure"].value_counts()


# In[207]:


df["tranche_age_age_vehicule"] = df["tranche_age"].astype(str) + "_" + df["age_vehicule"].astype(str)
df["tranche_age_vehicule_endommage"] = df["tranche_age"].astype(str) + "_" + df["vehicule_endommage"].astype(str)
df["tranche_age_ancien_assure"] = df["tranche_age"].astype(str) + "_" + df["ancien_assure"].astype(str)
df["age_vehicule_vehicule_endommage"] = df["age_vehicule"].astype(str) + "_" + df["vehicule_endommage"].astype(str)
df["age_vehicule_ancien_assure"] = df["age_vehicule"].astype(str) + "_" + df["ancien_assure"].astype(str)
df["vehicule_endommage_ancien_assure"] = df["vehicule_endommage"].astype(str) + "_" + df["ancien_assure"].astype(str)


# In[208]:


df.groupby("age_vehicule_ancien_assure")["reponse_client"].value_counts(normalize=True).unstack().fillna(0).sort_values(by=1, ascending=False)


# In[209]:


df.groupby("vehicule_endommage_ancien_assure")["reponse_client"].value_counts(normalize=True).unstack().fillna(0).sort_values(by=1, ascending=False)


# In[210]:


df.head()


# In[211]:


df.groupby("tranche_age_age_vehicule")["reponse_client"].value_counts(normalize=True).unstack().fillna(0).sort_values(by=1, ascending=False)


# In[212]:


df.groupby("tranche_age_vehicule_endommage")["reponse_client"].value_counts(normalize=True).unstack().fillna(0).sort_values(by=1, ascending=False)


# In[213]:


df.groupby("age_vehicule_vehicule_endommage")["reponse_client"].value_counts(normalize=True).unstack().fillna(0).sort_values(by=1, ascending=False)


# In[214]:


df.groupby("tranche_age_ancien_assure")["reponse_client"].value_counts(normalize=True).unstack().fillna(0).sort_values(by=1, ascending=False)


# In[215]:


from sklearn.preprocessing import OneHotEncoder
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
onehot = OneHotEncoder(drop="first", sparse_output=False)
encoded = onehot.fit_transform(df[onehot_cols])
encoded_df = pd.DataFrame(
    encoded,
    columns=onehot.get_feature_names_out(onehot_cols)
)


# In[216]:


encoded_df.head()


# In[217]:


df.columns


# In[218]:


df.shape


# In[219]:


encoded_df.shape


# In[220]:


# One-Hot variables
encoded_df.info()


# In[221]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_cols = ["age_vehicule"]

ordinal = OrdinalEncoder()
df[ordinal_cols] = ordinal.fit_transform(df[ordinal_cols])


# In[222]:


df.loc[:, ["age_vehicule"]]


# In[223]:


encoded_df.shape


# In[224]:


df_final = pd.concat([df, encoded_df], axis=1)


# In[225]:


df_final.head()


# In[226]:


df_final.shape


# In[227]:


# Encodage des variables à forte cardinalité
mean_target = df.groupby("canal_communication")["reponse_client"].mean()
df["canal_communication_encoded"] = df["canal_communication"].map(mean_target)


# In[228]:


mean_target = df.groupby("code_regional")["reponse_client"].mean()
mean_target.sort_values(ascending=False)
df["code_regional_encoded"] = df["code_regional"].map(mean_target)


# In[229]:


df.head()


# In[230]:


df = pd.concat([df.drop(columns=onehot_cols), encoded_df], axis=1)


# In[231]:


df.shape


# In[232]:


df.head()


# In[233]:


for col in df.columns:
    if df[col].dtype == "object":
        print(f"Colonne '{col}' is still object type")
    


# In[234]:


df.drop(columns=["code_regional", "id_client","canal_communication"], inplace=True)


# In[239]:


df.shape


# In[263]:


df.head()


# In[ ]:


## split data into train and test
from sklearn.model_selection import train_test_split
X = df.drop(columns=["reponse_client"])
y = df["reponse_client"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[264]:


X_train.head()


# In[248]:


# Standardisation des variables numériques
from sklearn.preprocessing import StandardScaler , MinMaxScaler , RobustScaler
# =========================
# 3. Définir les colonnes
# =========================

# Colonnes pour MinMaxScaler (valeurs normales)
minmax_cols = ["age", "permis_conduire"]

# Colonnes pour RobustScaler (avec outliers)
robust_cols = ["prime_annuelle"]

# Colonnes restantes → StandardScaler
# On enlève celles déjà utilisées
standard_cols = [
    col for col in X_train.columns
    if col not in minmax_cols + robust_cols
]

# =========================
# 4. Initialiser les scalers
# =========================
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()
standard_scaler = StandardScaler()

# =========================
# 5. Appliquer les scalers
# =========================

# MinMaxScaler
X_train[minmax_cols] = minmax_scaler.fit_transform(X_train[minmax_cols])
X_test[minmax_cols] = minmax_scaler.transform(X_test[minmax_cols])

# RobustScaler
X_train[robust_cols] = robust_scaler.fit_transform(X_train[robust_cols])
X_test[robust_cols] = robust_scaler.transform(X_test[robust_cols])

# StandardScaler
X_train[standard_cols] = standard_scaler.fit_transform(X_train[standard_cols])
X_test[standard_cols] = standard_scaler.transform(X_test[standard_cols])


# In[249]:


X_train.head()


# In[250]:


X_train.shape


# In[251]:


X_test.shape


# In[252]:


df["reponse_client"].value_counts()


# In[253]:


X_train.columns = X_train.columns.str.replace('[\\[\\]<>]', '_', regex=True)
X_test.columns = X_test.columns.str.replace('[\\[\\]<>]', '_', regex=True)


# In[254]:


import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

# Création des matrices XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Paramètres (corrigés)
params = {
    'objective': 'binary:logistic',  # ✅ classification binaire
    'max_depth': 4,
    'eta': 0.1,  # learning rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',  # métrique interne
    'seed': 42
}

# Entraînement
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

# =========================
# Prédictions
# =========================

y_pred_proba = model.predict(dtest)  # probabilités
y_pred = (y_pred_proba > 0.5).astype(int)  # seuil 0.5

# =========================
# Évaluation
# =========================

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Le modèle prédit presque tout en classe 0
# Il ignore complètement la classe 1

# In[ ]:





# In[255]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))


# In[256]:


print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))


# In[257]:


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[258]:


scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print("Scale_pos_weight:", scale_pos_weight)


# In[259]:


# Gestion du déséquilibre
# Approche 1 : pondération des classes
params = {
    'objective': 'binary:logistic',  
    'max_depth': 4,
    'eta': 0.1,  # learning rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',  
    'seed': 42, 
    'scale_pos_weight': scale_pos_weight  
}

# Entraînement
num_rounds = 100
model_xgb = xgb.train(params, dtrain, num_rounds)

# =========================
# Prédictions
# =========================

y_pred_proba = model_xgb.predict(dtest)  # probabilités
y_pred = (y_pred_proba > 0.5).astype(int)  # seuil 0.5

# =========================
# Évaluation
# =========================

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Le premier modèle présente une accuracy élevée, mais échoue à détecter la classe minoritaire, comme en témoigne un F1-score quasi nul.
# En revanche, après prise en compte du déséquilibre des classes, le second modèle améliore significativement le rappel et le F1-score, au prix d’une baisse de l’accuracy.
# Ce compromis est acceptable dans notre contexte, où la détection des clients susceptibles de répondre est prioritaire.

# mais on a toujour probleme dans la precession on predecte tros des positives alorque que c est moins en vrai 

# In[260]:


# poir ameliorer la precession je vais chnager le seuil de decision de 0.5 a 0.7
y_pred = (y_pred_proba > 0.7).astype(int)  # seuil 0.7
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[156]:


# Opptimisser les hyperparametres avec GridSearchCV 
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Modèle
xgb_model = XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

# Grille des paramètres
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# GridSearch
grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1',  
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Entraînement
grid.fit(X_train, y_train)

# Résultats
print("Best parameters:", grid.best_params_)
print("Best F1-score:", grid.best_score_)


# In[162]:


import  numpy as np
best_model = grid.best_estimator_
y_proba = best_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1,0.9, 0.05)
best_f1 = 0
best_thresh = 0

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    score = f1_score(y_test, y_pred)
    
    if score > best_f1:
        best_f1 = score
        best_thresh = t

print("Best threshold:", best_thresh)
print("Best F1:", best_f1)


# In[170]:


# # Opptimisser les hyperparametres avec GridSearchCV 
# from sklearn.model_selection import GridSearchCV
# from xgboost import XGBClassifier

# # Modèle
# xgb_model = XGBClassifier(
#     objective='binary:logistic',
#     random_state=42,
#     scale_pos_weight=scale_pos_weight
# )

# # Grille des paramètres
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.7, 0.8, 1],
#     'colsample_bytree': [0.7, 0.8, 1],
#     'gamma': [0, 1, 5],              
#     'min_child_weight': [1, 5, 10]   
# }

# # GridSearch
# grid = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     scoring='f1',  
#     cv=3,
#     verbose=1,
#     n_jobs=-1
# )

# # Entraînement avec aerly stopping
# grid.fit(
#     X_train, y_train,
#     eval_set=[(X_test, y_test)],
#     verbose=False
# )

# # Résultats
# print("Best parameters:", grid.best_params_)
# print("Best F1-score:", grid.best_score_)


# In[164]:


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import numpy as np

xgb_model = XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1],
    'gamma': [0, 1, 5],
    'min_child_weight': [1, 5, 10]
}

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=20,  # 🔥 seulement 20 essais au lieu de 6000+
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best params:", random_search.best_params_)
print("Best F1:", random_search.best_score_)


# In[166]:


best_model = random_search.best_estimator_
y_proba = best_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1,0.9, 0.05)
best_f1 = 0
best_thresh = 0

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    score = f1_score(y_test, y_pred)
    
    if score > best_f1:
        best_f1 = score
        best_thresh = t

print("Best threshold:", best_thresh)
print("Best F1:", best_f1)


# In[167]:


# importance des variables
importance = best_model.feature_importances_

# créer un dataframe
feat_imp = pd.Series(importance, index=X_train.columns)

# trier
feat_imp = feat_imp.sort_values(ascending=False)

# afficher top 10
print(feat_imp.head(10))

# visualisation
feat_imp.head(10).plot(kind='barh')
plt.title("Top 10 des variables importantes")
plt.gca().invert_yaxis()
plt.show()


# L’analyse de l’importance des variables met en évidence que les caractéristiques liées à l’état du véhicule et au statut d’ancien assuré jouent un rôle majeur dans la prédiction de la réponse des clients.
# 
# En particulier, la variable combinée vehicule_endommage_ancien_assure apparaît comme la plus influente, indiquant que l’interaction entre un véhicule endommagé et le statut d’assurance est déterminante dans le comportement des clients. Cela suggère que certains profils spécifiques de clients, notamment ceux ayant un véhicule endommagé et n’étant pas anciens assurés, sont plus susceptibles de répondre.
# 
# De plus, la variable ancien_assure seule possède également une importance significative, montrant que l’historique du client avec l’assurance influence fortement sa probabilité de réponse.
# 
# L’état du véhicule (vehicule_endommage) constitue également un facteur important, ce qui peut s’expliquer par un besoin accru de couverture chez les clients dont le véhicule a subi des dommages.
# 
# Les variables liées à l’âge, telles que tranche_age, ont un impact plus faible mais non négligeable, suggérant que certaines classes d’âge présentent des comportements légèrement différents.
# 
# Enfin, la variable canal_communication_encoded a une influence limitée, ce qui indique que le canal de communication joue un rôle secondaire par rapport aux caractéristiques liées au client et à son véhicule.
# 
# Globalement, ces résultats montrent que le modèle s’appuie principalement sur des facteurs liés au véhicule et à l’historique client pour prédire la réponse, ce qui est cohérent avec le contexte métier.

# La technique SMOTE a été utilisée afin de générer artificiellement de nouveaux exemples de la classe minoritaire, permettant ainsi de réduire le déséquilibre des données et d’améliorer la capacité du modèle à détecter les clients susceptibles de répondre.

# In[169]:


# XGBoost + SMOTE
from imblearn.over_sampling import SMOTE


# SMOTE uniquement sur TRAIN
smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Vérification
print("Avant SMOTE:", y_train.value_counts())
print("Après SMOTE:", y_train_res.value_counts())

# Entraînement
model = XGBClassifier(
    objective='binary:logistic',
    random_state=42
)

model.fit(X_train_res, y_train_res)

# Prédiction (SUR TEST NORMAL)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[171]:


print(classification_report(y_test, y_pred))


# In[172]:


# =========================
# Modèle avec meilleurs paramètres
# =========================
model = XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.7,
    gamma=1,
    min_child_weight=1,
    n_jobs=-1
)
# =========================
# Entraînement
# =========================
model.fit(X_train_res, y_train_res)

# =========================
# Prédictions (probabilités)
# =========================
y_proba = model.predict_proba(X_test)[:, 1]

# =========================
#  Seuil optimisé
# =========================
threshold = 0.65
y_pred = (y_proba > threshold).astype(int)

# =========================
#  Évaluation
# =========================
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# la methode SMOT est rendre l accurancy plus faible que avant 

# Le modèle final retenu est un modèle XGBoost (XGBClassifier) optimisé avec les paramètres suivants : une profondeur maximale de 4, un taux d’apprentissage de 0.1, un sous-échantillonnage de 0.8, ainsi qu’un paramètre scale_pos_weight fixé à 7 afin de prendre en compte le déséquilibre des classes.
# 
# Afin d’améliorer le compromis entre précision et rappel, un seuil de décision de 0.7 a été appliqué sur les probabilités de sortie du modèle.
# 
# Les performances obtenues montrent une accuracy de 0.79 et un F1-score de 0.47 pour la classe minoritaire, ce qui constitue une amélioration notable par rapport aux modèles précédents.
# 
# Plus précisément, le modèle atteint une précision de 0.35 et un rappel de 0.73 pour la classe des clients répondants. Cela signifie que le modèle est capable de détecter une proportion importante de clients susceptibles de répondre, bien qu’il génère encore un certain nombre de faux positifs.
# 
# Ce compromis est jugé acceptable dans le contexte métier, où l’objectif principal est de maximiser la détection des clients intéressés, même au prix d’une précision plus faible.
# 
# Ainsi, ce modèle offre un bon équilibre entre détection et fiabilité, ce qui en fait le meilleur choix pour la suite du projet.

# In[261]:


## instaler les methodes utiliser dans le projet pour reualiser le projet sur d'autres dataset 
import joblib

joblib.dump(model_xgb, "model.pkl")
joblib.dump(onehot, "onehot.pkl")
joblib.dump(ordinal, "ordinal.pkl")
joblib.dump(minmax_scaler, "minmax.pkl")
joblib.dump(robust_scaler, "robust.pkl")
joblib.dump(standard_scaler, "standard.pkl")


# In[262]:


## puisuqe on a pas reponse client dans le dataset de test on va faire des prediction sur ce dataset et on va les sauvegarder dans un fichier csv pour les utiliser dans le projet finaljoblib.dump(mean_canal, "mean_canal.pkl")
df_initial = pd.read_csv("Data/train_info.csv")
mean_canal = df_initial.groupby("canal_communication")["reponse_client"].mean()
mean_region = df_initial.groupby("code_regional")["reponse_client"].mean()
joblib.dump(mean_canal, "mean_canal.pkl")
joblib.dump(mean_region, "mean_region.pkl")


# In[265]:


# sauvegarder l’ordre des colonnes
joblib.dump(X_train.columns, "columns_order.pkl")


# In[267]:


joblib.dump(standard_cols, "standard_cols.pkl")
joblib.dump(minmax_cols, "minmax_cols.pkl")
joblib.dump(robust_cols, "robust_cols.pkl")


# In[ ]:




