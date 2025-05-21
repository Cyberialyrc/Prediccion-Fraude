import pandas as pd
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import OneHotEncoder, StandardScaler
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline
from sklearn.tree              import DecisionTreeClassifier, plot_tree
from sklearn.metrics           import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1) Variables predictoras y etiqueta
X = df_balanced.drop(columns=[
    'isFraud',        # etiqueta
    'isFlaggedFraud', # opcional
    'nameOrig',       # ID (texto)
    'nameDest'        # ID (texto)
])
y = df_balanced['isFraud']

# 2) Train/test split (manteniendo proporciones)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 3) Especifica columnas
cat_cols = ['type']
num_cols = [c for c in X.columns if c not in cat_cols]

# 4) Preprocesador
preprocessor = ColumnTransformer([
    ('ohe',   OneHotEncoder(drop='first', sparse_output=False), cat_cols),
    ('scale', StandardScaler(),                              num_cols)
])

# 5) Pipeline
clf = Pipeline([
    ('prep', preprocessor),
    ('tree', DecisionTreeClassifier(
        criterion='gini',
        max_depth=6,
        class_weight='balanced',
        random_state=42))
])

# 6) Entrena
clf.fit(X_train, y_train)

# 7) Evalúa
y_pred = clf.predict(X_test)
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, digits=4))

# 8) Visualiza el árbol (primeras 3 capas)
plt.figure(figsize=(20,10))
plot_tree(
    clf.named_steps['tree'],
    feature_names=(
        list(clf.named_steps['prep']
                .named_transformers_['ohe']
                .get_feature_names_out(cat_cols))
        + num_cols
    ),
    class_names=['No Fraud','Fraud'],
    filled=True, rounded=True,
    max_depth=3
)
plt.show()
