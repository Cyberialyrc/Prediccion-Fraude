# Importar librerías necesarias
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Preparación de datos
# Separar características (X) y variable objetivo (y)
X = df_balanced.drop(['isFraud', 'isFlaggedFraud'], axis=1)  # Excluimos la variable objetivo y la bandera
y = df_balanced['isFraud']

# 2. Preprocesamiento
# Identificar columnas categóricas y numéricas
categorical_cols = ['type']
numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Crear transformadores
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combinar transformadores en un preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 3. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Crear pipeline con SVM
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
])

# 5. Entrenar el modelo
svm_pipeline.fit(X_train, y_train)

# 6. Evaluar el modelo
# Predecir en el conjunto de prueba
y_pred = svm_pipeline.predict(X_test)

# Mostrar métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Vamos a seleccionar solo algunas columnas numéricas clave para visualizar
pairplot_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']

# Crear un subset del dataframe balanceado
df_pairplot = df_balanced[pairplot_cols].sample(n=1000, random_state=42)  # Muestreo para hacerlo manejable

# Convertir isFraud a string para mejor visualización
df_pairplot['isFraud'] = df_pairplot['isFraud'].astype(str)

# 2. Crear el pairplot
sns.pairplot(df_pairplot, hue='isFraud',
             palette={ '1': 'red', '0': 'blue' },
             plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
             height=2.5)

plt.suptitle('Pairplot de Variables Numéricas por Estado de Fraude', y=1.02)
plt.show()
