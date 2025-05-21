import pandas as pd

# Load your CSV into a DataFrame
df = pd.read_csv('AIML Dataset.csv')
print(df.shape)    # debe mostrar (636220, 10)
df.head()
df['isFraud'].value_counts()
# 1. Separa los dos subconjuntos
df_fraud     = df[df['isFraud'] == 1]
df_not_fraud = df[df['isFraud'] == 0]

# 2. Número de fraudes y cálculo del tamaño deseado de no-fraudes
n_frauds = len(df_fraud)
# Queremos que los fraudes sean el 20 %:
desired_fraud_pct = 0.20
# Total final de filas = n_frauds / desired_fraud_pct
total_desired = int(n_frauds / desired_fraud_pct)
# Entonces no-fraudes deseados = total_desired - n_frauds
n_not_frauds = total_desired - n_frauds

print(f"Hay {n_frauds} fraudes; muestreando {n_not_frauds} no-fraudes para alcanzar 20/80.")

# 3. Muestreo aleatorio de las no-fraudes
df_not_fraud_down = df_not_fraud.sample(n=n_not_frauds, random_state=42)

# 4. Une los dos subconjuntos
df_balanced = pd.concat([df_fraud, df_not_fraud_down]).sample(frac=1, random_state=42)  # shuffle

# 5. Comprueba proporciones
print(df_balanced['isFraud'].value_counts(normalize=True))
# Debe mostrar aproximadamente 0.20 para 1 y 0.80 para 0

# 6. (Opcional) Reinicia índices
df_balanced = df_balanced.reset_index(drop=True)

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 1. Cargar y balancear el dataset
df = pd.read_csv('AIML Dataset.csv')
df_fraude = df[df['isFraud'] == 1]
df_no_fraude = df[df['isFraud'] == 0].sample(n=len(df_fraude) * 4, random_state=42)  # Ratio 80/20
df_balanceado = pd.concat([df_fraude, df_no_fraude])

# 2. Preprocesamiento
X = df_balanceado.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
X = pd.get_dummies(X, columns=['type'], drop_first=True)
y = df_balanceado['isFraud']

# Escalar variables numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dividir datos (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. Entrenar modelo
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 5. Obtener probabilidades (no predicciones binarias aún)
y_probs = model.predict_proba(X_test)[:, 1]  # Probabilidad de ser fraude

# 6. Encontrar umbral óptimo para alta precisión (menos falsos positivos)
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
umbral_optimo = thresholds[np.argmax(precision >= 0.9)]  # Ajusta 0.9 según necesidad
print(f"\nUmbral óptimo para precisión >= 90%: {umbral_optimo:.3f}")

# 7. Aplicar umbral ajustado
y_pred_ajustado = (y_probs >= umbral_optimo).astype(int)

# 8. Reportes de evaluación
print("\n=== Reporte con Umbral por Defecto (0.5) ===")
print(classification_report(y_test, (y_probs >= 0.5).astype(int), target_names=["No Fraude", "Fraude"]))

print("\n=== Reporte con Umbral Ajustado ===")
print(classification_report(y_test, y_pred_ajustado, target_names=["No Fraude", "Fraude"]))

print("\n=== Matriz de Confusión (Umbral Ajustado) ===")
print(confusion_matrix(y_test, y_pred_ajustado))

# 9. Gráficas mejoradas
plt.figure(figsize=(18, 5))

# Gráfica 1: Distribución de probabilidades
plt.subplot(1, 3, 1)
plt.hist(y_probs[y_test == 0], bins=30, alpha=0.5, color='blue', label='No Fraude')
plt.hist(y_probs[y_test == 1], bins=30, alpha=0.5, color='red', label='Fraude')
plt.axvline(x=umbral_optimo, color='black', linestyle='--', label=f'Umbral: {umbral_optimo:.2f}')
plt.xlabel("Probabilidad de Fraude")
plt.ylabel("Frecuencia")
plt.title("Distribución de Probabilidades Predichas")
plt.legend()

# Gráfica 2: Curva ROC
plt.subplot(1, 3, 2)
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("Curva ROC")
plt.legend()

# Gráfica 3: Trade-off Precisión-Recall
plt.subplot(1, 3, 3)
plt.plot(thresholds, precision[:-1], 'b--', label='Precisión')
plt.plot(thresholds, recall[:-1], 'g-', label='Recall')
plt.axvline(x=umbral_optimo, color='r', linestyle='--', label=f'Umbral óptimo: {umbral_optimo:.2f}')
plt.xlabel("Umbral de Decisión")
plt.ylabel("Score")
plt.title("Trade-off Precisión vs Recall")
plt.legend()

plt.tight_layout()
plt.show()

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

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class InterfazFraude:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Detector de Fraude")
        self.window.geometry("600x800")
        
        # Variables para los modelos
        self.modelos = {
            'Árbol de Decisión': None,
            'Regresión Logística': None,
            'SVM': None
        }
        
        self.crear_interfaz()
        self.cargar_modelos()
        
    def crear_interfaz(self):
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(main_frame, text="Datos de la Transacción:", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=10)
        
        # Campos numéricos
        campos_numericos = [
            ('Step:', 'step'),
            ('Amount:', 'amount'),
            ('Old Balance Origin:', 'oldbalanceOrg'),
            ('New Balance Origin:', 'newbalanceOrig'),
            ('Old Balance Dest:', 'oldbalanceDest'),
            ('New Balance Dest:', 'newbalanceDest')
        ]
        
        self.campos_numericos = {}
        row = 1
        for texto, nombre in campos_numericos:
            ttk.Label(main_frame, text=texto).grid(row=row, column=0, pady=5, sticky=tk.W)
            entrada = ttk.Entry(main_frame, justify='right')
            entrada.insert(0, '0.0')
            entrada.grid(row=row, column=1, pady=5)
            self.campos_numericos[nombre] = entrada
            row += 1
        
        # Separador
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # Tipo de transacción
        ttk.Label(main_frame, text="Tipo de Transacción:").grid(row=row, column=0, pady=5)
        self.tipo_var = tk.StringVar()
        tipos = ['TRANSFER', 'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT']
        menu_tipos = ttk.OptionMenu(main_frame, self.tipo_var, tipos[0], *tipos)
        menu_tipos.grid(row=row, column=1, pady=5)
        row += 1
        
        # Botón predecir
        ttk.Button(main_frame, text="Predecir", command=self.predecir).grid(row=row, column=0, columnspan=2, pady=20)
        row += 1
        
        # Separador
        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # Resultados
        ttk.Label(main_frame, text="Resultados de Predicción:", font=('Arial', 12, 'bold')).grid(
            row=row, column=0, columnspan=2, pady=10)
        row += 1

        self.resultados_text = tk.Text(main_frame, height=10, width=50)
        self.resultados_text.grid(row=row, column=0, columnspan=2)

        # Configuración de colores en el texto
        self.resultados_text.tag_configure("fraude", foreground="red")
        self.resultados_text.tag_configure("nofraude", foreground="green")
        
    def cargar_modelos(self):
        # Columnas
        numeric_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                        'oldbalanceDest', 'newbalanceDest']
        categorical_cols = ['type']
        
        # Preprocesamiento
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
        
        # Modelos
        self.modelos['Árbol de Decisión'] = Pipeline([
            ('prep', preprocessor),
            ('tree', DecisionTreeClassifier(
                criterion='gini', max_depth=6, class_weight='balanced', random_state=42))
        ])
        
        self.modelos['Regresión Logística'] = Pipeline([
            ('prep', preprocessor),
            ('logreg', LogisticRegression(
                class_weight='balanced', max_iter=1000, random_state=42))
        ])
        
        self.modelos['SVM'] = Pipeline([
            ('prep', preprocessor),
            ('svm', SVC(
                kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
        ])
        
        # Datos de entrenamiento (ya balanceados)
        X = df_balanced.drop(['isFraud', 'isFlaggedFraud'], axis=1)
        y = df_balanced['isFraud']
        
        for modelo in self.modelos.values():
            modelo.fit(X, y)
        
    def preparar_datos(self):
        datos = {}
        for campo, entrada in self.campos_numericos.items():
            valor = entrada.get()
            if not valor:
                raise ValueError(f"El campo {campo} está vacío")
            try:
                float_val = float(valor)
                if float_val < 0:
                    raise ValueError(f"El campo {campo} no puede ser negativo")
                datos[campo] = [float_val]
            except ValueError:
                raise ValueError(f"Valor no válido en el campo {campo}")
        
        datos['type'] = [self.tipo_var.get()]
        return pd.DataFrame(datos)
        
    def predecir(self):
        try:
            self.resultados_text.delete(1.0, tk.END)
            
            if not self.tipo_var.get():
                raise ValueError("Por favor seleccione un tipo de transacción")
            
            X_new = self.preparar_datos()
            for nombre, modelo in self.modelos.items():
                prediccion = modelo.predict(X_new)[0]
                probabilidad = modelo.predict_proba(X_new)[0][1]
                texto = f"{nombre}: {'Fraude' if prediccion == 1 else 'No Fraude'} " \
                        f"(Probabilidad: {probabilidad:.2%})\n"
                
                tag = "fraude" if prediccion == 1 else "nofraude"
                self.resultados_text.insert(tk.END, texto, tag)
                
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error inesperado: {str(e)}")
            
    def iniciar(self):
        self.window.mainloop()

# Crear y ejecutar la aplicación
if __name__ == "__main__":
    app = InterfazFraude()
    app.iniciar()
