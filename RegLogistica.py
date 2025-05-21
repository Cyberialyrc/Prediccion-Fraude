import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 1. Cargar y balancear el dataset
df = pd.read_csv('/content/drive/MyDrive/Datas/AIML Dataset.csv')
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

model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 5. Obtener probabilidades (no predicciones binarias aún)
y_probs = model.predict_proba(X_test)[:, 1]

# 6. Encontrar umbral óptimo para alta precisión (menos falsos positivos)
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
umbral_optimo = thresholds[np.argmax(precision >= 0.9)]  
print(f"\nUmbral óptimo para precisión >= 90%: {umbral_optimo:.3f}")
y_pred_ajustado = (y_probs >= umbral_optimo).astype(int)

# 8. Reportes de evaluación
print("\n=== Reporte con Umbral por Defecto (0.5) ===")
print(classification_report(y_test, (y_probs >= 0.5).astype(int), target_names=["No Fraude", "Fraude"]))

print("\n=== Reporte con Umbral Ajustado ===")
print(classification_report(y_test, y_pred_ajustado, target_names=["No Fraude", "Fraude"]))

print("\n=== Matriz de Confusión (Umbral Ajustado) ===")
print(confusion_matrix(y_test, y_pred_ajustado))
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