import os
import argparse
import datetime
import platform
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# ===============================
# Configuración y utilidades
# ===============================
def configurar_logs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'log_{datetime.datetime.now().strftime("%d_%H%M")}.txt')
    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    return log, log_file

def log_hardware(log):
    log("\nResumen del hardware:")
    log(f"  Plataforma: {platform.system()} {platform.release()}")
    log(f"  Procesador: {platform.processor()}")
    log(f"  XGBoost version: {xgb.__version__}")
    log("")


# ===============================
# Carga y preparación de datos
# ===============================
def cargar_datos(path_csv):
    df = pd.read_csv(path_csv, dtype={"label": str})
    X = df[[f'emb_{i}' for i in range(1024)]].values.astype(np.float32)
    y = df["label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    return X_train, X_val, y_train, y_val, le, num_classes


# ===============================
# Entrenamiento
# ===============================
def entrenar_modelo_xgb(X_train, y_train, X_val, y_val, num_classes, output_dir, log):
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",  # usa GPU si quieres: 'gpu_hist'
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        verbosity=1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
    model_path = os.path.join(output_dir, "xgb_model.json")
    model.save_model(model_path)
    log(f"Modelo guardado en: {model_path}")
    return model


# ===============================
# Evaluación
# ===============================
def evaluar_modelo(model, X_val, y_val, le, output_dir, log):
    y_pred = model.predict(X_val)

    labels = list(range(len(le.classes_)))
    target_names = le.inverse_transform(labels)

    report = classification_report(y_val, y_pred, labels=labels, target_names=target_names, zero_division=0)
    log("\nReporte de resultados:")
    log(report)

    cm = confusion_matrix(y_val, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
    plt.title("Matriz de Confusión")
    plt.grid(False)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    log(f"Matriz de confusión guardada en: {cm_path}")


# ===============================
# Ejecución principal
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de clasificador XGBoost con embeddings BirdNET")
    parser.add_argument("--csv", type=str, default="embeddings_csv/all_embeddings.csv", help="Ruta al archivo CSV de embeddings")
    parser.add_argument("--output", type=str, default="outputs", help="Directorio de salida para logs y resultados")
    args = parser.parse_args()

    log, log_file = configurar_logs(args.output)
    log_hardware(log)

    X_train, X_val, y_train, y_val, le, num_classes = cargar_datos(args.csv)
    model = entrenar_modelo_xgb(X_train, y_train, X_val, y_val, num_classes, args.output, log)

    # Guardar LabelEncoder
    le_path = os.path.join(args.output, "label_encoder.pkl")
    joblib.dump(le, le_path)
    log(f"LabelEncoder guardado en: {le_path}")

    evaluar_modelo(model, X_val, y_val, le, args.output, log)
