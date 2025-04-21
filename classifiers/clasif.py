import os
import argparse
import datetime
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ===============================
# Configuracion y utilidades
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
    log(f"  PyTorch version: {torch.__version__}")
    log(f"  CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
    log("")


# ===============================
# Modelo CNN
# ===============================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 8 * 24, 128)  # ← CAMBIO AQUÍ
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ===============================
# Carga y preparacion de datos
# ===============================
def cargar_datos(path_csv):
    df = pd.read_csv(path_csv, dtype={"label": str})
    X = df[[f'emb_{i}' for i in range(3072)]].values.astype(np.float32)
    y = df["label"].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Cambiamos de 32x32 a 32x96 para reflejar los 3072 elementos (1x32x96)
    X_train = X_train.reshape(-1, 1, 32, 96)
    X_val = X_val.reshape(-1, 1, 32, 96)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    return train_ds, val_ds, le, num_classes


# ===============================
# Entrenamiento
# ===============================
def entrenar_modelo(model, train_dl, device, log, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        log(f"Epoca {epoch}: Loss = {avg_loss:.4f}")


# ===============================
# Evaluacion
# ===============================
def evaluar_modelo(model, val_dl, le, device, output_dir, log):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds)

    labels = list(range(len(le.classes_)))
    target_names = [str(c) for c in le.classes_]

    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)
    log("\nReporte de resultados:")
    log(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
    plt.title("Matriz de Confusion")
    plt.grid(False)

    # Marca temporal para el archivo
    timestamp = datetime.datetime.now().strftime("%d_%H%M")
    cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")

    plt.savefig(cm_path)
    plt.close()
    log(f"\nMatriz de confusion guardada como '{cm_path}'.")


# ===============================
# Ejecucion principal
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de clasificador CNN con embeddings BirdNET")
    parser.add_argument("--csv", type=str, default="embeddings_csv/all_embeddings.csv", help="Ruta al archivo CSV de embeddings")
    parser.add_argument("--output", type=str, default="outputs", help="Directorio de salida para logs y resultados")
    parser.add_argument("--epochs", type=int, default=20, help="Numero de epocas de entrenamiento")
    args = parser.parse_args()

    log, log_file = configurar_logs(args.output)

    log_hardware(log)

    train_ds, val_ds, le, num_classes = cargar_datos(args.csv)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(num_classes).to(device)

    entrenar_modelo(model, train_dl, device, log, epochs=args.epochs)
    evaluar_modelo(model, val_dl, le, device, args.output, log)

## Ejemplo

#cd ~ BN_R1.5.1

# python Clasif.py --csv embeddings_csv/all_embeddings.csv --output outputs --epochs 20
# python classifiers/Clasif.py --csv embeddings_csv/embeddings_MT_overlap.csv --output outputs --epochs 20
