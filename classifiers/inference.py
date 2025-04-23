import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

# ===============================
# Definición del modelo CNN original
# ===============================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 8 * 24, 128)  # input: (1, 32, 96)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# ===============================
# Script principal
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Inferencia con CNN y embeddings BirdNET")
    parser.add_argument("--csv", type=str, required=True, help="Ruta al archivo CSV con los embeddings")
    parser.add_argument("--modelo", type=str, required=True, help="Ruta al archivo del modelo (.pt)")
    parser.add_argument("--labels", type=str, required=True, help="Ruta al archivo LabelEncoder (.pkl)")
    parser.add_argument("--output", type=str, default="predicciones.csv", help="Archivo de salida")
    args = parser.parse_args()

    # ===============================
    # Cargar LabelEncoder
    # ===============================
    le = joblib.load(args.labels)
    print(le.classes_)
    print(f"Clases: {len(le.classes_)}")
    num_classes = len(le.classes_)

    # ===============================
    # Cargar modelo entrenado
    # ===============================
    model = CNNClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(args.modelo, map_location=torch.device("cpu")))
    model.eval()

    # ===============================
    # Cargar embeddings
    # ===============================
    df = pd.read_csv(args.csv)
    embedding_cols = [col for col in df.columns if col.startswith("em_")]
    X = df[embedding_cols].values.astype(np.float32)

    # CNN espera entrada (1, 32, 96)
    X = X.reshape(-1, 1, 32, 96)

    # ===============================
    # Inferencia
    # ===============================
    pred_indices = []
    pred_clases = []

    with torch.no_grad():
        for emb in tqdm(X, desc="Inferencia"):
            input_tensor = torch.tensor(emb).unsqueeze(0)  # shape: (1, 1, 32, 96)
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            clase = le.inverse_transform([pred])[0]

            pred_indices.append(pred)
            pred_clases.append(clase)

    # ===============================
    # Guardar predicciones
    # ===============================
    df["target"] = pred_indices
    df["clase"] = pred_clases
    df.to_csv(args.output, index=False)
    print(f"✅ Archivo guardado: {args.output}")

if __name__ == "__main__":
    main()
