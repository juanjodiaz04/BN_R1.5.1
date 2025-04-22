import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

# ===============================
# Arquitectura del modelo
# ===============================
class MiRedNeuronal(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ===============================
# Función principal
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Inferencia con modelo .pt y embeddings BirdNET")
    parser.add_argument("--csv", type=str, required=True, help="Ruta al archivo CSV con los embeddings")
    parser.add_argument("--modelo", type=str, required=True, help="Ruta al archivo del modelo (.pt)")
    parser.add_argument("--labels", type=str, required=True, help="Ruta al archivo LabelEncoder (.pkl)")
    parser.add_argument("--output", type=str, default="predicciones.csv", help="Archivo de salida")
    args = parser.parse_args()

    # ===============================
    # Cargar LabelEncoder y definir modelo
    # ===============================
    le = joblib.load(args.labels)
    num_classes = len(le.classes_)

    model = MiRedNeuronal(num_classes=num_classes)
    model.load_state_dict(torch.load(args.modelo, map_location=torch.device('cpu')))
    model.eval()

    # ===============================
    # Cargar embeddings del CSV
    # ===============================
    df = pd.read_csv(args.csv)
    embedding_cols = [col for col in df.columns if col.startswith("em_")]
    X = df[embedding_cols].values.astype(np.float32)

    # ===============================
    # Inferencia
    # ===============================
    pred_indices = []
    pred_clases = []

    with torch.no_grad():
        for emb in tqdm(X, desc="Inferencia"):
            input_tensor = torch.tensor(emb).unsqueeze(0)  # shape: (1, 3072)
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            clase = le.inverse_transform([pred])[0]

            pred_indices.append(pred)
            pred_clases.append(clase)

    # ===============================
    # Guardar resultados
    # ===============================
    df["target"] = pred_indices
    df["clase"] = pred_clases
    df.to_csv(args.output, index=False)
    print(f"✅ Archivo guardado: {args.output}")

if __name__ == "__main__":
    main()

# ejemplo

# python inference.py --csv embeddings_csv/embeddings_MT_overlap.csv--modelo outputs/modelo_22_1530.pt --labels outputs/label_encoder.pkl --output outputs/ample_submission.csv