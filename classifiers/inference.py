import os
import argparse
import torch
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from model import CNNClassifier

# ===============================
# Script principal con soporte para pesos
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Inferencia con CNN y embeddings BirdNET")
    parser.add_argument("--csv", type=str, required=True, help="Ruta al archivo CSV con los embeddings")
    parser.add_argument("--modelo", type=str, required=True, help="Ruta al modelo completo (.pt) o solo los pesos")
    parser.add_argument("--labels", type=str, required=True, help="Ruta al archivo LabelEncoder (.pkl)")
    parser.add_argument("--output", type=str, default="predicciones.csv", help="Archivo de salida")
    parser.add_argument("--solo-pesos", action="store_true", help="Usar solo state_dict en lugar del modelo completo")
    args = parser.parse_args()

    # ===============================
    # Cargar LabelEncoder
    # ===============================
    print("🔤 Cargando codificador de etiquetas...")
    le = joblib.load(args.labels)
    print(f"Clases cargadas: {le.classes_}")
    num_classes = len(le.classes_)

    # ===============================
    # Cargar modelo
    # ===============================
    print("📦 Cargando modelo...")
    if args.solo_pesos:
        model = CNNClassifier(num_classes=num_classes)
        state_dict = torch.load(args.modelo, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    else:
        model = torch.load(args.modelo, map_location=torch.device("cpu"))

    model.eval()

    # ===============================
    # Cargar CSV de embeddings
    # ===============================
    print("📊 Cargando CSV...")
    df = pd.read_csv(args.csv)
    embedding_cols = [col for col in df.columns if col.startswith("emb_")]
    X = df[embedding_cols].values.astype(np.float32)

    # ===============================
    # Inferencia
    # ===============================
    print("🤖 Realizando inferencia...")
    X = X.reshape(-1, 1, 32, 96)
    pred_indices, pred_clases = [], []

    with torch.no_grad():
        for emb in tqdm(X, desc="Inferencia"):
            input_tensor = torch.tensor(emb).unsqueeze(0)  # (1, 1, 32, 96)
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

