import os
import numpy as np
import pandas as pd

EMBEDDINGS_ROOT = "embeddings"
OUTPUT_CSV = "embeddings_csv/embeddings_3chunks_ST.csv"

def load_embedding(path):
    with open(path, "r") as f:
        line = f.readline().strip()
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"Formato inválido en archivo: {path}")
        emb_str = parts[2]  # la tercera columna
        emb = np.array([float(x) for x in emb_str.split(",")], dtype=np.float32)
        return emb

def get_base_audio_id(filename):
    return filename.split(".")[0].rsplit("_", 1)[0]  # iNat146544

def get_chunk_index(filename):
    return int(filename.split("_")[1].split(".")[0])  # 0, 1, 2...

rows = []

# Recorrer subcarpetas que representan especies
for label_folder in sorted(os.listdir(EMBEDDINGS_ROOT)):
    label_path = os.path.join(EMBEDDINGS_ROOT, label_folder)
    if not os.path.isdir(label_path):
        continue

    embeddings_by_audio = {}

    # Recorrer archivos dentro de la carpeta de la especie
    for file in sorted(os.listdir(label_path)):
        if file.endswith(".birdnet.embeddings.txt"):
            try:
                audio_id = get_base_audio_id(file)
                chunk_idx = get_chunk_index(file)
                emb = load_embedding(os.path.join(label_path, file))

                if audio_id not in embeddings_by_audio:
                    embeddings_by_audio[audio_id] = []
                embeddings_by_audio[audio_id].append((chunk_idx, emb))
            except Exception as e:
                print(f"Error al leer {file}: {e}")

    # Procesar cada audio
    for audio_id, chunk_list in embeddings_by_audio.items():
        chunk_list.sort(key=lambda x: x[0])  # ordenar por chunk index
        embeddings = [e[1] for e in chunk_list]

        for i in range(len(embeddings)):
            group = embeddings[i:i+3]
            if len(group) < 3:
                group += [group[-1]] * (3 - len(group))  # padding con el último
            if len(group) == 3:
                concatenated = np.concatenate(group)
                row_id = f"{audio_id}_em{i}"
                row = [row_id, label_folder] + concatenated.tolist()
                rows.append(row)

# Crear CSV
columns = ["ID", "label"] + [f"emb_{i}" for i in range(3072)]
df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False, float_format="%.8f")
print(f"CSV guardado como {OUTPUT_CSV}")

#ejemplo

# cd ~BN_R1.5.1
#python embed2csv/csv_MT_overlap.py