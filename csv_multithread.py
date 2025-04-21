import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time

# -------------------- CONFIGURACIÓN --------------------
EMBEDDINGS_ROOT = "embeddings"
OUTPUT_CSV = "embeddings_csv/embeddings_3chunks_MT.csv"

# Control de hilos
MAX_THREADS = 4  # Cambia esto entre 1 y 5
MAX_THREADS = min(MAX_THREADS, 5, os.cpu_count())
# -------------------------------------------------------

def load_embedding(path):
    """Lee el embedding como texto, conservando el formato decimal exacto."""
    with open(path, "r") as f:
        line = f.readline().strip()
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"Formato inválido en archivo: {path}")
        emb_str = parts[2]
        return emb_str.split(",")  # devuelve lista de strings, no floats

def parse_filename(path):
    """Extrae info desde el path completo."""
    file = os.path.basename(path)
    label = os.path.basename(os.path.dirname(path))
    audio_id = file.split(".")[0].rsplit("_", 1)[0]
    chunk_index = int(file.split("_")[1].split(".")[0])
    return label, audio_id, chunk_index

# Recolectar todos los paths
all_txt_files = []
for label_folder in os.listdir(EMBEDDINGS_ROOT):
    folder_path = os.path.join(EMBEDDINGS_ROOT, label_folder)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        if file.endswith(".birdnet.embeddings.txt"):
            all_txt_files.append(os.path.join(folder_path, file))

# Diccionario de: { (label, audio_id): [(chunk_index, embedding), ...] }
audio_chunks = defaultdict(list)

print(f"Cargando archivos en paralelo con {MAX_THREADS} hilos...")
start_time = time.time()

with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    future_to_path = {executor.submit(load_embedding, path): path for path in all_txt_files}
    for future in as_completed(future_to_path):
        path = future_to_path[future]
        try:
            emb = future.result()
            label, audio_id, chunk_idx = parse_filename(path)
            audio_chunks[(label, audio_id)].append((chunk_idx, emb))
        except Exception as e:
            print(f"Error en {path}: {e}")

print("Procesando agrupaciones...")
rows = []

for (label, audio_id), chunks in audio_chunks.items():
    chunks.sort(key=lambda x: x[0])
    embeddings = [e[1] for e in chunks]  # e[1] es una lista de strings

    for i in range(len(embeddings)):
        group = embeddings[i:i+3]
        if len(group) < 3:
            group += [group[-1]] * (3 - len(group))  # padding

        if len(group) == 3:
            concatenated = sum(group, [])  # concatena listas de strings
            row_id = f"{audio_id}_em{i}"
            row = [row_id, label] + concatenated  # todo como strings
            rows.append(row)

# Guardar
columns = ["ID", "label"] + [f"emb_{i}" for i in range(3072)]
df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)  # NO usar float_format

elapsed = time.time() - start_time
print(f"CSV guardado en {OUTPUT_CSV} con {len(rows)} filas.")
print(f"Tiempo total: {elapsed:.2f} segundos.")
