import os
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time

def load_embedding(path):
    with open(path, "r") as f:
        line = f.readline().strip()
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"Formato inválido en archivo: {path}")
        emb_str = parts[2]
        return emb_str.split(",")  # lista de strings

def parse_filename(path):
    file = os.path.basename(path)
    label = os.path.basename(os.path.dirname(path))
    audio_id = file.split(".")[0].rsplit("_", 1)[0]
    chunk_index = int(file.split("_")[1].split(".")[0])
    return label, audio_id, chunk_index

def generar_csv_embeddings(input_dir, output_csv, chunk_size=3, num_threads=4):
    max_threads = min(num_threads, os.cpu_count())
    all_txt_files = []

    for label_folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".birdnet.embeddings.txt"):
                all_txt_files.append(os.path.join(folder_path, file))

    audio_chunks = defaultdict(list)

    print(f"Cargando archivos en paralelo con {max_threads} hilos...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
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
        embeddings = [e[1] for e in chunks]

        for i in range(len(embeddings)):
            group = embeddings[i:i+chunk_size]
            if len(group) < chunk_size:
                group += [group[-1]] * (chunk_size - len(group))
            if len(group) == chunk_size:
                concatenated = sum(group, [])
                row = [audio_id, str(i), label] + concatenated
                rows.append(row)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    num_features = len(rows[0]) - 3 if rows else chunk_size * 1024
    columns = ["ID", "group", "label"] + [f"emb_{i}" for i in range(num_features)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    print(f"CSV guardado en {output_csv} con {len(rows)} filas.")
    print(f"Tiempo total: {elapsed:.2f} segundos.")

# ======================= EJECUCIÓN ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de CSV desde embeddings BirdNET")
    parser.add_argument("--input", type=str, default="embeddings", help="Directorio raíz de entrada con carpetas por clase")
    parser.add_argument("--output", type=str, default="embeddings_csv/embeddings_MT_overlap.csv", help="Ruta del archivo CSV de salida")
    parser.add_argument("--chunks", type=int, default=3, help="Número de embeddings a concatenar por fila")
    parser.add_argument("--threads", type=int, default=4, help="Número de hilos para procesamiento paralelo")
    args = parser.parse_args()

    generar_csv_embeddings(args.input, args.output, chunk_size=args.chunks, num_threads=args.threads)

#ejemplo
# cd ~BN_R1.5.1
# python embed2csv/embed_MT_P_OV.py --input embeddings --output embeddings_csv/embeddings_MT_overlap.csv --chunks 3 --threads 4