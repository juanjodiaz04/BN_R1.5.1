import os
import argparse
import pandas as pd

def extraer_embeddings(input_root, output_folder):
    """
    Convert a set of .txt files containing embeddings into a single CSV file.
    """
    output_path = os.path.join(output_folder, "all_embeddings.csv")
    os.makedirs(output_folder, exist_ok=True)

    all_rows = []

    for class_folder in os.listdir(input_root):
        class_path = os.path.join(input_root, class_folder)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if filename.endswith(".txt"):
                input_path = os.path.join(class_path, filename)
                id_name = filename.replace(".birdnet.embeddings.txt", "")

                with open(input_path, "r") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) != 3:
                            continue

                        embedding = list(map(float, parts[2].split(",")))
                        row = [id_name, class_folder] + embedding
                        all_rows.append(row)

    if not all_rows:
        print("No embeddings found.")
        return

    embedding_dim = len(all_rows[0]) - 2
    columns = ["id", "label"] + [f"emb_{i}" for i in range(embedding_dim)]

    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(output_path, index=False)

    print(f"\n Guardados {len(df)} registros en '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte embeddings en archivos .txt a un CSV consolidado.")
    parser.add_argument("--input", type=str, default="embeddings", help="Directorio de entrada con carpetas por clase")
    parser.add_argument("--output", type=str, default="embeddings_csv", help="Carpeta de salida para el archivo CSV")
    args = parser.parse_args()

    extraer_embeddings(args.input, args.output)


    # Example usage:
    # python csv_embed.py --input embeddings --output embeddings_csv
    # This will read all .txt files in the 'embeddings' directory and save the consolidated CSV in 'embeddings_csv/all_embeddings.csv'