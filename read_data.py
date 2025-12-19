import os
import re

def load_drs_instances(drs_dir):
    """
    Lê arquivos DRS de um diretório e retorna uma lista de tuplas:
    (path_anchor, path_a, path_b) para cada idx encontrado.
    """
    files = os.listdir(drs_dir)
    pattern = re.compile(r"(\d+)_(anchor|a|b)_drs\.txt")

    # Agrupar arquivos por índice
    instances = {}
    for fname in files:
        match = pattern.match(fname)
        if match:
            idx, label = match.groups()
            if idx not in instances:
                instances[idx] = {}
            instances[idx][label] = os.path.join(drs_dir, fname)

    # Criar lista de trios ordenados por idx
    data = []
    for idx in sorted(instances.keys(), key=int):
        entry = instances[idx]
        if all(k in entry for k in ["anchor", "a", "b"]):
            data.append({
                'idx': int(idx),  # Keep index for debugging
                'anchor': entry["anchor"],
                'a': entry["a"],
                'b': entry["b"]
            })
        else:
            print(f"Warning: instance {idx} incomplete, ignored.")

    return data