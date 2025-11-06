import os
import json
import shutil
from unidecode import unidecode
import text2story as t2s
from text2story.brat2viz.brat2drs import brat2drs

def narrative2brat(text):
    doc = t2s.Narrative('en', text, '2020-05-30')
    doc.extract_participants()
    doc.extract_events()
    doc.extract_times()
    doc.extract_semantic_role_links()
    doc.extract_objectal_links()
    return doc.ISO_annotation()

def process_jsonl_to_drs(jsonl_path, output_brat_dir, output_drs_dir, start_idx=0, end_idx=None):
    t2s.load("en")

    os.makedirs(output_brat_dir, exist_ok=True)
    os.makedirs(output_drs_dir, exist_ok=True)

    with open(jsonl_path, "r") as fd:
        for idx, line in enumerate(fd):
            if idx < start_idx:
                continue
            if end_idx is not None and idx >= end_idx:
                break

            data = json.loads(line)
            for label in ["anchor", "a", "b"]:
                print(f"Processing {label} of example {idx}")
                text = unidecode(data[f"text_{label}" if label != "anchor" else "anchor_text"])
                brat_txt = narrative2brat(text)
                brat_file = os.path.join(output_brat_dir, f"{idx}_{label}.ann")
                with open(brat_file, "w") as f:
                    f.write(brat_txt)

                # Converte para DRS (salvo no mesmo diretório do .ann)
                brat2drs.process(brat_file)

                # Move o arquivo DRS gerado para o diretório de saída
                drs_file = brat_file.replace(".ann", "_drs.txt")
                if os.path.exists(drs_file):
                    shutil.move(drs_file, os.path.join(output_drs_dir, os.path.basename(drs_file)))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-processing of semeval dataset to DRS")
    parser.add_argument("--jsonl", type=str, required=True, help="path to the JSONL dataset")
    parser.add_argument("--ann_dir", type=str, required=True, help="output directory for the .ann files")
    parser.add_argument("--drs_dir", type=str, required=True, help="output directory for the .drs files")
    parser.add_argument("--start", type=int, default=0, help="start instance (optional)")
    parser.add_argument("--end", type=int, default=None, help="end instance (optional)")

    args = parser.parse_args()

    process_jsonl_to_drs(
        jsonl_path=args.jsonl,
        output_brat_dir=args.ann_dir,
        output_drs_dir=args.drs_dir,
        start_idx=args.start,
        end_idx=args.end
    )