#!/usr/bin/env python3
"""parse synoptic CSV and apply scansion model to all verse lines
while keeping track of structure and metadata"""

import sys
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# src directory to path for imports
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from scan.apply import load_scanner, predict_batch
from scan.utils import preprocess_line, jsonify


def parse_synoptic_csv(input_file, model_path, vectorizer_path, output_file):
    """parse synoptic CSV file and apply scansion model.
    
    args:
        input_file: path to synoptic.csv
        model_path: path to scanner.keras
        vectorizer_path: path to vectorizer.json
        output_file: path to output JSON file
    """

    print(f"loading CSV file... - {input_file}")
    df = pd.read_csv(input_file)
    
    verse_line_ids = df.iloc[:, 0].tolist()
    manuscript_sigla = [col for col in df.columns[1:] if pd.notna(col) and str(col).strip()]
    
    print(f"found {len(verse_line_ids)} verse lines")
    print(f"found {len(manuscript_sigla)} manuscript sigla")
    
    print("loading scansion model...")
    model, vectorizer = load_scanner(model_path, vectorizer_path)
    
    results = []
    all_texts = []
    
    print("extracting verse lines...")
    for idx, verse_id in enumerate(tqdm(verse_line_ids, desc="processing rows")):
        if pd.isna(verse_id):
            continue
        
        verse_data = {
            "verse_line_id": str(verse_id),
            "manuscripts": {}
        }
        
        for siglum in manuscript_sigla:
            cell_value = df.iloc[idx, df.columns.get_loc(siglum)]
            
            if pd.isna(cell_value) or str(cell_value).strip() == "None":
                verse_data["manuscripts"][siglum] = {
                    "present": False,
                    "fragmented": False,
                    "text": None,
                    "scansion": None
                }
            else:
                text = str(cell_value).strip()
                is_fragmented = "..." in text
                clean_text = text.replace("...", "").strip()
                preprocessed = preprocess_line(clean_text)
                
                verse_data["manuscripts"][siglum] = {
                    "present": True,
                    "fragmented": is_fragmented,
                    "text": clean_text,
                    "preprocessed": preprocessed,
                    "scansion": None
                }
                
                if preprocessed:
                    all_texts.append(preprocessed)
        
        results.append(verse_data)
    
    total_lines = len(all_texts)
    print(f"found {total_lines} verse lines to scan")
    
    if not all_texts:
        print("no verse lines to process")
        return
    
    print("applying scansion model...")
    batch_size = 32
    all_classes = []
    
    pbar = tqdm(total=total_lines, desc="scanning verse lines", unit="line")
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i+batch_size]
        batch_classes = predict_batch(model, vectorizer, batch, batch_size=batch_size)
        all_classes.extend(batch_classes)
        pbar.update(len(batch))
    pbar.close()
    
    print("formatting results...")
    text_idx = 0
    for verse_data in results:
        for siglum in manuscript_sigla:
            ms_data = verse_data["manuscripts"][siglum]
            if ms_data["present"] and ms_data.get("preprocessed"):
                classes = all_classes[text_idx]
                scansion_json = jsonify(ms_data["preprocessed"], classes)
                ms_data["scansion"] = json.loads(scansion_json)
                del ms_data["preprocessed"]
                text_idx += 1
    
    print(f"Writing results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Done")


def main():
    if len(sys.argv) != 2:
        print("usage: python parse_and_scan.py <input_csv>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.is_absolute():
        input_file = Path.cwd() / input_file
    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)
    
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "scansion_model" / "scanner.keras"
    vectorizer_path = project_root / "scansion_model" / "vectorizer.json"
    output_file = project_root / "data" / "output" / "scansions.json"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    parse_synoptic_csv(input_file, model_path, vectorizer_path, output_file)


if __name__ == "__main__":
    main()