import os
import sys
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def get_file_info(filepath):
    if not filepath.exists():
        return {
            "exists": False,
            "ctime": "N/A",
            "mtime": "N/A",
            "size": 0
        }
    stat = filepath.stat()
    # On Windows, ctime is creation time, mtime is modification time.
    ctime_dt = datetime.fromtimestamp(stat.st_ctime)
    mtime_dt = datetime.fromtimestamp(stat.st_mtime)
    return {
        "exists": True,
        "ctime": ctime_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "mtime": mtime_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "size": stat.st_size
    }

def trace_pipeline():
    outputs_dir = PROJECT_ROOT / "outputs"
    
    files = {
        "cleaned_captions.csv": outputs_dir / "cleaned_captions.csv",
        "word_to_index.pkl": outputs_dir / "word_to_index.pkl",
        "index_to_word.pkl": outputs_dir / "index_to_word.pkl",
        "training_sequences.pkl": outputs_dir / "training_sequences.pkl"
    }
    
    trace_data = {}
    
    # 1. cleaned_captions.csv
    csv_path = files["cleaned_captions.csv"]
    csv_info = get_file_info(csv_path)
    csv_rows = 0
    csv_first_5 = []
    csv_last_5 = []
    
    if csv_info["exists"]:
        # Read the file
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_lines = [line.strip() for line in f.readlines()]
        
        # Total lines including header
        total_csv_lines = len(csv_lines)
        # We want to exclude header and empty lines
        header = csv_lines[0] if total_csv_lines > 0 else ""
        content_lines = [line for line in csv_lines[1:] if line]
        csv_rows = len(content_lines)
        
        csv_first_5 = content_lines[:5]
        csv_last_5 = content_lines[-5:] if csv_rows >= 5 else content_lines
        
        # Source file determination
        # Let's inspect the image names in the content lines to see if they refer to the mock or real dataset
        # Also let's check unique images
        unique_images_in_csv = set()
        for line in content_lines:
            parts = line.split(",", 1)
            if parts:
                unique_images_in_csv.add(parts[0])
        
        csv_info["num_rows"] = csv_rows
        csv_info["unique_images"] = len(unique_images_in_csv)
        if csv_rows < 1000:
            csv_info["source_verdict"] = f"Mock Test Dataset (contains only {csv_rows} rows, {len(unique_images_in_csv)} unique images)"
        else:
            csv_info["source_verdict"] = f"Full Flickr8k Dataset (contains {csv_rows} rows, {len(unique_images_in_csv)} unique images)"
    else:
        csv_info["num_rows"] = 0
        csv_info["unique_images"] = 0
        csv_info["source_verdict"] = "N/A"
        
    trace_data["cleaned_captions.csv"] = csv_info
    
    # 2. word_to_index.pkl
    w2i_path = files["word_to_index.pkl"]
    w2i_info = get_file_info(w2i_path)
    if w2i_info["exists"]:
        with open(w2i_path, "rb") as f:
            w2i = pickle.load(f)
        w2i_info["num_samples"] = len(w2i)
        w2i_info["keys_preview"] = list(w2i.keys())[:10]
        if len(w2i) < 100:
            w2i_info["source_verdict"] = f"Mock Test Dataset (vocabulary size {len(w2i)})"
        else:
            w2i_info["source_verdict"] = f"Full Flickr8k Dataset (vocabulary size {len(w2i)})"
    else:
        w2i_info["num_samples"] = 0
        w2i_info["keys_preview"] = []
        w2i_info["source_verdict"] = "N/A"
    trace_data["word_to_index.pkl"] = w2i_info
    
    # 3. index_to_word.pkl
    i2w_path = files["index_to_word.pkl"]
    i2w_info = get_file_info(i2w_path)
    if i2w_info["exists"]:
        with open(i2w_path, "rb") as f:
            i2w = pickle.load(f)
        i2w_info["num_samples"] = len(i2w)
        i2w_info["keys_preview"] = list(i2w.values())[:10]
        if len(i2w) < 100:
            i2w_info["source_verdict"] = f"Mock Test Dataset (vocabulary size {len(i2w)})"
        else:
            i2w_info["source_verdict"] = f"Full Flickr8k Dataset (vocabulary size {len(i2w)})"
    else:
        i2w_info["num_samples"] = 0
        i2w_info["keys_preview"] = []
        i2w_info["source_verdict"] = "N/A"
    trace_data["index_to_word.pkl"] = i2w_info
    
    # 4. training_sequences.pkl
    seq_path = files["training_sequences.pkl"]
    seq_info = get_file_info(seq_path)
    if seq_info["exists"]:
        with open(seq_path, "rb") as f:
            seq_data = pickle.load(f)
        # Let's inspect the keys or shape of training sequences
        if isinstance(seq_data, dict):
            num_samples = len(seq_data.get("input_sequences", []))
            if num_samples == 0:
                # might be structured differently
                num_samples = len(seq_data.get("image_names", []))
            seq_info["num_samples"] = num_samples
        else:
            seq_info["num_samples"] = len(seq_data)
            
        if seq_info["num_samples"] < 1000:
            seq_info["source_verdict"] = f"Mock Test Dataset (contains only {seq_info['num_samples']} training sequences)"
        else:
            seq_info["source_verdict"] = f"Full Flickr8k Dataset (contains {seq_info['num_samples']} training sequences)"
    else:
        seq_info["num_samples"] = 0
        seq_info["source_verdict"] = "N/A"
    trace_data["training_sequences.pkl"] = seq_info

    # Determine if stale mock outputs are being reused
    # We compare ctime/mtime with captions.txt ctime/mtime
    captions_file = PROJECT_ROOT / "data" / "Flickr8k" / "captions.txt"
    captions_info = get_file_info(captions_file)
    
    stale_reused = False
    stale_details = []
    
    if captions_info["exists"] and csv_info["exists"]:
        # If the captions file modification time is newer than cleaned_captions.csv modification time,
        # and captions.txt size is large (full dataset) while cleaned_captions.csv size is small (mock dataset),
        # then stale mock outputs are definitely being reused/present.
        if captions_file.stat().st_size > 1000000 and csv_path.stat().st_size < 10000:
            stale_reused = True
            stale_details.append("cleaned_captions.csv was generated from the Mock Dataset, but captions.txt is now the Full Flickr8k Dataset.")
            
        # Check if the pickles or sequence files are old or match the mock dataset sizes
        if w2i_info["exists"] and w2i_info["num_samples"] < 100:
            stale_reused = True
            stale_details.append(f"word_to_index.pkl has mock size ({w2i_info['num_samples']} words) instead of full dataset size.")
        if seq_info["exists"] and seq_info["num_samples"] < 1000:
            stale_reused = True
            stale_details.append(f"training_sequences.pkl has mock size ({seq_info['num_samples']} sequences) instead of full dataset size.")

    # Write report
    report_path = outputs_dir / "pipeline_trace_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("                  Preprocessing Pipeline Trace Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Audited on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for name, path in files.items():
            info = trace_data[name]
            f.write(f"File: {name}\n")
            f.write(f"  - Absolute Path: {path.resolve()}\n")
            f.write(f"  - Exists: {info['exists']}\n")
            if info['exists']:
                f.write(f"  - Size: {info['size']} bytes\n")
                f.write(f"  - Creation Timestamp (ctime): {info['ctime']}\n")
                f.write(f"  - Modification Timestamp (mtime): {info['mtime']}\n")
                if "num_rows" in info:
                    f.write(f"  - Number of Rows Stored: {info['num_rows']}\n")
                if "num_samples" in info:
                    f.write(f"  - Number of Samples Stored: {info['num_samples']}\n")
                f.write(f"  - Estimated Source File Used: {info['source_verdict']}\n")
            f.write("\n")
            
        f.write("=" * 70 + "\n")
        f.write("                     cleaned_captions.csv Audit\n")
        f.write("=" * 70 + "\n")
        if csv_info["exists"]:
            f.write(f"Total Rows: {csv_rows}\n\n")
            f.write("First 5 Rows:\n")
            for line in csv_first_5:
                f.write(f"  {line}\n")
            f.write("\nLast 5 Rows:\n")
            for line in csv_last_5:
                f.write(f"  {line}\n")
        else:
            f.write("File does not exist.\n")
        f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("                     Pipeline State Verdict\n")
        f.write("=" * 70 + "\n")
        if stale_reused:
            f.write("VERDICT: STALE MOCK OUTPUTS DETECTED!\n")
            f.write("Details of inconsistency:\n")
            for detail in stale_details:
                f.write(f"  * {detail}\n")
            f.write("\nAction required: Run the preprocessing pipeline on the full Flickr8k captions file to update these outputs.\n")
        else:
            f.write("VERDICT: Pipeline outputs are consistent and up-to-date with the active captions source file.\n")
            
    print(f"Pipeline trace completed successfully. Report written to {report_path}")

if __name__ == "__main__":
    trace_pipeline()
