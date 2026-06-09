import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import FLICKR_DIR, OUTPUT_DIR

def run_raw_audit():
    captions_file = FLICKR_DIR / "captions.txt"
    if not captions_file.exists():
        print(f"[ERROR] Raw captions file not found at {captions_file}")
        sys.exit(1)
        
    abs_path = captions_file.resolve()
    file_size_bytes = abs_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Read all lines
    with open(abs_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    total_lines = len(lines)
    
    # Extract first 10 and last 10 lines
    first_10 = lines[:10]
    last_10 = lines[-10:] if total_lines >= 10 else lines
    
    # Parse unique images and caption rows
    unique_images = set()
    caption_rows_count = 0
    
    # We skip the header line (lines[0])
    for line in lines[1:]:
        clean_line = line.strip()
        if not clean_line:
            continue
        parts = clean_line.split(",", 1)
        if len(parts) >= 1:
            img = parts[0].strip()
            if img:
                unique_images.add(img)
                caption_rows_count += 1
                
    unique_image_count = len(unique_images)
    
    # Generate report path
    report_path = OUTPUT_DIR / "captions_file_audit.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"1. Absolute file path: {abs_path}\n")
        f.write(f"2. File size in MB: {file_size_mb:.6f} MB\n")
        f.write(f"3. Total line count: {total_lines}\n")
        
        f.write("4. First 10 lines exactly as stored:\n")
        for line in first_10:
            f.write(line)
        if first_10 and not first_10[-1].endswith("\n"):
            f.write("\n")
            
        f.write("\n5. Last 10 lines exactly as stored:\n")
        for line in last_10:
            f.write(line)
        if last_10 and not last_10[-1].endswith("\n"):
            f.write("\n")
            
        f.write(f"\n6. Number of unique image names detected: {unique_image_count}\n")
        f.write(f"7. Number of caption rows detected: {caption_rows_count}\n")
        
    print(f"Audit completed successfully. Report written to {report_path}")

if __name__ == "__main__":
    run_raw_audit()

