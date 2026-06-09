import os
import sys
import time
import subprocess
import pandas as pd
import torch
from pathlib import Path

def run_smoke_test():
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent.parent
    outputs_dir = project_root / "outputs"
    checkpoints_dir = project_root / "checkpoints"
    
    # 1. Clean checkpoints and training history
    print("Cleaning stale checkpoint and history files...")
    history_csv = outputs_dir / "training_history.csv"
    if history_csv.exists():
        try:
            history_csv.unlink()
        except Exception as e:
            print(f"Warning: could not delete history CSV: {e}")
        
    for name in ["best_model.pth", "latest_model.pth"]:
        path = checkpoints_dir / name
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                print(f"Warning: could not delete checkpoint {name}: {e}")
            
    # 2. Setup training command
    cmd = [
        str(project_root / "venv" / "Scripts" / "python"),
        "src/training/train.py",
        "--train",
        "--epochs", "2",
        "--batch-size", "64",
        "--lr", "0.001"
    ]
    
    print("Launching smoke-test training session...")
    start_time = time.time()
    
    # Run process
    process = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8"
    )
    
    # Read output in real-time
    output_lines = []
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(line, end="")
        output_lines.append(line)
        
    process.wait()
    total_time = time.time() - start_time
    
    if process.returncode != 0:
        print(f"[ERROR] Training process exited with code: {process.returncode}")
        # Write failure report
        report_path = outputs_dir / "smoke_test_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Smoke Test Failed!\n")
            f.write(f"Return code: {process.returncode}\n")
            f.write("\nLast log lines:\n")
            f.write("".join(output_lines[-20:]))
        sys.exit(1)
        
    # 3. Read training history CSV
    if not history_csv.exists():
        print(f"[ERROR] History CSV not found at {history_csv}")
        sys.exit(1)
        
    df = pd.read_csv(history_csv)
    
    # Extract values
    epoch_1 = df[df["epoch"] == 1].iloc[0]
    epoch_2 = df[df["epoch"] == 2].iloc[0]
    
    e1_train = epoch_1["train_loss"]
    e1_val = epoch_1["validation_loss"]
    e2_train = epoch_2["train_loss"]
    e2_val = epoch_2["validation_loss"]
    
    # GPU detected
    gpu_detected = "N/A"
    if torch.cuda.is_available():
        gpu_detected = torch.cuda.get_device_name(0)
        
    # GPU memory usage
    gpu_mem = df["gpu_memory_usage"].max()
    
    # Checkpoint save status
    best_exists = (checkpoints_dir / "best_model.pth").exists()
    latest_exists = (checkpoints_dir / "latest_model.pth").exists()
    checkpoint_status = f"best_model.pth: {'Saved' if best_exists else 'Missing'}, latest_model.pth: {'Saved' if latest_exists else 'Missing'}"
    
    # Generate smoke_test_report.txt
    report_path = outputs_dir / "smoke_test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("             Flickr8K Training Smoke Test Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"1. GPU Detected               : {gpu_detected}\n")
        f.write(f"2. Peak GPU Memory Usage (MB) : {gpu_mem:.2f} MB\n")
        f.write(f"3. Epoch 1 Train Loss         : {e1_train:.6f}\n")
        f.write(f"4. Epoch 1 Validation Loss    : {e1_val:.6f}\n")
        f.write(f"5. Epoch 2 Train Loss         : {e2_train:.6f}\n")
        f.write(f"6. Epoch 2 Validation Loss    : {e2_val:.6f}\n")
        f.write(f"7. Total Training Time        : {total_time:.2f} s\n")
        f.write(f"8. Checkpoint Save Status     : {checkpoint_status}\n\n")
        
        f.write("Verdict: Smoke test training successfully completed.\n")
        
    print(f"Smoke test report saved to {report_path}")

if __name__ == "__main__":
    run_smoke_test()
