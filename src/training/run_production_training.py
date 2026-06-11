import os
import sys
import time
import subprocess
import pandas as pd
import torch
from pathlib import Path

def run_production_training():
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent.parent
    outputs_dir = project_root / "outputs"
    checkpoints_dir = project_root / "checkpoints"
    history_csv = outputs_dir / "training_history.csv"
    
    # 1. Determine if latest checkpoint exists for resuming
    latest_checkpoint = checkpoints_dir / "latest_model.pth"
    resume_arg = []
    if latest_checkpoint.exists():
        print(f"Latest checkpoint found at {latest_checkpoint}. Resuming training...")
        resume_arg = ["--resume", str(latest_checkpoint)]
    else:
        print("No existing checkpoint found. Starting training from scratch...")
        
    # 2. Setup training command for 20 epochs
    cmd = [
        str(project_root / "venv" / "Scripts" / "python"),
        "src/training/train.py",
        "--train",
        "--epochs", "20",
        "--batch-size", "64",
        "--lr", "0.001"
    ] + resume_arg
    
    print(f"Launching production training session (Max Epochs: 20)...")
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
        report_path = outputs_dir / "final_training_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Production Training Failed!\n")
            f.write(f"Return code: {process.returncode}\n")
            f.write("\nLast log lines:\n")
            f.write("".join(output_lines[-20:]))
        sys.exit(1)
        
    # 3. Read training history CSV and compile stats
    if not history_csv.exists():
        print(f"[ERROR] History CSV not found at {history_csv}")
        sys.exit(1)
        
    df = pd.read_csv(history_csv)
    
    # Extract values
    total_epochs = int(df["epoch"].max())
    best_train_loss = df["train_loss"].min()
    best_val_loss = df["validation_loss"].min()
    final_lr = df.iloc[-1]["learning_rate"]
    peak_gpu_mem = df["gpu_memory_usage"].max()
    
    # GPU detected
    gpu_detected = "N/A"
    if torch.cuda.is_available():
        gpu_detected = torch.cuda.get_device_name(0)
        
    # Checkpoint paths
    best_model_path = checkpoints_dir / "best_model.pth"
    latest_model_path = checkpoints_dir / "latest_model.pth"
    
    # Write final_training_report.txt
    report_path = outputs_dir / "final_training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("             Flickr8K Production Training Final Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"1. GPU Detected               : {gpu_detected}\n")
        f.write(f"2. Peak GPU Memory Usage (MB) : {peak_gpu_mem:.2f} MB\n")
        f.write(f"3. Total Epochs Completed     : {total_epochs}\n")
        f.write(f"4. Best Training Loss         : {best_train_loss:.6f}\n")
        f.write(f"5. Best Validation Loss       : {best_val_loss:.6f}\n")
        f.write(f"6. Final Learning Rate        : {final_lr:.6f}\n")
        f.write(f"7. Total Training Time        : {total_time:.2f} s ({total_time/60.0:.2f} minutes)\n")
        f.write(f"8. Best Checkpoint Path       : {best_model_path.resolve()}\n")
        f.write(f"9. Latest Checkpoint Path     : {latest_model_path.resolve()}\n\n")
        
        f.write("Verdict: Production training completed successfully.\n")
        
    print(f"Production training report saved to {report_path}")

if __name__ == "__main__":
    run_production_training()
