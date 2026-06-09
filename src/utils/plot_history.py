import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def plot_training_history(csv_path: Path, output_png_path: Path):
    """
    Load training metrics from CSV and save a loss curve plot.
    """
    csv_path = Path(csv_path)
    output_png_path = Path(output_png_path)
    
    if not csv_path.exists():
        print(f"[WARNING] CSV history file not found at {csv_path}. Cannot generate plot.")
        return
        
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("[WARNING] CSV history is empty. Cannot generate plot.")
            return
            
        epochs = df["epoch"].values
        train_loss = df["train_loss"].values
        val_loss = df["validation_loss"].values
        
        # Configure plotting styles
        plt.figure(figsize=(10, 6))
        
        # Draw curves
        plt.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", marker="o", linewidth=2.5, markersize=6)
        plt.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", marker="s", linewidth=2.5, markersize=6)
        
        # Labels and Title
        plt.title("Image Captioning Model - Training & Validation Loss", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Epoch", fontsize=12, labelpad=10)
        plt.ylabel("Loss (Cross Entropy)", fontsize=12, labelpad=10)
        
        # Formatting grid and ticks
        plt.grid(True, linestyle="--", alpha=0.6)
        
        # Ensure x-ticks align with integer epochs
        if len(epochs) <= 20:
            plt.xticks(epochs)
            
        plt.legend(fontsize=11, loc="upper right")
        plt.tight_layout()
        
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png_path, dpi=300)
        plt.close()
        print(f"Successfully generated loss curve plot: {output_png_path}")
    except Exception as e:
        print(f"[ERROR] Failed to generate loss curve plot: {e}")

if __name__ == "__main__":
    csv_input = PROJECT_ROOT / "outputs" / "training_history.csv"
    png_output = PROJECT_ROOT / "outputs" / "training_history.png"
    plot_training_history(csv_input, png_output)
