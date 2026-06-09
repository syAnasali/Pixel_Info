import os
import time
import csv
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger("trainer")

class Trainer:
    """
    Production-grade Training Pipeline Manager for the Image Captioning Model.
    
    Manages loops, optimizer steps, learning rate scheduling, mixed-precision (AMP) context,
    gradient clipping, validation pass, early stopping, and checkpoint saving/resuming.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        checkpoint_dir: Path,
        history_path: Path,
        report_path: Path,
        epochs: int = 2,
        patience: int = 5,
        max_grad_norm: float = 5.0,
        pad_idx: int = 0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        
        # Paths setup
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history_path = Path(history_path)
        self.report_path = Path(report_path)
        
        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Training Parameters
        self.epochs = epochs
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.pad_idx = pad_idx
        
        # AMP scaler (GradScaler will be inactive if device is CPU)
        self.use_amp = (device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Diagnostics & Tracking
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = []
        
    def train_epoch(self, epoch: int) -> float:
        """Run a single training epoch with AMP and gradient clipping."""
        self.model.train()
        total_loss = 0.0
        
        # Setup progress bar
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        
        for batch in progress_bar:
            # Load batch to device
            img_feats = batch["image_features"].to(self.device)
            input_seqs = batch["input_sequence"].to(self.device)
            targets = batch["target_word"].to(self.device)
            
            # Squeeze targets to 1D for CrossEntropyLoss
            targets = targets.squeeze(-1)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass under AMP autocast context
            with autocast(enabled=self.use_amp):
                logits = self.model(img_feats, input_seqs)
                loss = self.criterion(logits, targets)
                
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Loss became {loss.item()} (NaN/Inf). Aborting training.")
                
            # Backward pass and optimizer step using GradScaler
            self.scaler.scale(loss).backward()
            
            # Unscale gradients for clipping before step
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                # Under AMP, gradients can be NaN/Inf if scaling factor overflows in FP16.
                # The GradScaler handles this by skipping the optimizer step and scaling down.
                # We only abort if not using AMP, or if the scale factor has already shrunk to <= 1.0.
                if not self.use_amp or (self.scaler.get_scale() <= 1.0):
                    reason = "NaN" if torch.isnan(grad_norm) else "Inf"
                    raise ValueError(f"Gradient norm is {reason} (scale: {self.scaler.get_scale() if self.use_amp else 'N/A'}). Aborting training.")
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
        
    def validate(self) -> float:
        """Evaluate the model on the validation dataset."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                img_feats = batch["image_features"].to(self.device)
                input_seqs = batch["input_sequence"].to(self.device)
                targets = batch["target_word"].to(self.device)
                targets = targets.squeeze(-1)
                
                with autocast(enabled=self.use_amp):
                    logits = self.model(img_feats, input_seqs)
                    loss = self.criterion(logits, targets)
                    
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
        
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save training state checkpoints containing architecture metadata."""
        filepath = self.checkpoint_dir / filename
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "metadata": {
                "vocab_size": self.model.vocab_size,
                "embed_size": self.model.embed_size,
                "hidden_size": self.model.hidden_size
            }
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
        
    def resume_training(self, checkpoint_path: Path) -> int:
        """Restore model, optimizer, scheduler, scaler and metadata from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
            
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Verify architecture metadata matching
        meta = checkpoint.get("metadata", {})
        if meta:
            assert meta["vocab_size"] == self.model.vocab_size, "Vocab size mismatch in checkpoint."
            assert meta["embed_size"] == self.model.embed_size, "Embedding size mismatch in checkpoint."
            assert meta["hidden_size"] == self.model.hidden_size, "Hidden dimension mismatch in checkpoint."
            
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.patience_counter = checkpoint.get("patience_counter", 0)
        
        logger.info(f"Successfully loaded state. Resuming starting from epoch {self.start_epoch}")
        return self.start_epoch
        
    def save_history_csv(self):
        """Append/Write training history metrics to CSV file."""
        fields = ["epoch", "train_loss", "validation_loss", "learning_rate", "epoch_time", "gpu_memory_usage"]
        
        write_header = not self.history_path.exists()
        
        with open(self.history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for row in self.history:
                writer.writerow(row)
                
        # Clear local buffer
        self.history.clear()
        
    def write_summary_report(self, total_duration: float, stop_reason: str):
        """Write training session execution summary report."""
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("             Image Captioning Model Training Summary Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Training Status              : COMPLETED\n")
            f.write(f"Completion Reason            : {stop_reason}\n")
            f.write(f"Device Configured            : {self.device}\n")
            f.write(f"Total Epochs Configured      : {self.epochs}\n")
            f.write(f"Total Duration (Seconds)     : {total_duration:.2f} s\n")
            f.write(f"Best Validation Loss Achieved: {self.best_val_loss:.6f}\n\n")
            
            f.write("--- Model Architecture Metadata ---\n")
            f.write(f"Vocabulary Size              : {self.model.vocab_size}\n")
            f.write(f"Embedding Dimensions         : {self.model.embed_size}\n")
            f.write(f"LSTM Hidden Dimensions       : {self.model.hidden_size}\n\n")
            
            f.write("Checkpoints Saved:\n")
            f.write(f"  - Best Model               : {self.checkpoint_dir / 'best_model.pth'}\n")
            f.write(f"  - Latest Model             : {self.checkpoint_dir / 'latest_model.pth'}\n\n")
            
            f.write("Run history file saved to: outputs/training_history.csv\n")
            
        logger.info(f"Summary report successfully saved to: {self.report_path}")

    def train(self):
        """Run the complete multi-epoch training pipeline."""
        start_time = time.time()
        stop_reason = "Completed all configured epochs."
        
        logger.info(f"Starting model training pipeline on device: {self.device}")
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.time()
            
            # 1. Training step
            train_loss = self.train_epoch(epoch)
            
            # 2. Validation step
            val_loss = self.validate()
            
            epoch_duration = time.time() - epoch_start
            
            # 3. Scheduler step (based on validation loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    logger.info(f"Learning Rate Scheduler updated learning rate: {current_lr} -> {new_lr}")
                    current_lr = new_lr
                    
            # Log current learning rate
            logger.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
            
            # 4. Check GPU memory
            gpu_mem = 0.0
            if self.device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                
            # 5. Track history row
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "learning_rate": current_lr,
                "epoch_time": epoch_duration,
                "gpu_memory_usage": gpu_mem
            }
            self.history.append(row)
            self.save_history_csv()
            
            # Automatically plot training loss curves after every epoch
            try:
                from src.utils.plot_history import plot_training_history
                plot_training_history(self.history_path, self.history_path.parent / "training_history.png")
            except Exception as e:
                logger.error(f"Failed to generate training history plot automatically: {e}")
            
            # 6. Save latest checkpoint after every epoch
            self.save_checkpoint("latest_model.pth", epoch, val_loss)
            
            # 7. Check if validation loss improved (Best Model Tracking)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                logger.info(f"Validation loss improved. Saving best model checkpoint.")
                self.save_checkpoint("best_model.pth", epoch, val_loss)
            else:
                self.patience_counter += 1
                logger.info(f"Validation loss did not improve. Early stopping counter: {self.patience_counter}/{self.patience}")
                
            # 8. Early Stopping Check
            if self.patience_counter >= self.patience:
                stop_reason = f"Early stopping triggered at epoch {epoch} (No improvement for {self.patience} epochs)."
                logger.warning(stop_reason)
                break
                
        total_duration = time.time() - start_time
        self.write_summary_report(total_duration, stop_reason)
        
        # Plot training loss history automatically
        try:
            from src.utils.plot_history import plot_training_history
            plot_training_history(self.history_path, self.history_path.parent / "training_history.png")
        except Exception as e:
            logger.error(f"Failed to generate training history plot automatically: {e}")
            
        logger.info("Training pipeline execution completed.")
