import os
import sys
import argparse
import pickle
import logging
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import DEVICE, OUTPUT_DIR, BATCH_SIZE, LEARNING_RATE
from src.model.image_caption_model import ImageCaptionModel
from src.training.create_dataloaders import prepare_dataloaders
from src.training.trainer import Trainer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

def run_pipeline_validation(model: nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> bool:
    """
    Run one forward pass, one backward pass, and one optimizer step validation tests.
    Writes results to outputs/training_pipeline_validation.txt.
    """
    logger.info("Executing training pipeline verification checks...")
    
    forward_passed = False
    backward_passed = False
    optimizer_step_passed = False
    
    fwd_err = ""
    bwd_err = ""
    opt_err = ""
    
    # Get a single batch
    try:
        batch = next(iter(train_loader))
        img_feats = batch["image_features"].to(DEVICE)
        input_seqs = batch["input_sequence"].to(DEVICE)
        targets = batch["target_word"].to(DEVICE)
        targets = targets.squeeze(-1)  # (batch_size,)
    except Exception as e:
        logger.critical(f"Failed to fetch sample batch: {e}")
        return False
        
    model.train()
    optimizer.zero_grad()
    
    # 1. Forward Pass Test
    try:
        logits = model(img_feats, input_seqs)
        expected_shape = (img_feats.shape[0], model.vocab_size)
        if logits.shape == expected_shape:
            forward_passed = True
            logger.info(f"  [PASS] Forward test passed. Output shape: {list(logits.shape)}")
        else:
            fwd_err = f"Output shape mismatch: expected {expected_shape}, got {list(logits.shape)}"
            logger.error(f"  [FAIL] Forward test failed: {fwd_err}")
    except Exception as e:
        fwd_err = str(e)
        logger.error(f"  [FAIL] Forward pass crashed: {e}")
        
    # 2. Backward Pass Test
    if forward_passed:
        try:
            loss = criterion(logits, targets)
            loss.backward()
            
            # Check if gradient exists on image branch weight
            linear_layer = model.image_branch[0]
            if linear_layer.weight.grad is not None:
                grad_norm = linear_layer.weight.grad.norm().item()
                if grad_norm > 0:
                    backward_passed = True
                    logger.info(f"  [PASS] Backward test passed. Projected weights grad norm: {grad_norm:.6f}")
                else:
                    bwd_err = "Gradient exists but norm is zero."
                    logger.error(f"  [FAIL] Backward test failed: {bwd_err}")
            else:
                bwd_err = "Gradients are None."
                logger.error(f"  [FAIL] Backward test failed: {bwd_err}")
        except Exception as e:
            bwd_err = str(e)
            logger.error(f"  [FAIL] Backward pass crashed: {e}")
            
    # 3. Optimizer Step Test
    if backward_passed:
        try:
            linear_layer = model.image_branch[0]
            weight_before = linear_layer.weight.clone().detach()
            
            optimizer.step()
            
            weight_after = linear_layer.weight.clone().detach()
            weight_diff = torch.norm(weight_after - weight_before).item()
            
            if weight_diff > 0:
                optimizer_step_passed = True
                logger.info(f"  [PASS] Optimizer step test passed. Weights changed by norm: {weight_diff:.6f}")
            else:
                opt_err = "Weights did not change after optimizer step."
                logger.error(f"  [FAIL] Optimizer step test failed: {opt_err}")
        except Exception as e:
            opt_err = str(e)
            logger.error(f"  [FAIL] Optimizer step crashed: {e}")
            
    # Reset gradients after tests
    optimizer.zero_grad()
    
    # Write report file
    validation_report_path = OUTPUT_DIR / "training_pipeline_validation.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    overall_passed = forward_passed and backward_passed and optimizer_step_passed
    
    with open(validation_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("             Image Captioning Training Pipeline Validation\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"1. Forward Pass Test   : {'PASSED' if forward_passed else 'FAILED'}\n")
        if not forward_passed:
            f.write(f"   - Error: {fwd_err}\n")
            
        f.write(f"2. Backward Pass Test  : {'PASSED' if backward_passed else 'FAILED'}\n")
        if not backward_passed:
            f.write(f"   - Error: {bwd_err}\n")
            
        f.write(f"3. Optimizer Step Test : {'PASSED' if optimizer_step_passed else 'FAILED'}\n")
        if not optimizer_step_passed:
            f.write(f"   - Error: {opt_err}\n\n")
            
        f.write(f"OVERALL STATUS         : {'PASSED' if overall_passed else 'FAILED'}\n")
        if overall_passed:
            f.write("Verdict: All checks successfully passed. Training pipeline is ready.\n")
        else:
            f.write("Verdict: Validation checks failed. Do not run training loop.\n")
            
    logger.info(f"Validation report saved to: {validation_report_path}")
    return overall_passed

def main():
    parser = argparse.ArgumentParser(description="Image Captioning Training Pipeline Entrypoint")
    parser.add_argument("--epochs", type=int, default=2, help="Override epochs count")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Override batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Override learning rate")
    parser.add_argument("--train", action="store_true", help="Set this flag to run actual training")
    parser.add_argument("--resume", type=str, default="", help="Path to latest checkpoint to resume from")
    args = parser.parse_args()
    
    # Load Vocabulary Size dynamically
    w2i_path = OUTPUT_DIR / "word_to_index.pkl"
    if not w2i_path.exists():
        logger.critical(f"Vocabulary index not found at {w2i_path}. Please complete preprocessing first.")
        sys.exit(1)
        
    with open(w2i_path, "rb") as f:
        word_to_index = pickle.load(f)
    vocab_size = len(word_to_index)
    
    # Instantiate Model
    model = ImageCaptionModel(vocab_size=vocab_size)
    model.to(DEVICE)
    
    # Prepare Dataloaders
    train_loader, val_loader, stats = prepare_dataloaders(val_split=0.2, batch_size=args.batch_size, num_workers=0)
    
    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad_index is 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    
    # Run Verification Checks
    validation_passed = run_pipeline_validation(model, train_loader, optimizer, criterion)
    
    if not validation_passed:
        logger.critical("Pipeline validation checks failed. Training process aborted.")
        sys.exit(1)
        
    # Check if actual training requested
    if not args.train:
        logger.info("Pipeline verified successfully. Exiting without training as '--train' was not specified.")
        print("Training pipeline validation completed successfully. Model is training-ready.")
        sys.exit(0)
        
    # Paths setup for Trainer
    # Save checkpoints to PROJECT_ROOT / "checkpoints" as requested
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    history_path = OUTPUT_DIR / "training_history.csv"
    report_path = OUTPUT_DIR / "training_report.txt"
    
    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=DEVICE,
        checkpoint_dir=checkpoint_dir,
        history_path=history_path,
        report_path=report_path,
        epochs=args.epochs,
        patience=5,
        max_grad_norm=5.0,
        pad_idx=0
    )
    
    # Resume from checkpoint if requested
    if args.resume:
        trainer.resume_training(Path(args.resume))
        
    # Run training loop
    trainer.train()

if __name__ == "__main__":
    main()
