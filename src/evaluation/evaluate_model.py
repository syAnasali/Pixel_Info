import os
import sys
import time
import argparse
import csv
import logging
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import torch

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import DEVICE, OUTPUT_DIR
from src.model.image_caption_model import ImageCaptionModel
from src.training.create_dataloaders import load_preprocessed_data
from src.inference.caption_utils import greedy_decode, beam_search_decode
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_model")

def load_model(checkpoint_path: Path, vocab_size: int) -> ImageCaptionModel:
    model = ImageCaptionModel(vocab_size=vocab_size)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    logger.info(f"Loading model state dict from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Full Model Evaluation Pipeline")
    parser.add_argument("--model-path", type=str, default="", help="Path to checkpoint model weights")
    parser.add_argument("--beam-width", type=int, default=3, help="Beam width for beam search")
    parser.add_argument("--alpha", type=float, default=0.75, help="Length normalization parameter alpha")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.model_path) if args.model_path else PROJECT_ROOT / "checkpoints" / "best_model.pth"
    
    # Load vocabulary & preprocessed data
    try:
        word_to_index, seq_data, image_features_dict = load_preprocessed_data()
    except Exception as e:
        logger.critical(f"Failed to load preprocessed data: {e}")
        sys.exit(1)
        
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    vocab_size = len(word_to_index)
    
    # Load model
    try:
        model = load_model(checkpoint_path, vocab_size)
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        sys.exit(1)
        
    # Get validation split images deterministically
    image_names = seq_data["image_names"]
    unique_images = sorted(list(set(image_names)))
    
    import random
    random.seed(42)
    random.shuffle(unique_images)
    val_split = 0.2
    split_idx = int(len(unique_images) * (1 - val_split))
    val_images = sorted(list(unique_images[split_idx:]))
    
    logger.info(f"Total validation images to evaluate: {len(val_images)}")
    
    # Load human references
    image_to_references = defaultdict(list)
    cleaned_captions_path = OUTPUT_DIR / "cleaned_captions.csv"
    if not cleaned_captions_path.exists():
        logger.error(f"Cleaned captions file not found at: {cleaned_captions_path}")
        sys.exit(1)
        
    logger.info(f"Loading human reference captions from {cleaned_captions_path}...")
    with open(cleaned_captions_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row["image"]
            cap = row["cleaned_caption"]
            tokens = cap.lower().split()
            if tokens and tokens[0] == "startseq":
                tokens = tokens[1:]
            if tokens and tokens[-1] == "endseq":
                tokens = tokens[:-1]
            image_to_references[img].append(tokens)
            
    # Evaluation trackers
    greedy_bleu1_sum = 0.0
    greedy_bleu2_sum = 0.0
    greedy_bleu3_sum = 0.0
    greedy_bleu4_sum = 0.0
    greedy_len_sum = 0
    greedy_words = []
    
    beam_bleu1_sum = 0.0
    beam_bleu2_sum = 0.0
    beam_bleu3_sum = 0.0
    beam_bleu4_sum = 0.0
    beam_len_sum = 0
    beam_words = []
    
    total_evaluated = 0
    chencherry = SmoothingFunction()
    
    # Run evaluation loop
    logger.info("Starting full evaluation on all validation images...")
    start_time = time.time()
    
    # Use tqdm progress bar
    for image_name in tqdm(val_images, desc="Evaluating"):
        if image_name not in image_features_dict:
            logger.warning(f"Features missing for image {image_name}, skipping.")
            continue
            
        features = torch.tensor(image_features_dict[image_name], dtype=torch.float32)
        
        # Greedy decoding
        greedy_caption, _ = greedy_decode(
            model, features, word_to_index, index_to_word, max_len=38, device=DEVICE
        )
        greedy_tokens = greedy_caption.lower().split()
        
        # Beam search decoding
        beam_caption, _ = beam_search_decode(
            model, features, word_to_index, index_to_word,
            beam_width=args.beam_width, max_len=38, alpha=args.alpha, device=DEVICE
        )
        beam_tokens = beam_caption.lower().split()
        
        refs = image_to_references.get(image_name, [])
        if not refs:
            logger.warning(f"No reference captions found for {image_name}, skipping evaluation for this image.")
            continue
            
        # Calculate BLEU scores for Greedy
        g_b1 = sentence_bleu(refs, greedy_tokens, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=chencherry.method1)
        g_b2 = sentence_bleu(refs, greedy_tokens, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=chencherry.method1)
        g_b3 = sentence_bleu(refs, greedy_tokens, weights=(0.333, 0.333, 0.333, 0.0), smoothing_function=chencherry.method1)
        g_b4 = sentence_bleu(refs, greedy_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        
        # Calculate BLEU scores for Beam Search
        b_b1 = sentence_bleu(refs, beam_tokens, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=chencherry.method1)
        b_b2 = sentence_bleu(refs, beam_tokens, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=chencherry.method1)
        b_b3 = sentence_bleu(refs, beam_tokens, weights=(0.333, 0.333, 0.333, 0.0), smoothing_function=chencherry.method1)
        b_b4 = sentence_bleu(refs, beam_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        
        # Accumulate metrics
        greedy_bleu1_sum += g_b1
        greedy_bleu2_sum += g_b2
        greedy_bleu3_sum += g_b3
        greedy_bleu4_sum += g_b4
        greedy_len_sum += len(greedy_tokens)
        greedy_words.extend(greedy_tokens)
        
        beam_bleu1_sum += b_b1
        beam_bleu2_sum += b_b2
        beam_bleu3_sum += b_b3
        beam_bleu4_sum += b_b4
        beam_len_sum += len(beam_tokens)
        beam_words.extend(beam_tokens)
        
        total_evaluated += 1
        
    total_time = time.time() - start_time
    logger.info(f"Evaluation completed in {total_time:.2f} seconds.")
    
    if total_evaluated == 0:
        logger.error("No images were successfully evaluated.")
        sys.exit(1)
        
    # Calculate final averages
    avg_greedy_bleu1 = greedy_bleu1_sum / total_evaluated
    avg_greedy_bleu2 = greedy_bleu2_sum / total_evaluated
    avg_greedy_bleu3 = greedy_bleu3_sum / total_evaluated
    avg_greedy_bleu4 = greedy_bleu4_sum / total_evaluated
    avg_greedy_len = greedy_len_sum / total_evaluated
    greedy_vocab_diversity = len(set(greedy_words))
    
    avg_beam_bleu1 = beam_bleu1_sum / total_evaluated
    avg_beam_bleu2 = beam_bleu2_sum / total_evaluated
    avg_beam_bleu3 = beam_bleu3_sum / total_evaluated
    avg_beam_bleu4 = beam_bleu4_sum / total_evaluated
    avg_beam_len = beam_len_sum / total_evaluated
    beam_vocab_diversity = len(set(beam_words))
    
    # Calculate improvements & best strategies
    def compute_improvement(g_val, b_val):
        if g_val == 0:
            return 0.0
        return ((b_val - g_val) / g_val) * 100.0
        
    bleu1_imp = compute_improvement(avg_greedy_bleu1, avg_beam_bleu1)
    bleu2_imp = compute_improvement(avg_greedy_bleu2, avg_beam_bleu2)
    bleu3_imp = compute_improvement(avg_greedy_bleu3, avg_beam_bleu3)
    bleu4_imp = compute_improvement(avg_greedy_bleu4, avg_beam_bleu4)
    len_imp = compute_improvement(avg_greedy_len, avg_beam_len)
    vocab_imp = compute_improvement(greedy_vocab_diversity, beam_vocab_diversity)
    
    best_bleu1 = "Beam Search" if avg_beam_bleu1 > avg_greedy_bleu1 else "Greedy Decoding"
    best_bleu2 = "Beam Search" if avg_beam_bleu2 > avg_greedy_bleu2 else "Greedy Decoding"
    best_bleu3 = "Beam Search" if avg_beam_bleu3 > avg_greedy_bleu3 else "Greedy Decoding"
    best_bleu4 = "Beam Search" if avg_beam_bleu4 > avg_greedy_bleu4 else "Greedy Decoding"
    best_len = "Beam Search" if avg_beam_len > avg_greedy_len else "Greedy Decoding"
    best_vocab = "Beam Search" if beam_vocab_diversity > greedy_vocab_diversity else "Greedy Decoding"
    
    # Write outputs/evaluation_report.txt
    report_path = OUTPUT_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("        Pixel_Info Image Captioning Model Full Evaluation Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Evaluated Images      : {total_evaluated}\n")
        f.write(f"Evaluation Time             : {total_time:.2f} seconds\n")
        f.write(f"Device Used                 : {DEVICE}\n\n")
        
        f.write("1. Greedy Decoding Metrics:\n")
        f.write(f"  - BLEU-1                  : {avg_greedy_bleu1:.4f}\n")
        f.write(f"  - BLEU-2                  : {avg_greedy_bleu2:.4f}\n")
        f.write(f"  - BLEU-3                  : {avg_greedy_bleu3:.4f}\n")
        f.write(f"  - BLEU-4                  : {avg_greedy_bleu4:.4f}\n")
        f.write(f"  - Average Caption Length  : {avg_greedy_len:.2f} words\n")
        f.write(f"  - Vocabulary Diversity    : {greedy_vocab_diversity} unique words\n\n")
        
        f.write(f"2. Beam Search (beam_width={args.beam_width}, alpha={args.alpha}) Metrics:\n")
        f.write(f"  - BLEU-1                  : {avg_beam_bleu1:.4f}\n")
        f.write(f"  - BLEU-2                  : {avg_beam_bleu2:.4f}\n")
        f.write(f"  - BLEU-3                  : {avg_beam_bleu3:.4f}\n")
        f.write(f"  - BLEU-4                  : {avg_beam_bleu4:.4f}\n")
        f.write(f"  - Average Caption Length  : {avg_beam_len:.2f} words\n")
        f.write(f"  - Vocabulary Diversity    : {beam_vocab_diversity} unique words\n\n")
        
        f.write("3. Comparison and Analysis:\n")
        f.write(f"  - BLEU-1 Improvement      : {bleu1_imp:+.2f}% (Best: {best_bleu1})\n")
        f.write(f"  - BLEU-2 Improvement      : {bleu2_imp:+.2f}% (Best: {best_bleu2})\n")
        f.write(f"  - BLEU-3 Improvement      : {bleu3_imp:+.2f}% (Best: {best_bleu3})\n")
        f.write(f"  - BLEU-4 Improvement      : {bleu4_imp:+.2f}% (Best: {best_bleu4})\n")
        f.write(f"  - Length Change           : {len_imp:+.2f}% (Best: {best_len})\n")
        f.write(f"  - Vocab Diversity Change  : {vocab_imp:+.2f}% (Best: {best_vocab})\n\n")
        
        overall_best = "Beam Search" if (avg_beam_bleu4 > avg_greedy_bleu4) else "Greedy Decoding"
        f.write(f"Overall Best Strategy (based on BLEU-4): {overall_best}\n")
        f.write("============================================================\n")
        
    logger.info(f"Saved evaluation report to: {report_path}")
    
    # Write outputs/evaluation_summary.csv
    summary_path = OUTPUT_DIR / "evaluation_summary.csv"
    summary_data = [
        {"Metric": "BLEU-1", "Greedy Value": f"{avg_greedy_bleu1:.6f}", "Beam Value": f"{avg_beam_bleu1:.6f}", "Improvement (%)": f"{bleu1_imp:+.2f}%", "Best Strategy": best_bleu1},
        {"Metric": "BLEU-2", "Greedy Value": f"{avg_greedy_bleu2:.6f}", "Beam Value": f"{avg_beam_bleu2:.6f}", "Improvement (%)": f"{bleu2_imp:+.2f}%", "Best Strategy": best_bleu2},
        {"Metric": "BLEU-3", "Greedy Value": f"{avg_greedy_bleu3:.6f}", "Beam Value": f"{avg_beam_bleu3:.6f}", "Improvement (%)": f"{bleu3_imp:+.2f}%", "Best Strategy": best_bleu3},
        {"Metric": "BLEU-4", "Greedy Value": f"{avg_greedy_bleu4:.6f}", "Beam Value": f"{avg_beam_bleu4:.6f}", "Improvement (%)": f"{bleu4_imp:+.2f}%", "Best Strategy": best_bleu4},
        {"Metric": "Average Caption Length", "Greedy Value": f"{avg_greedy_len:.2f}", "Beam Value": f"{avg_beam_len:.2f}", "Improvement (%)": f"{len_imp:+.2f}%", "Best Strategy": best_len},
        {"Metric": "Vocabulary Diversity", "Greedy Value": str(greedy_vocab_diversity), "Beam Value": str(beam_vocab_diversity), "Improvement (%)": f"{vocab_imp:+.2f}%", "Best Strategy": best_vocab}
    ]
    
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Greedy Value", "Beam Value", "Improvement (%)", "Best Strategy"])
        writer.writeheader()
        for row in summary_data:
            writer.writerow(row)
            
    logger.info(f"Saved evaluation summary to: {summary_path}")
    print("\nFull model evaluation pipeline completed successfully.")

if __name__ == "__main__":
    main()
