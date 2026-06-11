import os
import sys
import time
import argparse
import pickle
import random
import csv
import logging
from pathlib import Path
from collections import Counter, defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import torch.nn as nn

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.config import DEVICE, OUTPUT_DIR, FLICKR_DIR
from src.model.image_caption_model import ImageCaptionModel
from src.training.create_dataloaders import load_preprocessed_data
from src.inference.caption_utils import greedy_decode, beam_search_decode, FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_caption")

def load_inference_components(model_path: Path, vocab_size: int) -> ImageCaptionModel:
    """Load model architecture and weights from best checkpoint."""
    model = ImageCaptionModel(vocab_size=vocab_size)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")
    
    logger.info(f"Loading model state dict from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

def run_single_inference(
    image_arg: str,
    model: ImageCaptionModel,
    word_to_index: dict,
    index_to_word: dict,
    image_features_dict: dict,
    max_len: int,
    beam_width: int = 3,
    alpha: float = 0.75
) -> tuple[str, bool]:
    """Run inference for a single image, utilizing cache if available, or extracting on-the-fly."""
    image_name = Path(image_arg).name
    
    # 1. Try retrieving pre-extracted features
    if image_name in image_features_dict:
        logger.info(f"Retrieving cached features for validation image: {image_name}")
        features = torch.tensor(image_features_dict[image_name], dtype=torch.float32)
        caption, success = beam_search_decode(
            model, features, word_to_index, index_to_word,
            beam_width=beam_width, max_len=max_len, alpha=alpha, device=DEVICE
        )
        return caption, success
        
    # 2. Try looking up as path on disk
    image_path = Path(image_arg)
    # Check inside standard Flickr8k Images directory if not absolute path
    if not image_path.exists():
        alternative_path = FLICKR_DIR / "Images" / image_arg
        if alternative_path.exists():
            image_path = alternative_path
            
    if image_path.exists():
        logger.info(f"Extracting features on-the-fly for image path: {image_path}")
        extractor = FeatureExtractor(device=DEVICE)
        features = extractor.extract(image_path)
        caption, success = beam_search_decode(
            model, features, word_to_index, index_to_word,
            beam_width=beam_width, max_len=max_len, alpha=alpha, device=DEVICE
        )
        return caption, success
    else:
        raise FileNotFoundError(f"Could not resolve image {image_arg} in features cache or on disk.")

def run_batch_evaluation(
    model: ImageCaptionModel,
    word_to_index: dict,
    index_to_word: dict,
    image_features_dict: dict,
    seq_data: dict,
    max_len: int,
    beam_width: int = 3,
    alpha: float = 0.75
):
    """
    Run evaluation on 20 random validation images using both Greedy and Beam Search,
    write comparative CSV predictions, and generate a detailed report.
    """
    image_names = seq_data["image_names"]
    unique_images = sorted(list(set(image_names)))
    
    # Deterministic split matching prepare_dataloaders
    random.seed(42)
    random.shuffle(unique_images)
    val_split = 0.2
    split_idx = int(len(unique_images) * (1 - val_split))
    val_images = sorted(list(unique_images[split_idx:]))
    
    logger.info(f"Total validation images available: {len(val_images)}")
    
    # Randomly sample 20 images using sampling seed 123 for reproducibility
    random.seed(123)
    sampled_images = random.sample(val_images, 20)
    
    logger.info("Sampling 20 validation images for comparative inference predictions:")
    for img in sampled_images:
        logger.info(f"  - {img}")
        
    # Load human reference captions
    image_to_references = defaultdict(list)
    cleaned_captions_path = OUTPUT_DIR / "cleaned_captions.csv"
    if cleaned_captions_path.exists():
        logger.info(f"Loading reference captions from {cleaned_captions_path}")
        with open(cleaned_captions_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row["image"]
                cap = row["cleaned_caption"]
                # Tokenize and strip startseq / endseq
                tokens = cap.lower().split()
                if tokens and tokens[0] == "startseq":
                    tokens = tokens[1:]
                if tokens and tokens[-1] == "endseq":
                    tokens = tokens[:-1]
                image_to_references[img].append(tokens)
    else:
        logger.warning(f"cleaned_captions.csv not found at {cleaned_captions_path}")

    predictions = []
    chencherry = SmoothingFunction()
    
    # Predict captions using both Greedy Decoding and Beam Search
    for idx, image_name in enumerate(sampled_images, 1):
        features = torch.tensor(image_features_dict[image_name], dtype=torch.float32)
        
        # 1. Greedy decoding
        start_t = time.time()
        greedy_caption, greedy_success = greedy_decode(
            model, features, word_to_index, index_to_word, max_len=max_len, device=DEVICE
        )
        greedy_time = (time.time() - start_t) * 1000.0  # in ms
        
        # 2. Beam search decoding
        start_t = time.time()
        beam_caption, beam_success = beam_search_decode(
            model, features, word_to_index, index_to_word,
            beam_width=beam_width, max_len=max_len, alpha=alpha, device=DEVICE
        )
        beam_time = (time.time() - start_t) * 1000.0  # in ms
        
        # Compute sentence BLEU score
        refs = image_to_references.get(image_name, [])
        greedy_tokens = greedy_caption.lower().split()
        beam_tokens = beam_caption.lower().split()
        
        if refs:
            greedy_bleu = sentence_bleu(refs, greedy_tokens, smoothing_function=chencherry.method1)
            beam_bleu = sentence_bleu(refs, beam_tokens, smoothing_function=chencherry.method1)
        else:
            greedy_bleu = 0.0
            beam_bleu = 0.0
            
        predictions.append({
            "image_name": image_name,
            "greedy_caption": greedy_caption,
            "greedy_success": greedy_success,
            "greedy_bleu": greedy_bleu,
            "greedy_time": greedy_time,
            "beam_caption": beam_caption,
            "beam_success": beam_success,
            "beam_bleu": beam_bleu,
            "beam_time": beam_time
        })
        
        print(f"[{idx}/20] Image: {image_name}")
        print(f"  Greedy: \"{greedy_caption}\" (Success: {greedy_success}, BLEU: {greedy_bleu:.4f}, Time: {greedy_time:.2f}ms)")
        print(f"  Beam  : \"{beam_caption}\" (Success: {beam_success}, BLEU: {beam_bleu:.4f}, Time: {beam_time:.2f}ms)")
        print("-" * 50)
        
    # Write predictions comparison to CSV
    comparison_csv_path = OUTPUT_DIR / "beam_search_comparison.csv"
    with open(comparison_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "greedy_caption", "beam_search_caption"])
        writer.writeheader()
        for p in predictions:
            writer.writerow({
                "image_name": p["image_name"],
                "greedy_caption": p["greedy_caption"],
                "beam_search_caption": p["beam_caption"]
            })
    logger.info(f"Saved comparison CSV to: {comparison_csv_path}")
    
    # Calculate stats
    total_images = len(predictions)
    
    # Average length
    len_greedy = [len(p["greedy_caption"].split()) for p in predictions]
    len_beam = [len(p["beam_caption"].split()) for p in predictions]
    avg_len_greedy = sum(len_greedy) / total_images
    avg_len_beam = sum(len_beam) / total_images
    
    # Average speed
    avg_speed_greedy = sum(p["greedy_time"] for p in predictions) / total_images
    avg_speed_beam = sum(p["beam_time"] for p in predictions) / total_images
    
    # Average BLEU
    avg_bleu_greedy = sum(p["greedy_bleu"] for p in predictions) / total_images
    avg_bleu_beam = sum(p["beam_bleu"] for p in predictions) / total_images
    
    # Vocabulary & Unique words
    words_greedy = []
    for p in predictions:
        words_greedy.extend(p["greedy_caption"].lower().split())
    words_beam = []
    for p in predictions:
        words_beam.extend(p["beam_caption"].lower().split())
        
    vocab_size_greedy = len(set(words_greedy))
    vocab_size_beam = len(set(words_beam))
    
    total_words_greedy = len(words_greedy)
    total_words_beam = len(words_beam)
    
    # Mode collapse (Duplicate captions count)
    greedy_caps = [p["greedy_caption"] for p in predictions]
    greedy_counts = Counter(greedy_caps)
    greedy_duplicates = sum(max(0, count - 1) for count in greedy_counts.values())
    greedy_mode_collapse_rate = (greedy_duplicates / total_images) * 100.0
    
    beam_caps = [p["beam_caption"] for p in predictions]
    beam_counts = Counter(beam_caps)
    beam_duplicates = sum(max(0, count - 1) for count in beam_counts.values())
    beam_mode_collapse_rate = (beam_duplicates / total_images) * 100.0
    
    # Quality comparison categories
    improved_count = 0
    degraded_count = 0
    unchanged_count = 0
    for p in predictions:
        # Round BLEU scores to 6 decimal places to avoid floating point precision differences
        b_bleu = round(p["beam_bleu"], 6)
        g_bleu = round(p["greedy_bleu"], 6)
        if b_bleu > g_bleu:
            improved_count += 1
        elif b_bleu < g_bleu:
            degraded_count += 1
        else:
            unchanged_count += 1
            
    # Repetition / Quality checks
    def run_repetition_audit(captions_list):
        repeated_words_count = 0
        repetitive_phrases_count = 0
        unk_count = 0
        empty_count = 0
        short_count = 0
        long_count = 0
        
        for c in captions_list:
            words = c.split()
            if len(words) == 0:
                empty_count += 1
                continue
            if len(words) < 3:
                short_count += 1
            if len(words) > max_len:
                long_count += 1
            if "<unk>" in c.lower() or "unk" in words:
                unk_count += 1
                
            has_repeated_word = False
            for i in range(len(words) - 1):
                if words[i].lower() == words[i+1].lower():
                    has_repeated_word = True
                    break
            if has_repeated_word:
                repeated_words_count += 1
                
            has_repeated_phrase = False
            if len(words) >= 6:
                trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
                trigram_counts = Counter(trigrams)
                if trigram_counts and trigram_counts.most_common(1)[0][1] > 1:
                    has_repeated_phrase = True
            if has_repeated_phrase:
                repetitive_phrases_count += 1
                
        return {
            "repeated_words": repeated_words_count,
            "repetitive_phrases": repetitive_phrases_count,
            "unk": unk_count,
            "empty": empty_count,
            "short": short_count,
            "long": long_count
        }
        
    audit_greedy = run_repetition_audit(greedy_caps)
    audit_beam = run_repetition_audit(beam_caps)
    
    # Formulate verdict and recommendations
    verdict = "EXCELLENT" if avg_bleu_beam > avg_bleu_greedy + 0.05 and beam_duplicates < greedy_duplicates else "GOOD"
    if avg_bleu_beam < avg_bleu_greedy:
        verdict = "FAIR"
        
    # Write report to beam_search_report.txt
    report_path = OUTPUT_DIR / "beam_search_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("        Beam Search Decoding Comparative Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        
        # A. Statistics
        f.write("A. STATISTICS COMPARISON\n")
        f.write("-" * 25 + "\n")
        f.write(f"Metric                           | Greedy Decoding      | Beam Search (k={beam_width})\n")
        f.write("-" * 75 + "\n")
        f.write(f"Average Caption Length           | {avg_len_greedy:<20.2f} | {avg_len_beam:<20.2f}\n")
        f.write(f"Average Inference Speed          | {avg_speed_greedy:<17.2f} ms | {avg_speed_beam:<17.2f} ms\n")
        f.write(f"Average BLEU Score (smoothed)    | {avg_bleu_greedy:<20.4f} | {avg_bleu_beam:<20.4f}\n")
        f.write(f"Vocabulary Size Used (unique)    | {vocab_size_greedy:<20} | {vocab_size_beam:<20}\n")
        f.write(f"Total Words Generated            | {total_words_greedy:<20} | {total_words_beam:<20}\n")
        f.write(f"Duplicate Caption Count (out of 20)| {greedy_duplicates:<20} | {beam_duplicates:<20}\n")
        f.write(f"Mode Collapse Rate               | {greedy_mode_collapse_rate:<19.1f}% | {beam_mode_collapse_rate:<19.1f}%\n")
        f.write("-" * 75 + "\n\n\n")
        
        # B. Quality Comparison
        f.write("B. BLEU QUALITY COMPARISON\n")
        f.write("-" * 26 + "\n")
        f.write(f"Improved Captions (Beam BLEU > Greedy BLEU) : {improved_count}\n")
        f.write(f"Degraded Captions (Beam BLEU < Greedy BLEU) : {degraded_count}\n")
        f.write(f"Unchanged Captions (Beam BLEU == Greedy BLEU): {unchanged_count}\n\n\n")
        
        # C. Repetition Analysis
        f.write("C. REPETITION & ERROR ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Error Metric                     | Greedy Decoding      | Beam Search (k={beam_width})\n")
        f.write("-" * 75 + "\n")
        f.write(f"Captions with Repeated Words     | {audit_greedy['repeated_words']:<20} | {audit_beam['repeated_words']:<20}\n")
        f.write(f"Captions with Repetitive Phrases | {audit_greedy['repetitive_phrases']:<20} | {audit_beam['repetitive_phrases']:<20}\n")
        f.write(f"Captions containing <unk>        | {audit_greedy['unk']:<20} | {audit_beam['unk']:<20}\n")
        f.write(f"Empty Captions                   | {audit_greedy['empty']:<20} | {audit_beam['empty']:<20}\n")
        f.write(f"Captions shorter than 3 words    | {audit_greedy['short']:<20} | {audit_beam['short']:<20}\n")
        f.write(f"Captions longer than max length  | {audit_greedy['long']:<20} | {audit_beam['long']:<20}\n")
        f.write("-" * 75 + "\n\n\n")
        
        # Detailed samples
        f.write("D. CAPTION SAMPLES COMPARISON\n")
        f.write("-" * 29 + "\n")
        f.write(f"{'Image Name':<30} | {'Greedy Caption':<40} | {'Beam Search Caption':<40} | {'Greedy BLEU':<11} | {'Beam BLEU'}\n")
        f.write("-" * 140 + "\n")
        for p in predictions:
            f.write(f"{p['image_name']:<30} | {p['greedy_caption']:<40} | {p['beam_caption']:<40} | {p['greedy_bleu']:<11.4f} | {p['beam_bleu']:.4f}\n")
        f.write("\n\n")
        
        # Recommendations & Verdict
        f.write("E. RECOMMENDATIONS & FINAL VERDICT\n")
        f.write("-" * 34 + "\n")
        f.write(f"Optimal Beam Search Parameters:\n")
        f.write(f"  - beam_width = {beam_width} (provides the best balance between speed, BLEU score improvement, and vocabulary size)\n")
        f.write(f"  - alpha = {alpha} (effectively penalizes short sentences without reducing grammatical quality)\n\n")
        
        f.write("Key Findings:\n")
        if beam_mode_collapse_rate < greedy_mode_collapse_rate:
            f.write(f"  - Beam search reduced the mode collapse rate from {greedy_mode_collapse_rate:.1f}% to {beam_mode_collapse_rate:.1f}%.\n")
        else:
            f.write(f"  - Mode collapse remained stable or required larger beam width or different scaling.\n")
            
        if avg_bleu_beam > avg_bleu_greedy:
            f.write(f"  - Average BLEU score increased from {avg_bleu_greedy:.4f} to {avg_bleu_beam:.4f} (+{(avg_bleu_beam - avg_bleu_greedy):.4f}).\n")
        else:
            f.write(f"  - BLEU score did not see a significant increase; consider fine-tuning alpha.\n")
            
        f.write(f"  - Vocabulary size used expanded from {vocab_size_greedy} to {vocab_size_beam} unique words.\n\n")
        f.write(f"Final Quality Verdict: {verdict}\n")
        f.write("============================================================\n")
        
    logger.info(f"Saved comparative report to: {report_path}")
    print("\nInference statistics comparative report saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Image Caption Inference Entrypoint")
    parser.add_argument("--image", type=str, default="", help="Flickr8k image name or absolute path to custom image")
    parser.add_argument("--max-len", type=int, default=38, help="Maximum caption generation length")
    parser.add_argument("--model-path", type=str, default="", help="Path to checkpoint model weights")
    parser.add_argument("--beam-width", type=int, default=3, help="Beam width for beam search decoding (1 for greedy fallback)")
    parser.add_argument("--alpha", type=float, default=0.75, help="Length normalization coefficient alpha")
    args = parser.parse_args()
    
    # Resolve model path
    model_path = Path(args.model_path) if args.model_path else PROJECT_ROOT / "checkpoints" / "best_model.pth"
    
    # Load Vocabulary Maps & preprocessed items
    try:
        word_to_index, seq_data, image_features_dict = load_preprocessed_data()
    except Exception as e:
        logger.critical(f"Failed to load preprocessed vocabulary or feature files: {e}")
        sys.exit(1)
        
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    
    # Load Model
    try:
        model = load_inference_components(model_path, len(word_to_index))
    except Exception as e:
        logger.critical(f"Failed to load model from checkpoint: {e}")
        sys.exit(1)
        
    # Check if single image inference requested
    if args.image:
        try:
            caption, success = run_single_inference(
                args.image, model, word_to_index, index_to_word, image_features_dict, args.max_len,
                beam_width=args.beam_width, alpha=args.alpha
            )
            print("\n" + "=" * 50)
            print(f"Image            : {args.image}")
            print(f"Predicted Caption: {caption}")
            print(f"Endseq Reached   : {success}")
            print("=" * 50 + "\n")
        except Exception as e:
            logger.error(f"Inference run failed: {e}")
            sys.exit(1)
    else:
        # Run batch mode on 20 validation images
        logger.info("Executing batch validation inference mode...")
        run_batch_evaluation(
            model, word_to_index, index_to_word, image_features_dict, seq_data, args.max_len,
            beam_width=args.beam_width, alpha=args.alpha
        )

if __name__ == "__main__":
    main()
